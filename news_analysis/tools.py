"""Инструменты аналитика влияния.

Все инструменты привязаны к одной анализируемой компании: модель не может запросить
профиль или платежи произвольного клиента, только карточки его контрагентов и группы.
Каждый вызов получает номер T1, T2, … — на него модель ссылается в evidence.
"""
import json
import time
from collections import Counter
from typing import Any, Callable

from django.db.models import Sum

from companies.models import Company, Payment

from . import bank_search
from .llm import Trace, dumps, tool_spec

BN = 1_000_000_000
MAX_RESULT_CHARS = 7000
PROFILE_SECTIONS = ["scale", "industry", "legal", "geography", "foreign_trade", "bank_relationship",
                    "group", "counterparties", "related_individuals", "sensitivities", "data_quality"]
CLIENT_STATUS = {"CLIENT": "действующий клиент", "CLOSE_CLIENT": "бывший клиент", "PROSPECT": "не клиент банка"}


def fmt_rub(amount: float | int | None) -> str:
    if not amount:
        return "0 ₽"
    if amount >= BN:
        return f"{amount / BN:.1f} млрд ₽".replace(".", ",")
    if amount >= 1_000_000:
        return f"{amount / 1_000_000:.1f} млн ₽".replace(".", ",")
    return f"{amount / 1000:.0f} тыс. ₽"


def stem(word: str) -> str:
    word = word.lower().replace("ё", "е").strip()
    return word[:6] if len(word) > 7 else word


TOOL_SPECS = [
    tool_spec(
        "get_company_section",
        "Раздел полного профиля анализируемой компании. Разделы: scale (масштаб), industry (отрасль, "
        "сырьё, драйверы спроса, регулирование), legal (собственность, возраст), geography, foreign_trade "
        "(страны ВЭД, валюты), bank_relationship (продукты Альфа-Банка), group (холдинг), counterparties "
        "(сводка по платежам и топ-5 контрагентов), related_individuals, sensitivities (теги чувствительности), "
        "data_quality (пробелы в данных).",
        {"type": "object", "properties": {"section": {"type": "string", "enum": PROFILE_SECTIONS}},
         "required": ["section"]},
    ),
    tool_spec(
        "get_counterparties",
        "Крупнейшие контрагенты анализируемой компании по платежам через банк: suppliers — кому компания "
        "платит, customers — кто платит компании. Для каждого: доля в объёме, сумма, отрасль, регион, выручка.",
        {"type": "object", "properties": {
            "role": {"type": "string", "enum": ["suppliers", "customers"]},
            "top_n": {"type": "integer", "minimum": 1, "maximum": 20}},
         "required": ["role", "top_n"]},
    ),
    tool_spec(
        "find_counterparties_by_keywords",
        "Ищет среди всех контрагентов компании тех, чья отрасль (вид деятельности по ОКВЭД) содержит "
        "ключевые слова, например ['программ', 'информац', 'связ'] или ['топлив', 'нефт']. Возвращает "
        "найденных контрагентов и их суммарную долю в платежах. Помогает понять, зависит ли компания "
        "от отрасли, которой касается новость.",
        {"type": "object", "properties": {
            "keywords": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 8},
            "role": {"type": "string", "enum": ["suppliers", "customers", "both"]}},
         "required": ["keywords", "role"]},
    ),
    tool_spec(
        "get_company_card",
        "Краткая карточка другой компании из датасета (обычно контрагента): отрасль, выручка, регион, "
        "статус в банке, ВЭД, а также её доля в платежах анализируемой компании.",
        {"type": "object", "properties": {"company_id": {"type": "string"}}, "required": ["company_id"]},
    ),
    tool_spec(
        "get_group_companies",
        "Компании той же группы (холдинга), что и анализируемая: их число, отрасли и крупнейшие участники.",
        {"type": "object", "properties": {}, "required": []},
    ),
    tool_spec(
        "search_bank_analytics",
        "Поиск по аналитическим материалам Альфа-Банка (макроэкономика, ставка, инфляция, долговой рынок, "
        "потребление, сырьё). Поиск по словам, не по смыслу: формулируй 2–5 ключевых слов по-русски. "
        "Возвращает фрагменты с id, документом, датой и страницей. Фрагменты могут оказаться не по теме.",
        {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    ),
]


class CompanyTools:
    def __init__(self, company: Company, news_date: str | None, trace: Trace):
        self.company = company
        self.profile = company.profile or {}
        self.news_date = news_date
        self.trace = trace
        self.calls: dict[str, dict] = {}  # T-номер -> {name, arguments, result}
        self.seen_bank_fragments: dict[str, dict] = {}
        self._payments: dict[str, list[dict]] | None = None
        self.handlers: dict[str, Callable[..., Any]] = {
            "get_company_section": self.get_company_section,
            "get_counterparties": self.get_counterparties,
            "find_counterparties_by_keywords": self.find_counterparties_by_keywords,
            "get_company_card": self.get_company_card,
            "get_group_companies": self.get_group_companies,
            "search_bank_analytics": self.search_bank_analytics,
        }

    # --- Выполнение ----------------------------------------------------------------------

    def execute(self, name: str, raw_arguments: str) -> tuple[str, str]:
        """Выполняет вызов модели. Возвращает (T-номер, текст результата для модели)."""
        call_id = f"T{len(self.calls) + 1}"
        started = time.monotonic()
        try:
            arguments = json.loads(raw_arguments or "{}")
            handler = self.handlers[name]
            result = handler(**arguments)
            error = None
        except Exception as exc:  # ошибку отдаём модели, а не роняем прогон
            arguments, result, error = raw_arguments, None, f"{type(exc).__name__}: {exc}"
        self.calls[call_id] = {"name": name, "arguments": arguments, "result": result, "error": error}
        self.trace.add(kind="tool", step="tool", call_id=call_id, name=name, arguments=arguments,
                       latency_s=round(time.monotonic() - started, 3), error=error,
                       result_preview=dumps(result)[:300] if result is not None else None)
        body = dumps({"error": error}) if error else dumps(result)
        if len(body) > MAX_RESULT_CHARS:
            body = body[:MAX_RESULT_CHARS] + "…(обрезано)"
        return call_id, f"[{call_id}] {name}({dumps(arguments)})\n{body}"

    # --- Инструменты ---------------------------------------------------------------------

    def get_company_section(self, section: str) -> dict:
        if section not in PROFILE_SECTIONS:
            raise ValueError(f"Нет раздела {section}; доступны: {', '.join(PROFILE_SECTIONS)}")
        return {"section": section, "path": f"/{section}", "data": self.profile.get(section)}

    def get_counterparties(self, role: str, top_n: int = 10) -> dict:
        rows = self._load_payments()[role]
        total = sum(r["amount"] for r in rows)
        top = rows[:max(1, min(top_n, 20))]
        return {
            "role": role,
            "meaning": "кому компания платит" if role == "suppliers" else "кто платит компании",
            "counterparties_total": len(rows),
            "volume_total": fmt_rub(total),
            "top": [self._card_line(r, total) for r in top],
            "note": "Платежи через Альфа-Банк за период датасета; отрасль неизвестна, если контрагента нет в датасете.",
        }

    def find_counterparties_by_keywords(self, keywords: list[str], role: str = "both") -> dict:
        stems = [stem(k) for k in keywords if k.strip()]
        roles = ["suppliers", "customers"] if role == "both" else [role]
        result = {"keywords": keywords}
        for r in roles:
            rows = self._load_payments()[r]
            total = sum(x["amount"] for x in rows)
            matched = [x for x in rows if x["okved_name"] and any(s in x["okved_name"].lower() for s in stems)]
            amount = sum(x["amount"] for x in matched)
            unknown = sum(x["amount"] for x in rows if not x["in_dataset"])
            result[r] = {
                "matched_count": len(matched),
                "matched_share": round(amount / total, 3) if total else 0,
                "matched_volume": fmt_rub(amount),
                "unknown_sector_share": round(unknown / total, 3) if total else 0,
                "matched": [self._card_line(x, total) for x in matched[:10]],
            }
        result["note"] = ("unknown_sector_share — доля платежей контрагентам, которых нет в датасете: "
                          "их отрасль неизвестна, поэтому реальная доля может быть выше найденной.")
        return result

    def get_company_card(self, company_id: str) -> dict:
        try:
            c = Company.objects.get(pk=company_id)
        except Company.DoesNotExist:
            return {"company_id": company_id, "found": False, "note": "компании нет в датасете"}
        payments = self._load_payments()
        relation = {}
        for role in ("suppliers", "customers"):
            total = sum(x["amount"] for x in payments[role])
            row = next((x for x in payments[role] if x["id"] == company_id), None)
            if row:
                relation[role] = {"share": round(row["amount"] / total, 3) if total else 0,
                                  "volume": fmt_rub(row["amount"])}
        return {
            "company_id": c.company_id, "found": True, "okved": f"{c.okved_code} {c.okved_name}",
            "sector": c.sector, "revenue": fmt_rub(c.revenue), "employees": c.employees_fns,
            "region": c.region, "bank_status": CLIENT_STATUS.get(c.client_type, c.client_type),
            "foreign_trade_countries": c.trade_countries or None,
            "same_group": bool(c.group_id) and c.group_id == self.company.group_id,
            "relation_to_analyzed_company": relation or "нет прямых платежей",
        }

    def get_group_companies(self) -> dict:
        members = Company.objects.filter(group_id=self.company.group_id).exclude(pk=self.company.pk)
        count = members.count()
        if not count:
            return {"group_size": 1, "note": "компания не входит в группу"}
        sectors = Counter(members.values_list("sector", flat=True))
        top = members.order_by("-revenue")[:10]
        return {
            "group_size": count + 1,
            "total_revenue": fmt_rub(sum(members.values_list("revenue", flat=True)) + self.company.revenue),
            "sectors": [{"sector": s, "companies": n} for s, n in sectors.most_common(6)],
            "largest_members": [{"company_id": m.company_id, "okved": m.okved_name, "revenue": fmt_rub(m.revenue),
                                 "region": m.region} for m in top],
        }

    def search_bank_analytics(self, query: str) -> dict:
        hits = bank_search.search(query, top_k=4, as_of=self.news_date)
        for h in hits:
            self.seen_bank_fragments[h["id"]] = h
        return {"query": query, "as_of": self.news_date, "fragments": hits,
                "note": "Это аналитика Альфа-Банка. Используй фрагмент, только если он по теме новости."}

    # --- Данные о платежах -------------------------------------------------------------

    def _load_payments(self) -> dict[str, list[dict]]:
        """Агрегирует платежи компании один раз на прогон; самоплатежи исключены."""
        if self._payments is not None:
            return self._payments
        cid = self.company.company_id
        out = (Payment.objects.filter(payer_id=cid).exclude(beneficiary_id=cid)
               .values("beneficiary_id").annotate(amount=Sum("amount"), count=Sum("count")))
        inc = (Payment.objects.filter(beneficiary_id=cid).exclude(payer_id=cid)
               .values("payer_id").annotate(amount=Sum("amount"), count=Sum("count")))
        raw = {
            "suppliers": [(r["beneficiary_id"], r["amount"], r["count"]) for r in out],
            "customers": [(r["payer_id"], r["amount"], r["count"]) for r in inc],
        }
        ids = {cp for rows in raw.values() for cp, _, _ in rows}
        info = Company.objects.in_bulk(list(ids))
        self._payments = {}
        for role, rows in raw.items():
            enriched = []
            for cp, amount, count in rows:
                c = info.get(cp)
                enriched.append({
                    "id": cp, "amount": amount, "count": count, "in_dataset": c is not None,
                    "okved_name": c.okved_name if c else None, "sector": c.sector if c else None,
                    "region": c.region if c else None, "revenue": c.revenue if c else None,
                    "same_group": bool(c and c.group_id and c.group_id == self.company.group_id),
                })
            self._payments[role] = sorted(enriched, key=lambda x: -x["amount"])
        return self._payments

    @staticmethod
    def _card_line(row: dict, total: int) -> dict:
        line = {"company_id": row["id"], "share": round(row["amount"] / total, 3) if total else 0,
                "volume": fmt_rub(row["amount"]), "payments": row["count"]}
        if row["in_dataset"]:
            line.update({"okved": row["okved_name"], "sector": row["sector"], "region": row["region"],
                         "revenue": fmt_rub(row["revenue"]), "same_group": row["same_group"]})
        else:
            line["okved"] = "нет данных (компании нет в датасете)"
        return line
