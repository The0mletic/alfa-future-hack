"""Перевод результата анализа в формат, который ожидают шаблоны (base/mock_data.py).

Сверх контракта mock_data добавлены поля, которых требует кейс: источник каждого вывода,
цепочки причин, действия, вопросы для проверки, уверенность и ограничения.
"""
import re
from datetime import datetime

from .schemas import ImpactAnalysis, NewsAnalysis

CATEGORY_LABELS = {
    "macro": "Макроэкономика", "markets": "Рынки", "commodities": "Сырьё", "regulation": "Регулирование",
    "technology": "Технологии", "industry": "Отрасли и компании", "other": "Другое",
}
MONTHS = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября",
          "октября", "ноября", "декабря"]
SOURCE_LABELS = {
    "news": "из новости", "company_profile": "по данным о компании", "bank_analytics": "аналитика Альфа-Банка",
    "bank_comment_in_news": "комментарий Альфа-Банка", "model_inference": "вывод AI",
}
STATUS_PREFIX = {"opinion": "Оценка", "forecast": "Прогноз", "claimed": "Заявление"}


REF_IN_PARENS = re.compile(r"\s*\((?:[FTCQ]\d{1,2}(?:\s*(?:,|;|и)\s*)?)+\)")
REF_BEFORE_PAREN = re.compile(r"[,;]\s*[FTCQ]\d{1,2}(?=\))")


def clean_refs(text: str | None) -> str | None:
    """Убирает служебные ссылки вида «(F4)», «(F7, F12)», «…, T2)» из текста для читателя."""
    if not text:
        return text
    return REF_BEFORE_PAREN.sub("", REF_IN_PARENS.sub("", text))


def short_tag(text: str, limit: int = 40) -> str:
    """Бейджи «Вывод основан на» не переносятся — длинная метка ломает вёрстку."""
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "…"


def format_date(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return ""
    return f"{dt.day} {MONTHS[dt.month - 1]} {dt.year}"


def _impact_item(item) -> dict:
    chain = [step.strip(" →-—") for step in item.chain if step.strip(" →-—")]
    return {"title": clean_refs(item.title), "text": clean_refs(item.text), "level": item.level, "horizon": item.horizon,
            "chain": chain, "source_type": item.source_type, "source_label": SOURCE_LABELS[item.source_type]}


def _fact(f) -> str:
    if f.status in STATUS_PREFIX and f.attributed_to:
        return f"{STATUS_PREFIX[f.status]} ({f.attributed_to}): {f.text}"
    return f.text


def to_frontend(news: dict, na: NewsAnalysis, impact: ImpactAnalysis, bank_fragments: dict, article: str) -> dict:
    words = len((article or news.get("summary") or "").split())
    bank_view = None
    sources = [{"name": news.get("source") or "Источник", "title": news.get("title", ""), "url": news.get("link") or None}]
    if impact.bank_view:
        v = impact.bank_view
        bank_view = {"text": clean_refs(v.text), "author": v.author, "doc": v.document, "date": v.date or "",
                     "source_type": v.source_type}
        for ref in v.refs:
            fragment = bank_fragments.get(ref)
            if fragment:
                sources.append({"name": "Альфа-Банк", "title": f"{fragment['document']} ({fragment['citation']})",
                                "url": None})
    basis = [short_tag(link.tag) for link in impact.company_links]
    if bank_view:
        basis.append("Аналитика Альфа-Банка")
    return {
        "id": news["id"],
        "title": news.get("title") if na.language == "ru" else na.headline_ru,
        "summary": na.summary,
        "category": na.category,
        "category_label": CATEGORY_LABELS[na.category],
        "source": news.get("source", ""),
        "published": format_date(news.get("published", "")),
        "read_time": f"{max(1, round(words / 180))} мин",
        "relevance": impact.significance,
        "impact_level": impact.impact_level,
        "link_type": impact.link_type,
        "effect_size": impact.effect_size,
        "urgent": impact.urgent,
        "urgent_reason": impact.urgent_reason or "",
        "why": clean_refs(impact.why_it_matters),
        "facts": [_fact(f) for f in na.facts[:6]],
        "body": [p for p in (article or news.get("summary", "")).split("\n\n") if p.strip()],
        "bank_view": bank_view,
        "risks": [_impact_item(r) for r in impact.risks],
        "opportunities": [_impact_item(o) for o in impact.opportunities],
        "actions": [{"title": clean_refs(a.title), "text": clean_refs(a.text), "kind": a.kind, "horizon": a.horizon}
                    for a in impact.actions],
        "questions": [clean_refs(q) for q in impact.questions_to_check],
        "basis": basis,
        "sources": sources,
        "confidence": impact.confidence,
        "limitations": impact.limitations,
    }
