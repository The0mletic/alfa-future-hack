"""Данные для шаблонов base/: карточки ленты, страница новости, профиль компании.

Форматы совпадают с контрактом шаблонов (base/mock_data.py) плюс поля состояния анализа.
"""
import re

from django.utils import timezone

from companies.models import Company
from news_analysis.frontend import CATEGORY_LABELS, MONTHS, clean_refs, short_tag

from .feed import analysis_state
from .models import CompanyNews

LABELS = {**CATEGORY_LABELS, "counterparties": "Контрагенты"}
# Пока полного анализа нет, тему новости определяем по главному каналу влияния от агента релевантности.
CHANNEL_CATEGORY = {
    "cost_of_money": "macro", "fx_trade": "markets", "input_costs": "commodities", "regulation": "regulation",
    "customer_demand": "industry", "market_structure": "industry", "logistics": "industry",
    "counterparties": "counterparties",
}
PARAGRAPH_CHARS = 700


def fmt_published(moment) -> str:
    if not moment:
        return ""
    local = timezone.localtime(moment)
    return f"{local.day} {MONTHS[local.month - 1]}, {local:%H:%M}"


def paragraphs(text: str) -> list[str]:
    """Парсер склеивает статью в одну строку — режем на абзацы по границам предложений."""
    parts = [p.strip() for p in re.split(r"\n\s*\n|\n", text or "") if p.strip()]
    if len(parts) > 1:
        return parts
    sentences, out, current = re.split(r"(?<=[.!?…»])\s+", text or ""), [], ""
    for sentence in sentences:
        if current and len(current) + len(sentence) > PARAGRAPH_CHARS:
            out.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    return out + ([current] if current else [])


def company_context(company: Company) -> dict:
    p = company.profile or {}
    scale, industry, trade, bank = (p.get("scale") or {}), (p.get("industry") or {}), \
        (p.get("foreign_trade") or {}), (p.get("bank_relationship") or {})
    return {
        "id": company.company_id, "name": f"Компания {company.company_id}", "initials": "U",
        "meta": f"{bank.get('client_status', 'клиент банка')} · данные обезличены",
        "industry": industry.get("sector") or company.sector, "okved": company.okved_code,
        "region": company.region, "revenue": scale.get("revenue_label", ""),
        "employees": scale.get("employees_label", ""),
        "foreign_trade": ", ".join(trade.get("countries") or []) or "нет",
        "products": bank.get("active_products") or [],
    }


def user_context(company: Company) -> dict:
    return {"name": "Руководитель", "initials": "Р", "role": f"Вход по ID {company.company_id}"}


def _category(cn: CompanyNews, frontend: dict | None) -> str:
    if frontend:
        return frontend["category"]
    if cn.channels:
        top = max(cn.channels, key=lambda c: c.get("impact") or 0)
        return CHANNEL_CATEGORY.get(top.get("channel"), "other")
    return "other"


def card(cn: CompanyNews) -> dict:
    news, state = cn.news, analysis_state(cn)
    frontend = cn.analysis.frontend if state == "ready" and cn.analysis else None
    category = _category(cn, frontend)
    summary = (frontend or {}).get("summary") or news.summary or (news.full_text[:300] + "…" if news.full_text else "")
    return {
        "id": news.pk, "title": news.title, "summary": summary, "category": category,
        "category_label": LABELS.get(category, "Другое"), "source": news.source,
        "published": fmt_published(news.published_at), "relevance": cn.score or 0,
        "why": clean_refs((frontend or {}).get("why") or cn.reason),
        "urgent": bool(frontend and frontend.get("urgent")), "urgent_reason": (frontend or {}).get("urgent_reason", ""),
        "risks": (frontend or {}).get("risks", []), "opportunities": (frontend or {}).get("opportunities", []),
        "bank_view": (frontend or {}).get("bank_view"),
        "analysis_state": state, "loading": state == "loading", "analysis_ready": state == "ready",
    }


def detail_item(cn: CompanyNews) -> dict:
    news, state = cn.news, analysis_state(cn)
    base_card = card(cn)
    if state == "ready":
        item = dict(cn.analysis.frontend)
        item.update(id=news.pk, title=news.title, analysis_state=state, relevance=cn.score or 0,
                    basis=[short_tag(b) for b in item.get("basis", [])])
        item["why"] = clean_refs(item.get("why"))
        if item.get("bank_view"):
            item["bank_view"] = {**item["bank_view"], "text": clean_refs(item["bank_view"]["text"])}
        if len(item.get("body") or []) <= 1:
            item["body"] = paragraphs(news.full_text) or item.get("body") or []
        return item
    basis = []
    for e in (cn.evidence or {}).get("company", []):
        label = (e.get("explanation") or "").split(":")[0]
        if label and label not in basis:
            basis.append(short_tag(label))
    return {
        **base_card, "read_time": f"{max(1, round(len((news.full_text or news.summary).split()) / 180))} мин",
        "facts": [], "body": paragraphs(news.full_text) or [news.summary],
        "bank_view": None, "risks": [], "opportunities": [], "basis": basis,
        "sources": [{"name": news.source, "title": news.title, "url": news.url}],
    }
