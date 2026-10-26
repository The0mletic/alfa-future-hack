"""Чат «AI-аналитик» на странице новости: агент applications/chat_agent с данными из базы.

ID новости для агента — «компания:новость»: у каждой компании свой разбор и своя история чата.
"""
import os
import re
import threading

from django.db.models import Q

from companies.models import Company
from news_analysis import bank_search

from . import presenters
from .models import CompanyNews, News

# Модель и адрес по умолчанию; ключ агент читает из LLM_API_KEY корневого .env.
os.environ.setdefault("LLM_MODEL", "deepseek/deepseek-v4.1-flash")
os.environ.setdefault("LLM_BASE_URL", "https://changellenge.up.railway.app/v1")

_agent = None
_lock = threading.Lock()


def _stems(text: str) -> set[str]:
    return {w[:5] for w in re.findall(r"[а-яёa-z0-9]+", (text or "").lower()) if len(w) >= 4}


def _split_key(news_id: str) -> tuple[str, int]:
    company_id, news_pk = str(news_id).split(":", 1)
    return company_id, int(news_pk)


class DbDataSource:
    def find_news(self, news_id) -> dict:
        company_id, news_pk = _split_key(news_id)
        cn = CompanyNews.objects.select_related("news", "analysis").get(company_id=company_id, news_id=news_pk)
        item = presenters.detail_item(cn)
        return {
            "id": news_id, "title": item["title"], "summary": item.get("summary", ""),
            "source": item["source"], "published": item["published"], "body": item.get("body") or [],
            "sources": item.get("sources") or [{"url": cn.news.url}],
            "why": item.get("why") or "", "facts": item.get("facts") or [],
            "risks": item.get("risks") or [], "opportunities": item.get("opportunities") or [],
            "bank_view": item.get("bank_view"),
        }

    def company_profile(self, news_id) -> str:
        company_id, _ = _split_key(news_id)
        company = Company.objects.get(pk=company_id)
        brief = company.brief or {}
        facts = brief.get("facts") or {}
        lines = [company.summary or f"Компания {company_id}"]
        for key in ("продукты_банка", "страны_вэд", "поставщики", "покупатели", "отрасли_поставщиков"):
            if facts.get(key):
                value = facts[key]
                lines.append(f"{key.replace('_', ' ')}: {', '.join(value) if isinstance(value, list) else value}")
        return "\n".join(lines)

    def search_analytics(self, query: str) -> str:
        hits = bank_search.search(query, top_k=3)
        return "\n\n".join(f"Обзор «{h['document']}» ({h['published_at']}, {h['citation']}): {h['text'][:700]}"
                           for h in hits) or "По этому запросу аналитики не найдено."

    def search_news(self, query: str) -> str:
        words = _stems(query)
        if not words:
            return "Пустой запрос."
        candidates = News.objects.filter(
            Q(*[Q(title__icontains=w) | Q(summary__icontains=w) for w in list(words)[:4]], _connector=Q.OR)
        ).order_by("-published_at")[:200]
        scored = []
        for n in candidates:
            score = 2 * len(words & _stems(n.title)) + len(words & _stems(n.summary or n.full_text[:500]))
            if score:
                scored.append((score, n))
        scored.sort(key=lambda x: (x[0], x[1].published_at or x[1].fetched_at), reverse=True)
        return "\n\n".join(
            f"{n.title} ({n.source}, {presenters.fmt_published(n.published_at)}). "
            f"{(n.summary or n.full_text)[:400]}\nСсылка: {n.url}" for _, n in scored[:3]
        ) or "По этому запросу новостей не найдено. Попробуй другой запрос."


def _get_agent():
    global _agent
    with _lock:
        if _agent is None:
            from applications.chat_agent import agent_chat_graph
            agent_chat_graph.set_data_source(DbDataSource())
            _agent = agent_chat_graph
        return _agent


def ask(cn: CompanyNews, question: str, session_key: str) -> str:
    return _get_agent().ask(news_id=f"{cn.company_id}:{cn.news_id}", question=question, user_id=session_key)
