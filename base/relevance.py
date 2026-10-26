"""Вызов агента релевантности (relevance_agent) из продукта.

Агент оценивает одну новость для одной компании (балл 0–100 и причина). Здесь — настройки по
умолчанию, общий граф на процесс и ограничение числа контрагентов: у крупных компаний их сотни,
и полный список не помещается в лимит контекста агента.
"""
import os
import threading

from relevance_agent.contracts import NewsItem
from relevance_agent.counterparties import CounterpartyTools
from relevance_agent.single_contracts import SingleNewsInput

# Те же значения, что в relevance_agent/.env.example; ключ агент берёт из LLM_API_KEY корневого .env.
os.environ.setdefault("RELEVANCE_BASE_URL", "https://changellenge.up.railway.app/v1")
os.environ.setdefault("RELEVANCE_MODEL", "deepseek/deepseek-v4.1-flash")

TOP_COUNTERPARTIES = 10  # крупнейших контрагентов в каждом направлении

_graph = None
_graph_lock = threading.Lock()


class TopCounterparties:
    """Как CounterpartyTools.collect_context, но только крупнейшие контрагенты по доле платежей."""

    def __init__(self, tools: CounterpartyTools, top_n: int = TOP_COUNTERPARTIES):
        self.tools = tools
        self.top_n = top_n
        self.lock = threading.Lock()  # кэши CounterpartyTools заполняются лениво

    def collect_context(self, company_id: str):
        with self.lock:
            links = self.tools.get_counterparties(company_id)
            kept = []
            for direction in ("outgoing", "incoming"):
                side = [x for x in links.links if x.direction == direction]
                kept += sorted(side, key=lambda x: -(x.share or 0))[:self.top_n]
            if len(kept) < len(links.links):
                note = (f"Переданы {len(kept)} крупнейших связей из {len(links.links)} "
                        f"(до {self.top_n} в каждом направлении): полный список не помещается в контекст.")
                links = links.model_copy(update={"links": kept, "complete": False,
                                                 "limitations": links.limitations + [note]})
            profiles = [self.tools.get_company_profile(uid) for uid in sorted({x.company_id for x in links.links})]
            return links, profiles


def get_graph():
    global _graph
    with _graph_lock:
        if _graph is None:
            from relevance_agent.graph import build_graph
            from relevance_agent.model import load_model
            _graph = build_graph(load_model(), counterparties=TopCounterparties(CounterpartyTools()))
        return _graph


def score(company_id: str, news_item: dict):
    """SingleNewsOutput агента релевантности для новости в формате парсера."""
    fields = {k: news_item[k] for k in ("id", "title", "summary", "full_text", "link", "published",
                                        "source", "author", "categories", "language")}
    request = SingleNewsInput(company_id=company_id, news=NewsItem(**fields))
    return get_graph().invoke({"request": request}, {"recursion_limit": 10})["output"]
