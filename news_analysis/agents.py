"""Три агента анализа «новость × компания»: аналитик новости, аналитик влияния, критик."""
import os
import threading
from collections import defaultdict
from dataclasses import dataclass, field

from django.db import IntegrityError

from companies.models import Company

from . import bank_search, prompts
from .llm import LLMClient, Trace, dumps
from .models import NewsAnalysisCache
from .schemas import AnalysisPlan, CriticReport, ImpactAnalysis, NewsAnalysis
from .tools import TOOL_SPECS, CompanyTools

NEWS_MODEL = os.environ.get("NEWS_ANALYST_MODEL", "google/gemma-4-26b-a4b-it")
IMPACT_MODEL = os.environ.get("IMPACT_ANALYST_MODEL", "deepseek/deepseek-v4.1-flash")
# Запасные модели на случай обрыва соединения у провайдера основной.
NEWS_FALLBACK = ("deepseek/deepseek-v4.1-flash", None)
IMPACT_FALLBACK = ("qwen/qwen3.6-35b-a3b", None)
CRITIC_MODEL = os.environ.get("CRITIC_MODEL", "z-ai/glm-5.3-flash")
# С полными рассуждениями GLM на длинном входе думает минутами и провайдер рвёт соединение;
# с reasoning_effort=low отвечает за ~10 с и находит те же проблемы.
CRITIC_EXTRA = {"reasoning_effort": "low"}
CRITIC_FALLBACK = ("qwen/qwen3.6-35b-a3b", {"reasoning_effort": "low"})

MAX_RESEARCH_CALLS = 6
MAX_REVISION_CALLS = 3
BANK_CONTEXT_SIZE = 5
MAX_ARTICLE_CHARS = 15000
# Аналитику влияния и критику текст нужен для проверки деталей; длинные статьи обрезаем ради бюджета.
MAX_ARTICLE_FOR_REVIEW = 8000
MAX_TOOL_RESULT_FOR_CRITIC = 2500


# --- Профиль компании в компактном виде ------------------------------------------------

# Поля, которые аналитик видит сразу. Структура и имена совпадают с полным профилем,
# поэтому JSON-пути из промпта проверяются по полному профилю.
PROFILE_VIEW = {
    "scale": ["revenue_label", "revenue_band", "employees_label", "is_target_audience", "staff_anomaly"],
    "industry": ["okved_code", "okved_name", "sector", "business_model", "is_capital_intensive",
                 "commodity_exposure", "demand_drivers", "regulation"],
    "legal": ["ownership", "has_foreign_capital", "has_state_capital", "age_label", "development_stage"],
    "geography": ["region", "federal_district"],
    "foreign_trade": ["is_international", "countries", "likely_currencies", "unfriendly_countries",
                      "transit_hub_countries", "note"],
    "bank_relationship": ["client_status", "active_products", "missing_products", "uses_debt_financing",
                          "debt_instruments", "has_free_cash"],
    "group": ["group_size_label", "group_top_sectors"],
    "counterparties": ["has_payment_data", "suppliers_count", "customers_count", "suppliers_volume_label",
                       "customers_volume_label", "top5_suppliers_share", "top5_customers_share",
                       "suppliers_concentration", "customers_concentration", "suppliers_sectors",
                       "customers_sectors"],
}


def compact_profile(profile: dict) -> dict:
    view = {"summary": profile.get("summary")}
    for section, keys in PROFILE_VIEW.items():
        data = profile.get(section) or {}
        view[section] = {k: data.get(k) for k in keys if k in data}
    return view


def review_text(news: dict, article: str) -> str:
    text = article or news.get("summary") or ""
    if len(text) > MAX_ARTICLE_FOR_REVIEW:
        text = text[:MAX_ARTICLE_FOR_REVIEW].rsplit(" ", 1)[0] + "…(обрезано)"
    return text or "нет текста"


# --- Контекст прогона ------------------------------------------------------------------

@dataclass
class RunContext:
    company: Company
    news: dict
    trace: Trace = field(default_factory=Trace)
    use_cache: bool = True

    def __post_init__(self):
        self.llm = LLMClient(self.trace)
        self.news_date = (self.news.get("published") or "")[:10] or None
        self.tools = CompanyTools(self.company, self.news_date, self.trace)
        self.profile_view = compact_profile(self.company.profile or {})


# --- Агент 1 -----------------------------------------------------------------------------

# Параллельные прогоны одной новости ждут друг друга: разбор делается один раз и берётся из кэша.
_news_locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)


def analyze_news(ctx: RunContext, article: str, article_source: str) -> NewsAnalysis:
    with _news_locks[ctx.news["id"]]:
        return _analyze_news(ctx, article, article_source)


def _analyze_news(ctx: RunContext, article: str, article_source: str) -> NewsAnalysis:
    news = ctx.news
    if ctx.use_cache:
        cached = NewsAnalysisCache.objects.filter(news_id=news["id"], prompt_version=prompts.NEWS_PROMPT_VERSION,
                                                  model=NEWS_MODEL).first()
        if cached:
            ctx.trace.add(kind="cache", step="news_analyst", note="разбор новости взят из кэша")
            return NewsAnalysis.model_validate(cached.result)

    messages = [
        {"role": "system", "content": prompts.NEWS_ANALYST_SYSTEM},
        {"role": "user", "content": prompts.NEWS_ANALYST_USER.format(
            source=news.get("source") or "не указан", published=news.get("published") or "не указана",
            title=news.get("title", ""), summary=news.get("summary") or "нет",
            article_source=article_source, article=article[:MAX_ARTICLE_CHARS] or "нет текста, только анонс")},
    ]
    result = ctx.llm.structured(step="news_analyst", model=NEWS_MODEL, messages=messages,
                                schema=NewsAnalysis, max_tokens=12000, fallback=NEWS_FALLBACK)
    try:
        NewsAnalysisCache.objects.update_or_create(
            news_id=news["id"], prompt_version=prompts.NEWS_PROMPT_VERSION, model=NEWS_MODEL,
            defaults={"result": result.model_dump()})
    except IntegrityError:
        pass  # параллельный прогон той же новости уже записал разбор
    return result


# --- Обязательный поиск по аналитике банка (код, не агент) -----------------------------

def retrieve_bank_context(ctx: RunContext, news_analysis: NewsAnalysis) -> list[dict]:
    found: dict[str, dict] = {}
    for query in news_analysis.bank_search_queries:
        for hit in bank_search.search(query, top_k=3, as_of=ctx.news_date):
            if hit["id"] not in found or hit["score"] > found[hit["id"]]["score"]:
                found[hit["id"]] = {**hit, "query": query}
    context = sorted(found.values(), key=lambda h: -h["score"])[:BANK_CONTEXT_SIZE]
    ctx.trace.add(kind="code", step="bank_retrieval", queries=news_analysis.bank_search_queries,
                  as_of=ctx.news_date, fragments=[{"id": h["id"], "document": h["document"], "score": h["score"]}
                                                  for h in context])
    for h in context:
        ctx.tools.seen_bank_fragments[h["id"]] = h
    return context


# --- Агент 2 -----------------------------------------------------------------------------

class ImpactAnalyst:
    """Агент с инструментами: план → сбор данных → итог; при замечаниях критика — доработка."""

    def __init__(self, ctx: RunContext, news_analysis: NewsAnalysis, bank_context: list[dict],
                 article: str, article_source: str):
        self.ctx = ctx
        self.messages = [
            {"role": "system", "content": prompts.IMPACT_SYSTEM},
            {"role": "user", "content": prompts.IMPACT_CONTEXT.format(
                news_analysis=dumps(news_analysis.model_dump()), article_source=article_source,
                article=review_text(ctx.news, article), company_id=ctx.company.company_id,
                profile=dumps(ctx.profile_view),
                bank_context=dumps(bank_context) if bank_context else "ничего не найдено")},
        ]

    def plan(self) -> AnalysisPlan:
        self.messages.append({"role": "user", "content": prompts.PLAN_INSTRUCTION})
        plan = self.ctx.llm.structured(step="impact_plan", model=IMPACT_MODEL, messages=self.messages,
                                       schema=AnalysisPlan, max_tokens=6000, fallback=IMPACT_FALLBACK)
        self.messages.append({"role": "assistant", "content": dumps(plan.model_dump())})
        return plan

    def research(self, instruction: str, max_calls: int, step: str) -> None:
        """Цикл вызова инструментов: модель сама решает, что и когда запросить."""
        self.messages.append({"role": "user", "content": instruction})
        calls_made = 0
        for _ in range(max_calls + 2):
            response = self.ctx.llm.chat(step=step, model=IMPACT_MODEL, messages=self.messages,
                                         tools=TOOL_SPECS, max_tokens=6000, fallback=IMPACT_FALLBACK)
            assistant = {"role": "assistant", "content": response.content or ""}
            if response.tool_calls:
                assistant["tool_calls"] = response.tool_calls
            if response.message.get("reasoning_content"):
                assistant["reasoning_content"] = response.message["reasoning_content"]
            self.messages.append(assistant)
            if not response.tool_calls:
                return
            for call in response.tool_calls:
                if calls_made >= max_calls:
                    text = "Лимит вызовов инструментов исчерпан. Завершай сбор данных."
                else:
                    _, text = self.ctx.tools.execute(call["function"]["name"], call["function"]["arguments"])
                    calls_made += 1
                self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": text})
        self.messages.append({"role": "user", "content": "Сбор данных завершён."})

    def write(self, instruction: str, step: str) -> ImpactAnalysis:
        self.messages.append({"role": "user", "content": instruction})
        analysis = self.ctx.llm.structured(step=step, model=IMPACT_MODEL, messages=self.messages,
                                           schema=ImpactAnalysis, max_tokens=10000, fallback=IMPACT_FALLBACK)
        self.messages.append({"role": "assistant", "content": dumps(analysis.model_dump())})
        return analysis

    def revise(self, issues: list[dict], round_no: int) -> ImpactAnalysis:
        listing = "\n".join(f"- [{i['severity']}] {i['location']}: {i['problem']} → {i['fix']}" for i in issues)
        self.research(prompts.REVISE_INSTRUCTION.format(issues=listing, max_calls=MAX_REVISION_CALLS),
                      MAX_REVISION_CALLS, step=f"impact_revise_research{round_no}")
        return self.write(prompts.REVISE_FINAL_INSTRUCTION, step=f"impact_revise{round_no}")


# --- Агент 3 -----------------------------------------------------------------------------

def criticize(ctx: RunContext, news_analysis: NewsAnalysis, analysis: ImpactAnalysis, code_issues: list[dict],
              article: str, article_source: str) -> tuple[CriticReport, list[dict]]:
    tool_results = {
        call_id: {"name": c["name"], "arguments": c["arguments"],
                  "result": (dumps(c["result"])[:MAX_TOOL_RESULT_FOR_CRITIC] if c["result"] is not None else c["error"])}
        for call_id, c in ctx.tools.calls.items()
    }
    messages = [
        {"role": "system", "content": prompts.CRITIC_SYSTEM},
        {"role": "user", "content": prompts.CRITIC_USER.format(
            news_analysis=dumps(news_analysis.model_dump()), article_source=article_source,
            article=review_text(ctx.news, article), company_id=ctx.company.company_id,
            profile=dumps(ctx.profile_view), tool_results=dumps(tool_results) if tool_results else "вызовов не было",
            bank_fragments=dumps(list(ctx.tools.seen_bank_fragments.values())) or "нет",
            analysis=dumps(analysis.model_dump()), code_issues=dumps(code_issues) if code_issues else "нет")},
    ]
    report = ctx.llm.structured(step="critic", model=CRITIC_MODEL, messages=messages, schema=CriticReport,
                                max_tokens=8000, extra=CRITIC_EXTRA, fallback=CRITIC_FALLBACK)
    # Замечания кода не зависят от мнения критика: добавляем их, если критик их не повторил.
    issues = [i.model_dump() | {"source": "critic"} for i in report.issues]
    seen = {(i["location"], i["problem"]) for i in issues}
    issues += [i for i in code_issues if (i["location"], i["problem"]) not in seen]
    return report, issues


def blocking(issues: list[dict]) -> list[dict]:
    return [i for i in issues if i["severity"] in ("critical", "major")]
