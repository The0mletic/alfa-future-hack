"""Граф анализа «новость × компания» на LangGraph.

prepare → news_analyst → bank_retrieval → impact_plan → impact_research → impact_write
        → validate → critic ─┬─ pass ───────────────────────────────→ finalize
                             └─ revise (не более MAX_REVISIONS раз) → validate → critic …
Если после MAX_REVISIONS критик всё ещё недоволен, анализ отдаётся с низкой уверенностью.
"""
import time
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from companies.models import Company

from . import agents, prompts, validation
from .article import fetch_article
from .frontend import to_frontend
from .models import AnalysisRun
from .schemas import AnalysisPlan, ImpactAnalysis, NewsAnalysis

MAX_REVISIONS = 2


class State(TypedDict, total=False):
    article: str
    article_source: str
    news_analysis: NewsAnalysis
    bank_context: list[dict]
    plan: AnalysisPlan
    impact: ImpactAnalysis
    code_issues: list[dict]
    critic_reports: list[dict]
    open_issues: list[dict]
    revisions: int
    status: str


def build_graph(ctx: agents.RunContext):
    analyst: dict[str, agents.ImpactAnalyst] = {}

    def prepare(state: State) -> State:
        news = ctx.news
        if news.get("content"):
            return {"article": news["content"], "article_source": "текст из входных данных", "revisions": 0}
        text, error = fetch_article(news.get("link", ""))
        ctx.trace.add(kind="code", step="fetch_article", url=news.get("link"), ok=bool(text), error=error)
        if text:
            return {"article": text, "article_source": "загружен по ссылке", "revisions": 0}
        return {"article": "", "article_source": f"только анонс ({error})", "revisions": 0}

    def news_analyst(state: State) -> State:
        return {"news_analysis": agents.analyze_news(ctx, state["article"], state["article_source"])}

    def bank_retrieval(state: State) -> State:
        return {"bank_context": agents.retrieve_bank_context(ctx, state["news_analysis"])}

    def impact_plan(state: State) -> State:
        analyst["a"] = agents.ImpactAnalyst(ctx, state["news_analysis"], state["bank_context"],
                                            state["article"], state["article_source"])
        return {"plan": analyst["a"].plan()}

    def impact_research(state: State) -> State:
        analyst["a"].research(prompts.RESEARCH_INSTRUCTION.format(max_calls=agents.MAX_RESEARCH_CALLS),
                              agents.MAX_RESEARCH_CALLS, step="impact_research")
        return {}

    def impact_write(state: State) -> State:
        return {"impact": analyst["a"].write(prompts.FINAL_INSTRUCTION, step="impact_write")}

    def validate(state: State) -> State:
        issues = validation.check(state["impact"], state["news_analysis"], ctx.company.profile or {},
                                  ctx.tools.calls, set(ctx.tools.seen_bank_fragments))
        ctx.trace.add(kind="code", step="validate", issues=issues)
        return {"code_issues": issues}

    def critic(state: State) -> State:
        report, issues = agents.criticize(ctx, state["news_analysis"], state["impact"], state["code_issues"],
                                          state["article"], state["article_source"])
        reports = state.get("critic_reports", []) + [
            {"round": state["revisions"], "report": report.model_dump(), "issues": issues}]
        return {"critic_reports": reports, "open_issues": agents.blocking(issues)}

    def route(state: State) -> str:
        if not state["open_issues"]:
            return "finalize"
        return "revise" if state["revisions"] < MAX_REVISIONS else "finalize"

    def revise(state: State) -> State:
        round_no = state["revisions"] + 1
        impact = analyst["a"].revise(state["open_issues"], round_no)
        return {"impact": impact, "revisions": round_no}

    def finalize(state: State) -> State:
        impact = state["impact"]
        if state["open_issues"]:
            unresolved = [f"{i['location']}: {i['problem']}" for i in state["open_issues"]]
            impact = impact.model_copy(update={
                "confidence": "low",
                "limitations": impact.limitations + [f"Не устранено после проверки: {u}" for u in unresolved],
            })
            return {"impact": impact, "status": "low_confidence"}
        return {"status": "ok"}

    graph = StateGraph(State)
    for name, fn in [("prepare", prepare), ("news_analyst", news_analyst), ("bank_retrieval", bank_retrieval),
                     ("impact_plan", impact_plan), ("impact_research", impact_research),
                     ("impact_write", impact_write), ("validate", validate), ("critic", critic),
                     ("revise", revise), ("finalize", finalize)]:
        graph.add_node(name, fn)
    graph.set_entry_point("prepare")
    for a, b in [("prepare", "news_analyst"), ("news_analyst", "bank_retrieval"), ("bank_retrieval", "impact_plan"),
                 ("impact_plan", "impact_research"), ("impact_research", "impact_write"),
                 ("impact_write", "validate"), ("validate", "critic"), ("revise", "validate"),
                 ("finalize", END)]:
        graph.add_edge(a, b)
    graph.add_conditional_edges("critic", route, {"revise": "revise", "finalize": "finalize"})
    return graph.compile()


def run_analysis(news: dict, company_id: str, use_cache: bool = True) -> AnalysisRun:
    """Полный прогон для пары «новость × компания». Результат и трассировка сохраняются в базу."""
    started = time.monotonic()
    company = Company.objects.get(pk=company_id)
    if not company.profile:
        raise ValueError(f"У компании {company_id} нет профиля: анализ возможен только для рабочего набора")
    ctx = agents.RunContext(company=company, news=news, use_cache=use_cache)
    run = AnalysisRun(news_id=news["id"], company_id=company_id, news=news)
    state: dict[str, Any] = {}
    try:
        # stream вместо invoke: при ошибке сохраняется состояние последнего успешного шага.
        for state in build_graph(ctx).stream({}, {"recursion_limit": 50}, stream_mode="values"):
            pass
        run.status = state["status"]
    except Exception as exc:
        run.status, run.error = "error", f"{type(exc).__name__}: {exc}"
    run.news_analysis = state["news_analysis"].model_dump() if state.get("news_analysis") else None
    run.plan = state["plan"].model_dump() if state.get("plan") else None
    run.impact = state["impact"].model_dump() if state.get("impact") else None
    run.critic_reports = state.get("critic_reports", [])
    run.revisions = state.get("revisions", 0)
    if state.get("impact") and state.get("news_analysis"):
        run.frontend = to_frontend(news, state["news_analysis"], state["impact"], ctx.tools.seen_bank_fragments,
                                   state.get("article", ""))
    run.trace = ctx.trace.steps
    run.cost = round(ctx.trace.cost, 4)
    run.duration_s = round(time.monotonic() - started, 1)
    run.save()
    return run
