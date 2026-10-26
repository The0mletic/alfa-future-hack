import json
import re
from functools import wraps

from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_POST

from companies.models import Company
from news_analysis.render import EXTRA_CSS, trace_section

from . import feed, presenters
from .mock_data import MARKETS
from .models import CompanyNews

COMPANY_ID_RE = re.compile(r"^U_\d+$")
FEED_SIZE = 40


def company_required(view):
    """Вход только по ID компании (U_…), без пароля: ID хранится в сессии."""
    @wraps(view)
    def wrapper(request, *args, **kwargs):
        company_id = request.session.get("company_id")
        company = Company.objects.filter(pk=company_id, in_working_set=True).first() if company_id else None
        if company is None:
            if request.path.startswith("/api/"):
                return JsonResponse({"error": "Нужен вход"}, status=401)
            return redirect("login")
        request.company = company
        return view(request, *args, **kwargs)
    return wrapper


def _common_context(company: Company) -> dict:
    return {"user": presenters.user_context(company), "company": presenters.company_context(company)}


def login_view(request):
    error, value = "", ""
    if request.method == "POST":
        value = request.POST.get("company_id", "").strip().upper()
        if not COMPANY_ID_RE.match(value):
            error = "ID компании выглядит как U_ и цифры, например U_44564"
        elif not Company.objects.filter(pk=value, in_working_set=True).exists():
            error = "Такой компании нет среди клиентов сервиса"
        else:
            request.session["company_id"] = value
            return redirect("index")
    return render(request, "base/login.html", {"error": error, "value": value})


def logout_view(request):
    request.session.flush()
    return redirect("login")


def _job_status(job) -> dict:
    if job is None:
        return {"status": "none"}
    return {
        "status": job.status, "stage": job.stage, "news_total": job.news_total, "news_scored": job.news_scored,
        "analyses_total": job.analyses_total, "analyses_done": job.analyses_done, "error": job.error[:300],
    }


def _feed_items(company: Company):
    return (CompanyNews.objects.filter(company=company, status="scored", score__gt=0)
            .select_related("news", "analysis").order_by("-score", "-news__published_at"))


@company_required
def index(request):
    company = request.company
    job = feed.ensure_feed(company)
    cards = [presenters.card(cn) for cn in _feed_items(company)[:FEED_SIZE]]
    counts: dict[str, int] = {}
    for c in cards:
        counts[c["category"]] = counts.get(c["category"], 0) + 1
    categories = [{"slug": slug, "label": presenters.LABELS.get(slug, slug), "count": n}
                  for slug, n in sorted(counts.items(), key=lambda x: -x[1])]
    finished = job.finished_at if job.status != "running" else None
    context = {
        **_common_context(company),
        "news": cards,
        "urgent_news": [c for c in cards if c["urgent"]],
        "categories": categories,
        "markets": MARKETS,
        "feed_updated": timezone.localtime(finished).strftime("%H:%M") if finished else "…",
        "job": _job_status(job),
        "loading_count": sum(1 for c in cards if c["loading"]),
    }
    return render(request, "base/index.html", context)


@company_required
def feed_status(request):
    """Опрос со страницы ленты: пока собирается лента или готовится анализ — страница ждёт."""
    company = request.company
    job = feed.latest_job(company)
    loading = [c["id"] for c in map(presenters.card, _feed_items(company)[:FEED_SIZE]) if c["loading"]]
    return JsonResponse({**_job_status(job), "loading_ids": loading})


@company_required
@ensure_csrf_cookie  # токен для POST-запросов чата из app.js
def news_detail(request, news_id):
    company = request.company
    cn = get_object_or_404(CompanyNews.objects.select_related("news", "analysis"), company=company, news_id=news_id)
    item = presenters.detail_item(cn)
    related = [presenters.card(x) for x in _feed_items(company).exclude(pk=cn.pk)[:3]]
    details = ""
    if item["analysis_state"] == "ready":
        run = cn.analysis
        details = f"<style>{EXTRA_CSS}</style>" + trace_section({
            "impact": run.impact, "news_analysis": run.news_analysis, "plan": run.plan, "status": run.status,
            "revisions": run.revisions, "cost": run.cost, "duration_s": run.duration_s, "trace": run.trace,
            "critic_reports": run.critic_reports})
    return render(request, "base/news_detail.html", {
        **_common_context(company), "item": item, "related": related, "analysis_details": details})


@company_required
@require_POST
def news_analyze(request, news_id):
    cn = get_object_or_404(CompanyNews, company=request.company, news_id=news_id)
    feed.request_analysis(cn)
    return redirect("news_detail", news_id=news_id)


@company_required
def news_status(request, news_id):
    cn = get_object_or_404(CompanyNews, company=request.company, news_id=news_id)
    return JsonResponse({"analysis_state": feed.analysis_state(cn)})


@company_required
@require_POST
def chat_api(request, news_id):
    """Мини-чат по новости. Ожидает {"message": "..."}, возвращает {"answer": "..."}."""
    try:
        message = (json.loads(request.body).get("message") or "").strip()
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    if not message:
        return JsonResponse({"error": "Пустой вопрос"}, status=400)
    cn = get_object_or_404(CompanyNews, company=request.company, news_id=news_id)
    from .chat import ask
    return JsonResponse({"answer": ask(cn, message, request.session.session_key or "web")})
