"""Сборка ленты компании: новости → агент релевантности → полный анализ топ-новостей.

MVP без очередей (Celery, Redis): работа идёт в фоновых потоках процесса Django. Состояние хранится
в базе (FeedJob, CompanyNews), поэтому страница просто опрашивает статус. Если процесс перезапустили
посреди работы, «зависшая» задача распознаётся по давности обновления и запускается заново.
"""
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta

from django.db import close_old_connections, connection
from django.db.models import F, Max, Q
from django.utils import timezone

from companies.models import Company

from . import relevance
from .models import CompanyNews, FeedJob, News
from .news_ingest import fetch_and_store

log = logging.getLogger(__name__)

FEED_NEWS_LIMIT = 30          # сколько новых новостей оценивать за одну сборку (≈0,2 ₽ и 15 с на новость)
NEWS_MAX_AGE_HOURS = 72       # оцениваем только свежие новости
NEWS_REFRESH_MINUTES = 60     # если парсер не запускался дольше — сначала обновляем новости
FEED_REFRESH_MINUTES = 60     # готовую ленту пересобираем не чаще (оцениваются только новые новости)
TOP_ANALYSES = 3              # по скольким главным новостям сразу готовить полный анализ
MIN_SCORE_FOR_ANALYSIS = 20   # слабые новости не тратят бюджет на полный анализ
RELEVANCE_WORKERS = 5
ANALYSIS_WORKERS = 3
STALE_MINUTES = 15            # столько без обновлений — задача или анализ считаются прерванными

_start_lock = threading.Lock()


# --- Состояние ---------------------------------------------------------------------------

def latest_job(company: Company) -> FeedJob | None:
    return FeedJob.objects.filter(company=company).first()


def is_stale(moment) -> bool:
    return moment is None or moment < timezone.now() - timedelta(minutes=STALE_MINUTES)


def analysis_state(cn: CompanyNews) -> str:
    """none / loading / ready / error — с учётом анализов, прерванных перезапуском."""
    if cn.analysis_status == "ready" and cn.analysis_id:
        return "ready"
    if cn.analysis_status in ("queued", "running"):
        return "error" if is_stale(cn.analysis_started_at or cn.updated_at) else "loading"
    return cn.analysis_status if cn.analysis_status == "error" else "none"


def ensure_feed(company: Company) -> FeedJob:
    """Возвращает текущую сборку ленты; при необходимости запускает новую в фоне."""
    with _start_lock:
        job = latest_job(company)
        if job and job.status == "running":
            if not is_stale(job.updated_at):
                return job
            job.status, job.error, job.finished_at = "error", "Сборка прервана (перезапуск сервера)", timezone.now()
            job.save()
        elif job and job.status == "ready" and job.finished_at and \
                job.finished_at > timezone.now() - timedelta(minutes=FEED_REFRESH_MINUTES):
            return job
        job = FeedJob.objects.create(company=company, stage="Готовим ленту")
        threading.Thread(target=_run_job, args=(job.pk,), daemon=True, name=f"feed-{company.pk}").start()
        return job


def request_analysis(cn: CompanyNews) -> None:
    """Полный анализ новости по запросу пользователя (для новостей вне топ-3)."""
    if analysis_state(cn) in ("loading", "ready"):
        return
    CompanyNews.objects.filter(pk=cn.pk).update(analysis_status="queued", analysis_started_at=timezone.now())
    threading.Thread(target=_analyze_worker, args=(cn.pk,), daemon=True, name=f"analysis-{cn.pk}").start()


# --- Фоновая работа ----------------------------------------------------------------------

def _update_job(job_id: int, **fields) -> None:
    FeedJob.objects.filter(pk=job_id).update(**fields, updated_at=timezone.now())


def _run_job(job_id: int) -> None:
    close_old_connections()
    try:
        job = FeedJob.objects.select_related("company").get(pk=job_id)
        company = job.company

        last_fetch = News.objects.aggregate(last=Max("fetched_at"))["last"]
        if last_fetch is None or last_fetch < timezone.now() - timedelta(minutes=NEWS_REFRESH_MINUTES):
            _update_job(job_id, stage="Загружаем свежие новости")
            try:
                fetch_and_store()
            except Exception:  # без свежих новостей всё равно соберём ленту из имеющихся
                log.exception("Парсер новостей завершился с ошибкой")

        candidates = _select_candidates(company)
        _update_job(job_id, stage="Оцениваем важность новостей", news_total=len(candidates))
        with ThreadPoolExecutor(RELEVANCE_WORKERS) as pool:
            futures = [pool.submit(_score_worker, company.pk, news.pk) for news in candidates]
            for _ in as_completed(futures):
                _update_job(job_id, news_scored=F("news_scored") + 1)

        top = _pick_top(company)
        done = sum(1 for cn in top if analysis_state(cn) == "ready")
        todo = [cn for cn in top if analysis_state(cn) in ("none", "error")]
        CompanyNews.objects.filter(pk__in=[cn.pk for cn in todo]).update(
            analysis_status="queued", analysis_started_at=timezone.now())
        _update_job(job_id, stage="Готовим аналитику по главным новостям",
                    analyses_total=len(top), analyses_done=done)
        with ThreadPoolExecutor(ANALYSIS_WORKERS) as pool:
            futures = [pool.submit(_analyze_worker, cn.pk) for cn in todo]
            for _ in as_completed(futures):
                _update_job(job_id, analyses_done=F("analyses_done") + 1)

        _update_job(job_id, status="ready", stage="Лента готова", finished_at=timezone.now())
    except Exception as exc:
        log.exception("Сборка ленты завершилась с ошибкой")
        _update_job(job_id, status="error", error=f"{type(exc).__name__}: {exc}"[:2000], finished_at=timezone.now())
    finally:
        connection.close()


def _select_candidates(company: Company) -> list[News]:
    """Свежие новости, которых ещё нет в ленте компании; по очереди из разных источников."""
    cutoff = timezone.now() - timedelta(hours=NEWS_MAX_AGE_HOURS)
    seen = CompanyNews.objects.filter(company=company).values_list("news_id", flat=True)
    fresh = News.objects.exclude(pk__in=seen).filter(
        Q(published_at__gte=cutoff) | Q(published_at__isnull=True, fetched_at__gte=cutoff))
    pool = list(fresh.order_by("-published_at")[:300])
    if len(pool) < 10:  # база давно не обновлялась — берём последние имеющиеся
        pool = list(News.objects.exclude(pk__in=seen).order_by("-published_at")[:300])
    # Круговой обход источников, чтобы одна лента (например, ЦБ) не заняла всю выборку.
    by_source: dict[str, list[News]] = {}
    for news in pool:
        by_source.setdefault(news.source, []).append(news)
    picked: list[News] = []
    while len(picked) < FEED_NEWS_LIMIT and any(by_source.values()):
        for items in by_source.values():
            if items and len(picked) < FEED_NEWS_LIMIT:
                picked.append(items.pop(0))
    return picked


def _pick_top(company: Company) -> list[CompanyNews]:
    return list(CompanyNews.objects.filter(company=company, status="scored", score__gte=MIN_SCORE_FOR_ANALYSIS)
                .select_related("news").order_by("-score", "-news__published_at")[:TOP_ANALYSES])


def _score_worker(company_id: str, news_pk: int) -> None:
    close_old_connections()
    try:
        news = News.objects.get(pk=news_pk)
        started = time.monotonic()
        try:
            out = relevance.score(company_id, news.as_parser_item())
        except Exception as exc:
            log.exception("Агент релевантности упал на новости %s", news.uid)
            CompanyNews.objects.update_or_create(company_id=company_id, news=news, defaults=dict(
                status="error", score=None, warnings=[f"{type(exc).__name__}: {exc}"[:500]]))
            return
        a = out.assessment
        CompanyNews.objects.update_or_create(company_id=company_id, news=news, defaults=dict(
            status=out.status,
            score=a.relevance_score if a else None,
            reason=(a.reason or "") if a else "",
            channels=[c.model_dump() for c in a.channels] if a else [],
            evidence={"company": [e.model_dump() for e in a.company_evidence],
                      "counterparties": [e.model_dump() for e in a.counterparty_evidence]} if a else {},
            warnings=out.warnings[:15],
            relevance_seconds=round(time.monotonic() - started, 1),
        ))
    finally:
        connection.close()


def _analyze_worker(cn_pk: int) -> None:
    from news_analysis.graph import run_analysis

    close_old_connections()
    try:
        cn = CompanyNews.objects.select_related("news").get(pk=cn_pk)
        CompanyNews.objects.filter(pk=cn_pk).update(analysis_status="running", analysis_started_at=timezone.now())
        try:
            run = run_analysis(cn.news.as_parser_item(), cn.company_id)
            ok = run.status in ("ok", "low_confidence") and run.frontend
            CompanyNews.objects.filter(pk=cn_pk).update(analysis=run, analysis_status="ready" if ok else "error")
        except Exception:
            log.exception("Полный анализ упал: %s", cn)
            CompanyNews.objects.filter(pk=cn_pk).update(analysis_status="error")
    finally:
        connection.close()
