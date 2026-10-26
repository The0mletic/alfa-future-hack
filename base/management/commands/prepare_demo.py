"""Заранее собирает ленты демо-компаний, чтобы на показе ничего не грузилось.

    python manage.py prepare_demo                          # компании по умолчанию
    python manage.py prepare_demo --company U_19717 --no-fetch

Для каждой компании: свежие новости → оценка агентом релевантности → полный анализ топ-3
(неудачный анализ повторяется). В конце — список новостей, которые стоит открыть на демо.
Запускайте сервер с FEED_AUTO_REFRESH=0, чтобы готовые ленты не пересобирались при входе.
"""
import json
from datetime import datetime
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from base import feed
from base.models import CompanyNews, FeedJob, News
from base.news_ingest import fetch_and_store
from companies.models import Company

DEMO_COMPANIES = ["U_19717", "U_44564", "U_9250"]
SAMPLES_DIR = Path(settings.BASE_DIR) / "applications" / "news_samples"


def import_samples() -> list[News]:
    """Проверенные новости из news_samples (формат парсера): гарантируют содержательное демо."""
    news = []
    for path in sorted(SAMPLES_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in (data.get("items") if isinstance(data, dict) else None) or []:
            published = datetime.fromisoformat(item["published"]) if item.get("published") else None
            obj, _ = News.objects.update_or_create(uid=item["id"], defaults=dict(
                title=item["title"], summary=item.get("summary", ""),
                full_text=item.get("content") or item.get("summary", ""), url=item.get("link") or "https://example.org",
                source=item["source"], author=item.get("author", ""), categories=item.get("categories", []),
                language=item.get("language", ""), published_at=published))
            news.append(obj)
    return news


class Command(BaseCommand):
    help = "Готовит ленты и аналитику для демо-компаний"

    def add_arguments(self, parser):
        parser.add_argument("--company", action="append", default=[], help="ID компании, можно несколько")
        parser.add_argument("--no-fetch", action="store_true", help="не запускать парсер новостей")
        parser.add_argument("--no-samples", action="store_true", help="не добавлять проверенные новости из news_samples")

    def handle(self, *args, **opts):
        if not opts["no_fetch"]:
            self.stdout.write("Загружаем свежие новости…")
            self.stdout.write(str(fetch_and_store()))
        samples = [] if opts["no_samples"] else import_samples()
        self.stdout.write(f"Проверенных новостей из news_samples: {len(samples)}")
        for company_id in opts["company"] or DEMO_COMPANIES:
            company = Company.objects.get(pk=company_id, in_working_set=True)
            self.stdout.write(f"\n=== {company_id}: собираем ленту", ending="\n")
            seen = set(CompanyNews.objects.filter(company=company).values_list("news_id", flat=True))
            for news in samples:
                if news.pk not in seen:
                    feed._score_worker(company_id, news.pk)
            job = FeedJob.objects.create(company=company, stage="Подготовка демо")
            feed._run_job(job.pk)
            job.refresh_from_db()
            # Повторяем анализ, если он упал (обрывы соединения у провайдера моделей).
            for cn in feed._pick_top(company):
                if feed.analysis_state(cn) != "ready":
                    self.stdout.write(f"  повторный анализ: {cn.news.title[:60]}")
                    feed._analyze_worker(cn.pk)
            self.report(company, job)

    def report(self, company, job):
        self.stdout.write(f"  сборка: {job.status} {job.error[:200]}")
        items = (CompanyNews.objects.filter(company=company, status="scored", score__gt=0)
                 .select_related("news").order_by("-score", "-news__published_at")[:8])
        for cn in items:
            state = feed.analysis_state(cn)
            self.stdout.write(f"  /news/{cn.news_id}/  балл {cn.score:3}  анализ: {state:7}  {cn.news.title[:70]}")
