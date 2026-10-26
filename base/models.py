"""Данные продукта: новости из парсера, их оценка для каждой компании и задачи сборки ленты.

Профили компаний — в приложении companies, полный анализ новости для компании — в news_analysis.
"""
from django.db import models

from companies.models import Company


class News(models.Model):
    """Новость из RSS-парсера. Общая для всех компаний."""

    uid = models.CharField("ID новости (sha1 ссылки, как у парсера и агентов)", max_length=32, unique=True)
    title = models.CharField(max_length=500)
    summary = models.TextField("анонс из RSS", blank=True)
    full_text = models.TextField("полный текст статьи", blank=True)
    url = models.URLField(max_length=1000)
    source = models.CharField(max_length=255)
    author = models.CharField(max_length=255, blank=True)
    categories = models.JSONField("рубрики источника", default=list)
    language = models.CharField(max_length=8, blank=True)
    published_at = models.DateTimeField(null=True, blank=True, db_index=True)
    fetched_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        verbose_name = "новость"
        verbose_name_plural = "новости"
        ordering = ["-published_at"]

    def __str__(self):
        return self.title

    def as_parser_item(self) -> dict:
        """Формат новости, который ждут агенты (как у RSS-парсера)."""
        return {
            "id": self.uid, "title": self.title, "summary": self.summary, "full_text": self.full_text,
            "content": self.full_text, "link": self.url, "source": self.source, "author": self.author,
            "published": self.published_at.isoformat() if self.published_at else "",
            "categories": self.categories, "language": self.language,
        }


class CompanyNews(models.Model):
    """Новость в ленте конкретной компании: оценка агента релевантности и состояние полного анализа."""

    STATUS_CHOICES = [("scored", "оценена"), ("not_scored", "не оценена"), ("error", "ошибка")]
    ANALYSIS_CHOICES = [("none", "не запускался"), ("queued", "в очереди"), ("running", "готовится"),
                        ("ready", "готов"), ("error", "ошибка")]

    company = models.ForeignKey(Company, on_delete=models.CASCADE, related_name="feed")
    news = models.ForeignKey(News, on_delete=models.CASCADE, related_name="assessments")

    status = models.CharField(max_length=16, choices=STATUS_CHOICES)
    score = models.IntegerField("значимость 0–100 (агент релевантности)", null=True, db_index=True)
    reason = models.TextField("почему важно (кратко)", blank=True)
    channels = models.JSONField("каналы влияния", default=list)
    evidence = models.JSONField("признаки компании и контрагентов", default=dict)
    warnings = models.JSONField(default=list)
    relevance_seconds = models.FloatField(default=0)

    analysis_status = models.CharField(max_length=16, choices=ANALYSIS_CHOICES, default="none")
    analysis = models.ForeignKey("news_analysis.AnalysisRun", null=True, blank=True, on_delete=models.SET_NULL)
    analysis_started_at = models.DateTimeField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "новость в ленте компании"
        verbose_name_plural = "ленты компаний"
        constraints = [models.UniqueConstraint(fields=["company", "news"], name="uniq_company_news")]
        ordering = ["-score", "-news__published_at"]

    def __str__(self):
        return f"{self.company_id} × {self.news_id}: {self.score}"


class FeedJob(models.Model):
    """Сборка ленты компании в фоне: оценка новостей, затем полный анализ топ-новостей."""

    STATUS_CHOICES = [("running", "собирается"), ("ready", "готова"), ("error", "ошибка")]

    company = models.ForeignKey(Company, on_delete=models.CASCADE, related_name="feed_jobs")
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default="running")
    stage = models.CharField("текущий шаг", max_length=255, blank=True)
    news_total = models.IntegerField(default=0)
    news_scored = models.IntegerField(default=0)
    analyses_total = models.IntegerField(default=0)
    analyses_done = models.IntegerField(default=0)
    error = models.TextField(blank=True)
    started_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        verbose_name = "сборка ленты"
        verbose_name_plural = "сборки лент"
        ordering = ["-started_at"]

    def __str__(self):
        return f"{self.company_id}: {self.status} ({self.stage})"
