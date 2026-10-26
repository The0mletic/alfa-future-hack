from django.db import models


class NewsAnalysisCache(models.Model):
    """Разбор новости агентом 1. Один на новость: не зависит от компании, поэтому кэшируется."""

    news_id = models.CharField(max_length=64, db_index=True)
    prompt_version = models.CharField(max_length=32)
    model = models.CharField(max_length=64)
    result = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "разбор новости"
        verbose_name_plural = "разборы новостей"
        constraints = [models.UniqueConstraint(fields=["news_id", "prompt_version", "model"],
                                               name="uniq_news_analysis_version")]


class AnalysisRun(models.Model):
    """Один прогон анализа «новость × компания» со всей трассировкой."""

    STATUS_CHOICES = [("ok", "готово"), ("low_confidence", "низкая уверенность"), ("error", "ошибка")]

    news_id = models.CharField(max_length=64, db_index=True)
    company_id = models.CharField(max_length=32, db_index=True)
    status = models.CharField(max_length=16, choices=STATUS_CHOICES)
    news = models.JSONField("исходная новость")
    news_analysis = models.JSONField("разбор новости", null=True)
    plan = models.JSONField("план аналитика влияния", null=True)
    impact = models.JSONField("итоговый анализ", null=True)
    critic_reports = models.JSONField("отчёты критика", default=list)
    frontend = models.JSONField("данные для шаблонов фронтенда", null=True)
    trace = models.JSONField("шаги агентов", default=list)
    revisions = models.IntegerField(default=0)
    cost = models.FloatField("стоимость, ед. API", default=0)
    duration_s = models.FloatField(default=0)
    error = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "прогон анализа"
        verbose_name_plural = "прогоны анализа"
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.news_id} × {self.company_id} ({self.status})"
