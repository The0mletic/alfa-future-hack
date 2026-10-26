from django.contrib import admin

from .models import AnalysisRun, NewsAnalysisCache


@admin.register(AnalysisRun)
class AnalysisRunAdmin(admin.ModelAdmin):
    list_display = ("news_id", "company_id", "status", "revisions", "cost", "duration_s", "created_at")
    list_filter = ("status",)
    search_fields = ("news_id", "company_id")


@admin.register(NewsAnalysisCache)
class NewsAnalysisCacheAdmin(admin.ModelAdmin):
    list_display = ("news_id", "model", "prompt_version", "created_at")
