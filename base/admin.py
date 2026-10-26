from django.contrib import admin

from .models import CompanyNews, FeedJob, News


@admin.register(News)
class NewsAdmin(admin.ModelAdmin):
    list_display = ("title", "source", "published_at", "fetched_at")
    search_fields = ("title", "summary", "uid")
    list_filter = ("source",)


@admin.register(CompanyNews)
class CompanyNewsAdmin(admin.ModelAdmin):
    list_display = ("company", "news", "score", "status", "analysis_status")
    list_filter = ("status", "analysis_status")
    search_fields = ("company__company_id", "news__title")


@admin.register(FeedJob)
class FeedJobAdmin(admin.ModelAdmin):
    list_display = ("company", "status", "stage", "news_scored", "news_total", "analyses_done", "started_at")
    list_filter = ("status",)
