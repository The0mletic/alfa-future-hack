from django.contrib import admin
from .models import Company, Category, News

@admin.register(Company)
class CompanyAdmin(admin.ModelAdmin):
    list_display = ('name', 'inn', 'industry')

@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
    list_display = ('label', 'slug')

@admin.register(News)
class NewsAdmin(admin.ModelAdmin):
    list_display = ('title', 'category', 'source', 'relevance', 'urgent', 'published_at')
    list_filter = ('category', 'urgent')
    search_fields = ('title', 'summary')
