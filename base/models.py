from django.db import models
from django.utils import timezone


class Company(models.Model):
    name = models.CharField(max_length=255)
    initials = models.CharField(max_length=10)
    inn = models.CharField(max_length=20)
    client_since = models.CharField(max_length=4)
    industry = models.CharField(max_length=255)
    okved = models.CharField(max_length=20)
    region = models.CharField(max_length=255)
    revenue = models.CharField(max_length=50)
    employees = models.CharField(max_length=50)
    foreign_trade = models.CharField(max_length=255)
    products = models.JSONField(default=list)  # ['Кредиты', 'Лизинг', ...]

    def __str__(self):
        return self.name


class Category(models.Model):
    slug = models.SlugField(unique=True)
    label = models.CharField(max_length=100)

    def __str__(self):
        return self.label


class News(models.Model):
    # Основные данные
    title = models.CharField(max_length=500)
    summary = models.TextField()
    source = models.CharField(max_length=255)
    url = models.URLField(max_length=1000, blank=True, null=True)
    published_at = models.DateTimeField(default=timezone.now)

    # Категория и теги
    category = models.ForeignKey(Category, on_delete=models.SET_NULL, null=True, related_name='news')

    # AI-аналитика (заполняется агентом)
    relevance = models.IntegerField(default=0)
    read_time = models.CharField(max_length=20, default='3 мин')
    urgent = models.BooleanField(default=False)
    urgent_reason = models.CharField(max_length=255, blank=True)

    # Глубокий контекст (сохраняем как JSON, чтобы совпадало с mock_data)
    why = models.TextField(blank=True, help_text="Почему это важно для компании")
    facts = models.JSONField(default=list)  # list[str]
    body = models.JSONField(default=list)  # list[str] (абзацы)
    bank_view = models.JSONField(default=dict, blank=True)  # {text, author, doc, date}
    risks = models.JSONField(default=list)  # list[dict]
    opportunities = models.JSONField(default=list)  # list[dict]
    basis = models.JSONField(default=list)  # list[str]
    sources = models.JSONField(default=list)  # list[dict]

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-published_at', '-relevance']

    def __str__(self):
        return self.title
