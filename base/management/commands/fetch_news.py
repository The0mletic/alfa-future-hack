"""Загрузка свежих новостей из RSS-лент в базу.

    python manage.py fetch_news                  # по 5 статей с каждой ленты из feeds.txt
    python manage.py fetch_news --max-items 10

Анализ здесь не запускается: ленту для компании собирает base/feed.py при входе.
"""
from pathlib import Path

from django.core.management.base import BaseCommand

from base.news_ingest import FEEDS_FILE, fetch_and_store


class Command(BaseCommand):
    help = "Парсит RSS-ленты с полным текстом статей и сохраняет новые новости в базу"

    def add_arguments(self, parser):
        parser.add_argument("--feeds-file", default=str(FEEDS_FILE), help="файл со списком лент")
        parser.add_argument("--max-items", type=int, default=5, help="сколько статей брать с одной ленты")

    def handle(self, *args, **options):
        self.stdout.write("Парсим ленты…")
        result = fetch_and_store(options["max_items"], Path(options["feeds_file"]))
        self.stdout.write(self.style.SUCCESS(
            f"Лент: {result['feeds']}, ответили: {result['sources']}, новостей: {result['parsed']}, "
            f"новых в базе: {result['created']}"))
