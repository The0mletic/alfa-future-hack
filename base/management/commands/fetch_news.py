import asyncio
import json
from django.core.management.base import BaseCommand
from django.utils import timezone
from base.models import News, Category, Company

from scripts.rss_full_parser import load_feeds, build_report


class Command(BaseCommand):
    help = 'Парсит RSS, прогоняет через AI и сохраняет в БД'

    def add_arguments(self, parser):
        parser.add_argument('--feeds-file', default='feeds.txt', help='Путь к файлу с лентами')

    def handle(self, *args, **options):
        self.stdout.write("Запуск парсера RSS...")

        # 1. Загружаем ленты
        feeds = load_feeds(options['feeds_file'])
        if not feeds:
            self.stdout.write(self.style.ERROR("Нет лент для парсинга"))
            return

        # 2. Парсим (запускаем асинхронный код)
        report = asyncio.run(build_report(feeds, max_items_per_feed=5))

        self.stdout.write(f"Спарсено {report['total_items']} новостей. Начинаем анализ...")

        # 3. Сохраняем в БД
        created_count = 0
        for source_data in report['sources']:
            for item in source_data['items']:
                # Пропускаем, если новость с таким title уже есть (простая дедупликация)
                if News.objects.filter(title=item['title']).exists():
                    continue

                # === ЗДЕСЬ ДОЛЖЕН БЫТЬ ВЫЗОВ ВАШЕГО AI-АГЕНТА ===
                # Он должен прочитать item['content_for_llm'] и профиль компании,
                # и вернуть JSON с relevance, risks, opportunities, why, facts и т.д.
                # Пока что мы используем заглушки (Hardcode), чтобы фронт ожил.
                # TODO: добавить LLM
                category, _ = Category.objects.get_or_create(
                    slug='macro', defaults={'label': 'Макроэкономика'}
                )

                news = News.objects.create(
                    title=item['title'],
                    summary=item['summary_rss'][:500] or "Описание отсутствует",
                    source=source_data['source'],
                    url=item['link'],
                    published_at=timezone.now(),  # В реальности парсить item['published']
                    category=category,
                    relevance=85,  # Заглушка
                    urgent=False,
                    why="Это событие может повлиять на ваши логистические цепочки.",
                    facts=["Факт 1 из новости", "Факт 2"],
                    body=[item['content_for_llm'][:1000]],  # Обрезаем текст
                    risks=[{"title": "Риск 1", "text": "Описание риска", "level": "high", "horizon": "1-3 мес"}],
                    opportunities=[
                        {"title": "Возможность 1", "text": "Описание", "level": "medium", "horizon": "Сейчас"}],
                    basis=["ОКВЭД 49.41", "Аналитика Альфа-Банка"],
                    sources=[{"name": source_data['source'], "title": "Оригинал", "url": item['link']}]
                )
                created_count += 1

        self.stdout.write(self.style.SUCCESS(f"Успешно добавлено {created_count} новых новостей"))