"""Загрузка новостей: RSS-парсер (scripts/rss_full_parser.py) → единый формат агентов → база."""
import asyncio
from datetime import datetime
from pathlib import Path

from django.conf import settings

from relevance_agent.parser_adapter import normalize_report
from scripts.rss_full_parser import build_report, load_feeds

from .models import News

FEEDS_FILE = Path(settings.BASE_DIR) / "feeds.txt"


def parse_published(value: str):
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None
    except ValueError:
        return None


def fetch_and_store(max_items_per_feed: int = 5, feeds_file: Path = FEEDS_FILE) -> dict:
    """Парсит ленты и сохраняет новые новости. ID — sha1 ссылки, как у парсера и агентов."""
    feeds = load_feeds(str(feeds_file))
    report = asyncio.run(build_report(feeds, max_items_per_feed=max_items_per_feed))
    items = normalize_report(report)
    known = set(News.objects.filter(uid__in=[i["id"] for i in items]).values_list("uid", flat=True))
    created = 0
    for item in items:
        if item["id"] in known:
            continue
        News.objects.create(
            uid=item["id"], title=item["title"][:500], summary=item["summary"], full_text=item["full_text"],
            url=item["link"], source=item["source"], author=item["author"][:255],
            categories=item["categories"], language=item.get("language") or "",
            published_at=parse_published(item["published"]),
        )
        created += 1
    return {"feeds": len(feeds), "sources": report["total_sources"], "parsed": len(items), "created": created}
