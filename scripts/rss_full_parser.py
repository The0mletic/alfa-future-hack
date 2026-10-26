#!/usr/bin/env python3
"""
pip install aiohttp feedparser trafilatura newspaper3k beautifulsoup4 requests lxml python-dateutil lxml_html_clean
rss_full_parser.py — парсит RSS-ленты и загружает полный текст статей.

Для каждой ленты из списка извлекаются элементы, затем по ссылке
загружается HTML статьи и вытаскивается основной контент.

Результат — JSON, сгруппированный по источникам.

Быстрый запуск python rss_full_parser.py результат в rss_full_report.json
"""
import asyncio
import json
import re
import ssl
import sys
from datetime import datetime, timezone
from urllib.parse import urlparse

import aiohttp
import certifi
import feedparser
import trafilatura
from bs4 import BeautifulSoup
from dateutil import parser as dateparser

# Попытка импорта newspaper3k (опционально)
try:
    from newspaper import Article

    HAS_NEWSPAPER = True
except ImportError:
    HAS_NEWSPAPER = False

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
}

# Список RSS-лент (дубликаты удалены)
FEEDS = [
    "http://www.cbr.ru/rss/RssNews",
    "http://www.cbr.ru/rss/eventrss",
    "http://www.cbr.ru/rss/RssCurrency",
    "https://rosstat.gov.ru/opendata",
    "https://www.rbc.ru/rss",
    "http://www.vedomosti.ru/rss/news",
    "https://www.vedomosti.ru/rss/rubric/economics/macro.xml",
    "https://www.kommersant.ru/rss/news.xml",
    "https://www.kommersant.ru/rss/section-economics.xml",
    "https://www.kommersant.ru/rss/section-business.xml",
    "http://rss.metalinfo.ru/",
    "https://vc.ru/rss/all",
    "https://www.cnews.ru/inc/rss/news.xml",
    "https://www.reutersagency.com/feed/",
    "https://feeds.bloomberg.com/markets/news.rss",
    "https://www.federalreserve.gov/feeds/press_all.xml",
    "https://www.ecb.europa.eu/rss/press.html",
    "http://feeds.bbci.co.uk/news/business/rss.xml",
    "https://www.theguardian.com/business/economics/rss",
    "https://www.vedomosti.ru/rss/rubric/economics",
]

# ---------- Утилиты ----------

def clean_text(raw: str | None, max_len: int | None = None) -> str:
    """Убирает HTML-теги, нормализует пробелы, обрезает."""
    if not raw:
        return ""
    text = BeautifulSoup(raw, "html.parser").get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    if max_len and len(text) > max_len:
        text = text[:max_len].rsplit(" ", 1)[0].rstrip(",.;:") + "…"
    return text


def parse_date(entry) -> str:
    """Возвращает ISO-8601 UTC или пустую строку."""
    for key in ("published", "updated", "created"):
        value = entry.get(key)
        if value:
            try:
                dt = dateparser.parse(value)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc).isoformat()
            except (ValueError, OverflowError):
                continue
    return ""


def source_name(url: str) -> str:
    """Человекочитаемое имя источника по домену."""
    domain = urlparse(url).netloc.lower()
    mapping = {
        "cbr.ru": "Банк России",
        "rosstat.gov.ru": "Росстат",
        "rbc.ru": "РБК",
        "vedomosti.ru": "Ведомости",
        "kommersant.ru": "Коммерсантъ",
        "metalinfo.ru": "Metalinfo",
        "vc.ru": "vc.ru",
        "cnews.ru": "CNews",
        "reutersagency.com": "Reuters",
        "bloomberg.com": "Bloomberg",
        "federalreserve.gov": "ФРС США",
        "ecb.europa.eu": "ЕЦБ",
        "bbci.co.uk": "BBC",
        "theguardian.com": "The Guardian",
    }
    for key, name in mapping.items():
        if key in domain:
            return name
    return domain


# ---------- Извлечение полного текста ----------

def extract_with_trafilatura(html: str, url: str) -> str:
    """Основной экстрактор — trafilatura."""
    try:
        text = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
        )
        return text or ""
    except Exception as exc:
        print(f"[trafilatura] {url}: {exc}", file=sys.stderr)
        return ""


def extract_with_newspaper(url: str) -> str:
    """Fallback — newspaper3k."""
    if not HAS_NEWSPAPER:
        return ""
    try:
        article = Article(url, language="ru")
        article.download()
        article.parse()
        return article.text or ""
    except Exception as exc:
        print(f"[newspaper] {url}: {exc}", file=sys.stderr)
        return ""


def extract_with_bs4(html: str) -> str:
    """Последний fallback — эвристики BeautifulSoup."""
    soup = BeautifulSoup(html, "html.parser")
    # Удаляем скрипты, стили, навигацию
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()

    # Пробуем найти контейнер статьи
    candidates = []
    for selector in [
        "article",
        "div.article__text",
        "div.article-body",
        "div.story-body",
        "div[itemprop='articleBody']",
        "div.content",
        "main",
    ]:
        candidates.extend(soup.select(selector))

    if candidates:
        # Берём самый длинный текстовый блок
        best = max(candidates, key=lambda el: len(el.get_text(strip=True)))
        return best.get_text(" ", strip=True)

    # Иначе — все параграфы
    paragraphs = soup.find_all("p")
    return " ".join(p.get_text(" ", strip=True) for p in paragraphs)


async def fetch_article_text(
        session: aiohttp.ClientSession, url: str, timeout: int = 20
) -> str:
    """Загружает страницу и извлекает основной текст."""
    try:
        async with session.get(url, timeout=timeout) as resp:
            if resp.status != 200:
                print(f"[warn] {url} -> HTTP {resp.status}", file=sys.stderr)
                return ""
            html = await resp.text()
    except Exception as exc:
        print(f"[warn] {url}: {exc}", file=sys.stderr)
        return ""

    # 1) trafilatura
    text = extract_with_trafilatura(html, url)
    if text and len(text) > 100:
        return clean_text(text)

    # 2) newspaper3k (синхронный — выносим в поток)
    text = await asyncio.to_thread(extract_with_newspaper, url)
    if text and len(text) > 100:
        return clean_text(text)

    # 3) BeautifulSoup
    text = extract_with_bs4(html)
    return clean_text(text)


# ---------- Обработка одной ленты ----------

async def process_feed(
        session: aiohttp.ClientSession,
        feed_url: str,
        max_items: int = 15,
) -> dict:
    """Скачивает RSS, извлекает элементы, для каждого грузит полный текст."""
    print(f"[feed] {feed_url}", file=sys.stderr)

    try:
        async with session.get(feed_url, timeout=20) as resp:
            if resp.status != 200:
                print(f"[feed-error] {feed_url} -> HTTP {resp.status}", file=sys.stderr)
                return {"source": source_name(feed_url), "feed_url": feed_url, "items": []}
            content = await resp.read()
    except Exception as exc:
        print(f"[feed-error] {feed_url}: {exc}", file=sys.stderr)
        return {"source": source_name(feed_url), "feed_url": feed_url, "items": []}

    parsed = feedparser.parse(content)
    entries = parsed.entries[:max_items]

    async def build_item(entry) -> dict | None:
        title = clean_text(entry.get("title", ""))
        link = entry.get("link", "")
        if not title or not link:
            return None

        # Анонс из RSS (может быть пустым)
        summary_rss = clean_text(
            entry.get("summary") or entry.get("description") or ""
        )

        # Полный текст статьи
        full_text = await fetch_article_text(session, link)

        # Если полный текст пуст — используем summary из RSS
        content = full_text or summary_rss

        return {
            "title": title,
            "link": link,
            "published": parse_date(entry),
            "author": clean_text(entry.get("author", "")),
            "summary_rss": summary_rss,
            "full_text": full_text,
            "content_for_llm": content,  # то, что реально пойдёт в модель
            "categories": [
                clean_text(t.get("term", ""))
                for t in entry.get("tags", [])
                if t.get("term")
            ],
        }

    # Параллельно обрабатываем до 5 статей из ленты
    sem = asyncio.Semaphore(5)

    async def bounded(entry):
        async with sem:
            return await build_item(entry)

    items = await asyncio.gather(*(bounded(e) for e in entries))
    items = [i for i in items if i is not None]

    return {
        "source": source_name(feed_url),
        "feed_url": feed_url,
        "items": items,
    }


# ---------- Основной пайплайн ----------

async def build_report(feeds: list[str], max_items_per_feed: int = 15) -> dict:
    """Обрабатывает все ленты и собирает отчёт."""

    ssl_context = ssl.create_default_context(cafile=certifi.where())

    # Передаем этот контекст в TCPConnector
    connector = aiohttp.TCPConnector(limit=10, ssl=ssl_context)
    timeout = aiohttp.ClientTimeout(total=60)

    async with aiohttp.ClientSession(
            headers=HEADERS, connector=connector, timeout=timeout
    ) as session:
        tasks = [
            process_feed(session, feed, max_items_per_feed) for feed in feeds
        ]
        results = await asyncio.gather(*tasks)

    # Убираем пустые источники
    results = [r for r in results if r["items"]]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_sources": len(results),
        "total_items": sum(len(r["items"]) for r in results),
        "sources": results,
    }


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="RSS parser with full-text article extraction for LLM pipelines"
    )
    p.add_argument(
        "--max-items",
        type=int,
        default=10,
        help="Максимум статей с одной ленты (по умолчанию 10)",
    )
    p.add_argument(
        "--output",
        default="rss_full_report.json",
        help="Файл для сохранения JSON (по умолчанию rss_full_report.json)",
    )
    p.add_argument(
        "--pretty",
        action="store_true",
        help="Форматировать JSON с отступами",
    )
    args = p.parse_args()

    report = asyncio.run(build_report(FEEDS, max_items_per_feed=args.max_items))

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2 if args.pretty else None)

    print(
        f"Готово: {report['total_items']} новостей из {report['total_sources']} источников -> {args.output}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
