#!/usr/bin/env python3
"""
rss_news_parser.py — парсер RSS-лент с выводом в формате,
удобном для подачи в LLM и мультиагентные системы.

Зависимости:
    pip install feedparser httpx beautifulsoup4 python-dateutil

Примеры:
    python rss_news_parser.py https://habr.com/ru/rss/news/?fl=ru https://lenta.ru/rss/news --format json --limit 30 --pretty

    python rss_news_parser.py "@feeds.txt" --format markdown --output news.md
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Iterable

import feedparser
import httpx
from bs4 import BeautifulSoup
from dateutil import parser as dateparser

USER_AGENT = "RSSNewsParser/1.0"
WS_RE = re.compile(r"\s+")


# ---------- Утилиты ----------

def clean_text(raw: str | None, max_len: int | None = None) -> str:
    """Убирает HTML-теги, нормализует пробелы, обрезает по длине."""
    if not raw:
        return ""
    text = BeautifulSoup(raw, "html.parser").get_text(" ", strip=True)
    text = WS_RE.sub(" ", text).strip()
    if max_len and len(text) > max_len:
        text = text[:max_len].rsplit(" ", 1)[0].rstrip(",.;:") + "…"
    return text


def _parse_date(entry) -> str:
    """Возвращает ISO-8601 UTC; пустая строка, если даты нет."""
    for key in ("published", "updated", "created"):
        value = entry.get(key)
        if not value:
            continue

        try:
            dt = dateparser.parse(value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)

            return dt.astimezone(timezone.utc).isoformat()
        except (ValueError, OverflowError):
            continue
    return ""


def _make_id(link: str, title: str) -> str:
    key = (link or title).strip().lower()
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


# ---------- Модель ----------

@dataclass
class NewsItem:
    id: str
    title: str
    summary: str
    link: str
    published: str  # ISO-8601 UTC
    source: str
    author: str = ""
    categories: list[str] = field(default_factory=list)
    language: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------- Парсинг ----------

def parse_feed(content: bytes, source_url: str, summary_max_len: int = 1000) -> list[NewsItem]:
    parsed = feedparser.parse(content)
    source_title = clean_text(parsed.feed.get("title", "")) if parsed.feed else ""
    source_title = source_title or source_url
    language = (parsed.feed.get("language", "") if parsed.feed else "") or ""

    items: list[NewsItem] = []
    for entry in parsed.entries:
        title = clean_text(entry.get("title", ""))
        if not title:
            continue
        summary = clean_text(
            entry.get("summary") or entry.get("description") or "",
            max_len=summary_max_len,
        )
        link = entry.get("link", "")
        categories = [
            clean_text(t.get("term", "")) for t in entry.get("tags", []) if t.get("term")
        ]
        items.append(
            NewsItem(
                id=_make_id(link, title),
                title=title,
                summary=summary,
                link=link,
                published=_parse_date(entry),
                source=source_title,
                author=clean_text(entry.get("author", "")),
                categories=categories,
                language=language,
            )
        )
    return items


async def _fetch_one(client: httpx.AsyncClient, url: str, summary_max_len: int) -> list[NewsItem]:
    try:
        resp = await client.get(url, timeout=15.0, follow_redirects=True)
        resp.raise_for_status()
        return parse_feed(resp.content, url, summary_max_len)
    except Exception as exc:
        print(f"[warn] {url}: {exc}", file=sys.stderr)
        return []


async def fetch_all(urls: list[str], summary_max_len: int, concurrency: int = 8) -> list[NewsItem]:
    sem = asyncio.Semaphore(concurrency)
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*",
    }

    async with httpx.AsyncClient(headers=headers) as client:
        async def _bounded(u: str) -> list[NewsItem]:
            async with sem:
                return await _fetch_one(client, u, summary_max_len)

        results = await asyncio.gather(*(_bounded(u) for u in urls))

    items: list[NewsItem] = []
    for r in results:
        items.extend(r)

    return items


# ---------- Постобработка ----------

def dedupe(items: Iterable[NewsItem]) -> list[NewsItem]:
    seen_ids: set[str] = set()
    seen_titles: set[str] = set()
    out: list[NewsItem] = []
    for it in items:
        norm_title = re.sub(r"\W+", "", it.title.lower())[:80]
        if it.id in seen_ids or norm_title in seen_titles:
            continue

        seen_ids.add(it.id)
        seen_titles.add(norm_title)
        out.append(it)
    return out


def sort_items(items: list[NewsItem]) -> list[NewsItem]:
    def key(it: NewsItem) -> datetime:
        try:
            return datetime.fromisoformat(it.published)
        except Exception:
            return datetime.min.replace(tzinfo=timezone.utc)

    return sorted(items, key=key, reverse=True)


# ---------- Форматтеры ----------

def to_json(items: list[NewsItem], pretty: bool = False) -> str:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(items),
        "items": [it.to_dict() for it in items],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None)


def to_markdown(items: list[NewsItem]) -> str:
    lines = [f"# Новости (всего: {len(items)})\n"]
    for i, it in enumerate(items, 1):
        date = it.published[:10] if it.published else "—"
        meta = [f"источник: {it.source}", f"дата: {date}"]
        if it.author:
            meta.append(f"автор: {it.author}")
        if it.categories:
            meta.append("теги: " + ", ".join(it.categories))
        lines.append(f"## {i}. {it.title}")
        lines.append(f"_{' · '.join(meta)}_")
        if it.summary:
            lines.append(it.summary)
        lines.append(f"[ссылка]({it.link})")
        lines.append("")
    return "\n".join(lines)


def to_compact(items: list[NewsItem], max_chars: int | None = None) -> str:
    out: list[str] = []
    total = 0
    for it in items:
        line = f"[{it.source}] {it.title}"
        if it.summary:
            line += f" — {it.summary}"
        line += f" <{it.link}>"
        if max_chars and total + len(line) > max_chars:
            break
        out.append(line)
        total += len(line) + 1
    return "\n".join(out)


# ---------- CLI ----------

def _load_feeds(args_feeds: list[str]) -> list[str]:
    expanded: list[str] = []
    for f in args_feeds:
        if f.startswith("@"):
            with open(f[1:], encoding="utf-8") as fh:
                expanded.extend(
                    ln.strip() for ln in fh
                    if ln.strip() and not ln.startswith("#")
                )
        else:
            expanded.append(f)
    return expanded


async def build_output(args) -> str:
    urls = _load_feeds(args.feeds)
    items = await fetch_all(urls, summary_max_len=args.summary_len, concurrency=args.concurrency)
    items = dedupe(items)
    items = sort_items(items)
    if args.limit:
        items = items[: args.limit]

    if args.format == "json":
        return to_json(items, pretty=args.pretty)
    if args.format == "markdown":
        return to_markdown(items)
    if args.format == "compact":
        return to_compact(items, max_chars=args.max_chars)
    raise SystemExit(f"unknown format: {args.format}")


def main() -> None:
    p = argparse.ArgumentParser(description="RSS parser for LLM / multi-agent pipelines")
    p.add_argument("feeds", nargs="+",
                   help="URL лент или @file.txt с одним URL на строку")
    p.add_argument("--format", choices=["json", "markdown", "compact"], default="json")
    p.add_argument("--limit", type=int, default=0, help="макс. новостей (0 = без лимита)")
    p.add_argument("--summary-len", type=int, default=3000, help="макс. длина summary")
    p.add_argument("--max-chars", type=int, default=None, help="для compact: обрезать вывод")
    p.add_argument("--pretty", action="store_true", help="JSON с отступами")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--output", help="файл для записи вместо stdout")
    args = p.parse_args()

    if args.limit == 0:
        args.limit = None

    out = asyncio.run(build_output(args))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(out)
        print(f"written to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(out)


if __name__ == "__main__":
    main()
