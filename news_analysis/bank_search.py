"""Поиск по базе знаний аналитики Альфа-Банка (applications/knowledge_base).

Обёртка над retrieve() из scripts/knowledge_base/search.py: тот же BM25-поиск,
плюс фильтр бесполезных для анализа фрагментов и компактный формат для модели.
"""
import importlib.util
import json
from functools import lru_cache
from pathlib import Path

from django.conf import settings

KB_DIR = Path(settings.BASE_DIR) / "applications" / "knowledge_base"
SEARCH_SCRIPT = Path(settings.BASE_DIR) / "scripts" / "knowledge_base" / "search.py"
MAX_TEXT_CHARS = 1500

# Графики без проверенного описания содержат только OCR-подписи осей — смысла для модели нет.
SKIP_QUALITY = {"chart_labels_only"}


@lru_cache(maxsize=1)
def _retrieve():
    spec = importlib.util.spec_from_file_location("kb_search", SEARCH_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.retrieve


@lru_cache(maxsize=1)
def _chunks() -> tuple[dict, ...]:
    with open(KB_DIR / "chunks.jsonl", encoding="utf-8") as f:
        return tuple(c for c in map(json.loads, f) if c["quality"] not in SKIP_QUALITY)


def search(query: str, top_k: int = 4, as_of: str | None = None) -> list[dict]:
    """Фрагменты аналитики, опубликованные не позже as_of (YYYY-MM-DD)."""
    hits = _retrieve()(query, list(_chunks()), top_k=top_k, as_of=as_of)
    return [compact(h) for h in hits]


def compact(chunk: dict) -> dict:
    text = chunk["text"]
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS].rsplit(" ", 1)[0] + "…"
    return {
        "id": chunk["chunk_id"],
        "document": chunk["document_title"],
        "published_at": chunk["published_at"] or chunk.get("publication_month"),
        "page": chunk["page_start"],
        "citation": chunk["citation"],
        "kind": chunk["kind"],
        "score": chunk.get("score"),
        "text": text,
    }
