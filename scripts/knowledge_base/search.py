"""Small offline lexical retrieval example. Does not call an LLM or alter the site.

python scripts/knowledge_base/search.py 'ключевая ставка стоимость кредита' --top-k 3
python scripts/knowledge_base/search.py 'золото ФРС' --as-of 2026-08-31
"""
import argparse
import json
import math
import re
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STOP = set("и в во на по за из к с со для не что это как до от при мы он она они его ее их или а о об но то же только более все г гг год года руб млн млрд тыс источник источники альфа банк".split())


def tokens(text):
    words = re.findall(r"[а-яa-z0-9]+", text.lower().replace("ё", "е"))
    # Lightweight prefix normalization handles common Russian inflections; not semantic search.
    return [word[:6] if re.fullmatch(r"[а-я]{7,}", word) else word for word in words if word not in STOP and len(word) > 1]


def retrieve(query, chunks, top_k=5, as_of=None, include_historical=False, include_unverified_tables=False):
    candidates = []
    for chunk in chunks:
        if chunk["historical"] and not include_historical:
            continue
        if chunk["quality"] == "ocr_layout_requires_cell_review" and not include_unverified_tables:
            continue
        if as_of:
            published = chunk["published_at"]
            if published is None:
                # A cover month can be the reporting period, not the publication date.
                # Don't invent a date or leak unknown-date material into backtests.
                continue
            if published > as_of:
                continue
        candidates.append(chunk)
    if not candidates:
        return []
    # Body dominates scoring. Title/topic matches add a bounded bonus and cannot swamp content.
    counts = [Counter(tokens(c["title"] + " " + c["text"])) for c in candidates]
    dfs = Counter(t for counts_i in counts for t in counts_i)
    lengths = [sum(c.values()) for c in counts]
    avg = sum(lengths) / len(lengths) or 1
    query_tokens = set(tokens(query))
    results = []
    for chunk, counts_i, length in zip(candidates, counts, lengths):
        score = 0.0
        for term in query_tokens:
            tf = counts_i[term]
            idf = math.log(1 + (len(candidates) - dfs[term] + .5) / (dfs[term] + .5))
            score += idf * tf * 2.2 / (tf + 1.2 * (.25 + .75 * length / avg))
        if score > 0:
            results.append({**chunk, "score": round(score, 5)})
    return sorted(results, key=lambda c: (-c["score"], c["chunk_id"]))[:top_k]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--as-of", type=date.fromisoformat)
    parser.add_argument("--include-historical", action="store_true")
    parser.add_argument("--include-unverified-tables", action="store_true")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("top-k must be positive")
    chunks = [json.loads(line) for line in (ROOT / "applications/knowledge_base/chunks.jsonl").read_text().splitlines()]
    found = retrieve(args.query, chunks, args.top_k, args.as_of.isoformat() if args.as_of else None,
                     args.include_historical, args.include_unverified_tables)
    keys = ("chunk_id", "document_title", "published_at", "publication_month", "page_start", "citation", "quality", "score", "text", "warnings", "source_pdf")
    print(json.dumps([{k: c[k] for k in keys} for c in found], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
