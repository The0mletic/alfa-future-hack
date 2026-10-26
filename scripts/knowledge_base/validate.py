"""Read-only integrity and retrieval checks for the generated knowledge base."""
import hashlib
import json
from pathlib import Path

from search import retrieve

ROOT = Path(__file__).resolve().parents[2]
KB = ROOT / "applications/knowledge_base"


def load(name):
    return [json.loads(line) for line in (KB / name).read_text().splitlines()]


def check(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    docs, blocks, chunks, tables = [load(n) for n in ("documents.jsonl", "blocks.jsonl", "chunks.jsonl", "tables.jsonl")]
    check(len(docs) == 9 and sum(d["pages"] for d in docs) == 64, "Expected 9 documents / 64 pages")
    block_ids = {b["block_id"] for b in blocks}
    chunk_ids = {c["chunk_id"] for c in chunks}
    check(len(block_ids) == len(blocks), "Duplicate blocks")
    check(len(chunk_ids) == len(chunks), "Duplicate chunks")
    for doc in docs:
        check(hashlib.sha256((ROOT / doc["source_pdf"]).read_bytes()).hexdigest() == doc["source_sha256"], "Source hash mismatch")
        pages = {b["page"] for b in blocks if b["document_id"] == doc["document_id"]}
        check(pages == set(range(1, doc["pages"] + 1)), "Lost page")
        check((KB / doc["markdown_path"]).is_file(), "Missing Markdown")
    for block in blocks:
        check(bool(block["text"].strip()), "Empty block")
        for name in ("image_path", "layout_path"):
            if name in block:
                check((KB / block[name]).is_file(), f"Missing asset: {block[name]}")
        x0, y0, x1, y1 = block["bbox"]
        check(0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1, "Invalid crop")
    for chunk in chunks:
        check(chunk["block_id"] in block_ids, "Broken chunk provenance")
        check(chunk["kind"] not in ("legal", "cover"), "Boilerplate leaked into retrieval")
        check(chunk["page_start"] == chunk["page_end"] == chunk["page"], "Page reference mismatch")
        for name in ("previous_chunk_id", "next_chunk_id"):
            check(chunk[name] is None or chunk[name] in chunk_ids, "Broken neighbor link")
        check(len(chunk["text"]) <= 5000 or chunk["oversized_atomic_table"], "Unexpected huge chunk")
    check(len(tables) == 21, "Lost table")
    for table in tables:
        check(table["table"] is not None, "Unstructured table remains")
        check(all(len(row) == len(table["table"]["headers"]) for row in table["table"]["rows"]), "Ragged table")
    # Regression checks for distinctions that directly affect a news analyst's conclusions.
    by_id = {t["block_id"]: t["table"] for t in tables}
    early = by_id["macro_7m26-p03-b02"]
    late = by_id["policy_rate_2026_09_11-p02-b01"]
    check(early["rows"][2][-2] == "13,5%" and late["rows"][2][-2] == "14,0%", "Forecast versions conflated")
    check(by_id["bond_brief_2026_09_14-p03-b02"]["rows"][0][1:] == ["-123", "-193", "-52"], "Lost negative cash-flow signs")
    check(by_id["bond_brief_2026_09_14-p02-b02"]["rows"][0][4] == "N/A", "Missing value changed to zero")
    for query, expected_doc in [("РусГидро EBITDA", "bond_brief_2026_09_14"), ("золото ФРС", "gold_august_2026"), ("ключевая ставка 14,0%", "policy_rate_2026_09_11")]:
        results = retrieve(query, chunks, top_k=5)
        check(any(c["document_id"] == expected_doc for c in results), f"Retrieval failed: {query}")
    dated = retrieve("ключевая ставка", chunks, top_k=1000, as_of="2026-08-01")
    check(bool(dated), "No dated retrieval results")
    check(all(c["published_at"] and c["published_at"] <= "2026-08-01" for c in dated), "Future or undated knowledge leaked")
    check(not any(c["historical"] for c in retrieve("сталь", chunks, top_k=1000)), "Historical data leaked by default")
    check(any(c["historical"] for c in retrieve("сталь", chunks, top_k=1000, include_historical=True)), "Historical corpus inaccessible")
    print(f"PASS: {len(docs)} documents, 64 pages, {len(tables)} structured tables, {len(chunks)} chunks; provenance and retrieval checks passed")


if __name__ == "__main__":
    main()
