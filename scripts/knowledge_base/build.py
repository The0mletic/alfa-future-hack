"""Build Markdown and retrieval JSONL from the checked-in local OCR cache.

No network, API keys or third-party Python packages are needed for this step.
Only applications/knowledge_base is written. Source PDFs are read and hash checked.
"""
import argparse
import hashlib
import json
import re
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
KB = ROOT / "applications/knowledge_base"
MAX_CHARS = 3600


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def visual_rows(lines):
    """Join fragments on the same baseline. Do not interleave separate regions/columns."""
    rows = []
    for line in sorted(lines, key=lambda l: ((l["bbox"][1] + l["bbox"][3]) / 2, l["bbox"][0])):
        if line["text"].strip() in {"A", "А"} and line["bbox"][0] > .85:
            continue  # Bank logo, not article content.
        y = (line["bbox"][1] + line["bbox"][3]) / 2
        height = line["bbox"][3] - line["bbox"][1]
        if rows and abs(rows[-1]["y"] - y) < .48 * min(height, rows[-1]["height"]):
            rows[-1]["lines"].append(line)
        else:
            rows.append(dict(y=y, height=height, lines=[line]))
    for row in rows:
        row["lines"].sort(key=lambda l: l["bbox"][0])
    return rows


def prose(lines):
    rows = visual_rows(lines)
    if not rows:
        return ""
    median_height = statistics.median(row["height"] for row in rows)
    paragraphs, current = [], []
    previous = None
    for row in rows:
        text = " ".join(l["text"].strip() for l in row["lines"])
        text = re.sub(r"[ \t]+", " ", text).strip()
        new = previous is not None and (
            row["y"] - previous["y"] > median_height * 1.9
            or row["height"] > median_height * 1.3
            or previous["height"] > median_height * 1.3
            or re.match(r"^[•□▪➢]", text)
        )
        if new and current:
            paragraphs.append(" ".join(current))
            current = []
        current.append(text)
        previous = row
    if current:
        paragraphs.append(" ".join(current))
    # Preserve source hyphens, decimal separators, dates, percentages and units.
    return "\n\n".join(paragraphs)


def positioned_text(lines):
    """Fixed-width layout keeps uncertain table cells / chart labels in spatial context."""
    result = []
    for row in visual_rows(lines):
        buffer = ""
        for line in row["lines"]:
            pos = round(line["bbox"][0] * 180)
            buffer += " " * max(1 if buffer else 0, pos - len(buffer)) + line["text"]
        result.append(buffer.rstrip())
    return "\n".join(result)


def markdown_table(table):
    def row(values):
        return "| " + " | ".join(str(v).replace("|", "\\|").replace("\n", "<br>") for v in values) + " |"
    return "\n".join([row(table["headers"]), row(["---"] * len(table["headers"]))] + [row(r) for r in table["rows"]])


def split_text(text, max_chars=MAX_CHARS):
    """Prefer paragraph/sentence boundaries; no dropped characters or arbitrary summarization."""
    parts = []
    while len(text) > max_chars:
        cut = text.rfind("\n\n", max_chars // 3, max_chars)
        if cut < 0:
            candidates = list(re.finditer(r"[.!?][ \n]", text[:max_chars]))
            cut = candidates[-1].end() if candidates else text.rfind(" ", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        parts.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        parts.append(text)
    return parts


def table_parts(table):
    groups, current = [], []
    for row in table["rows"]:
        candidate = {**table, "rows": current + [row]}
        if current and len(markdown_table(candidate)) > MAX_CHARS:
            groups.append(markdown_table({**table, "rows": current}))
            current = []
        current.append(row)
    if current:
        groups.append(markdown_table({**table, "rows": current}))
    # Every fragment repeats all column headers and the full note / unit / forecast legend.
    return [g + "\n\n" + table.get("notes", "") for g in groups]


def caption(lines, fallback):
    top = [l for l in lines if l["bbox"][1] < .16 and len(re.findall(r"[А-Яа-яA-Za-z]", l["text"])) >= 4]
    candidate = prose(top)
    # Do not treat a run of axis numbers as a title.
    if len(re.findall(r"[А-Яа-яA-Za-z]", candidate)) < 8:
        return fallback
    return candidate.replace("\n\n", " ")[:500]


def build():
    config = read_json(HERE / "layout.json")
    verified = read_json(HERE / "verified_tables.json")
    overrides_path = HERE / "text_corrections.json"
    corrections = read_json(overrides_path) if overrides_path.exists() else {}
    pages = read_jsonl(KB / "ocr/pages.jsonl")
    regions_list = read_jsonl(KB / "ocr/regions.jsonl")
    regions = {r["source"]: r for r in regions_list}
    if len(regions) != len(regions_list):
        raise ValueError("Duplicate OCR region IDs")
    provenance = {s["file"]: s for s in read_json(KB / "source_verification.json")}
    for sub in ("documents", "layouts"):
        (KB / sub).mkdir(exist_ok=True)
    docs, blocks, chunks, tables, figures = [], [], [], [], []
    for doc in config["documents"]:
        source_path = Path("applications/analytics") / doc["filename"]
        sha = hashlib.sha256((ROOT / source_path).read_bytes()).hexdigest()
        if sha != provenance[doc["filename"]]["sha256"]:
            raise ValueError(f"Source hash changed: {source_path}")
        source_pages = [p for p in pages if p["source"] == doc["filename"]]
        if sorted(p["page"] for p in source_pages) != list(range(1, len(doc["pages"]) + 1)):
            raise ValueError(f"Missing or duplicate source pages: {doc['id']}")
        source = dict(document_id=doc["id"], document_title=doc["title"], source_pdf=str(source_path),
                      source_sha256=sha, published_at=doc["published_at"],
                      publication_month=doc["publication_month"], publication_precision=doc["publication_precision"],
                      data_period=doc["data_period"], topics=doc["topics"],
                      historical=doc["id"] == "russian_steel_2021_04_15")
        md = [f"# {doc['title']}", f"Дата публикации: {doc['published_at'] or doc['publication_month'] + ' (точный день не указан)'}. Период данных: {doc['data_period']}.",
              f"Источник: [PDF](../../analytics/{doc['filename']}). SHA-256: `{sha}`.",
              "Текст распознан локально. Факты, оценки и прогнозы воспроизведены в контексте даты источника; это не текущие рыночные данные."]
        if source["historical"]:
            md.append("**Исторический обзор 2021 года. Не использовать его цены и прогнозы как актуальные для 2026 года.**")
        for page_number, specs in sorted(doc["pages"].items(), key=lambda x: int(x[0])):
            md.append(f"## Страница {page_number}")
            headings = [s for s in specs if s["title"] == "Заголовок"]
            section_title = prose(regions[headings[0]["id"]]["lines"]) if headings else doc["title"]
            for spec in specs:
                region = regions[spec["id"]]
                if not region["lines"]:
                    raise ValueError(f"Empty OCR region: {spec['id']}")
                kind, block_id = spec["kind"], spec["id"]
                title = spec["title"]
                text = prose(region["lines"])
                confidence = min(l["confidence"] for l in region["lines"])
                quality = "ocr_transcription"
                warnings = []
                parts = []
                if block_id in corrections:
                    for change in corrections[block_id]:
                        if change["before"] not in text:
                            raise ValueError(f"Correction no longer matches OCR: {block_id}: {change['before']}")
                        text = text.replace(change["before"], change["after"])
                if kind == "text":
                    # Narrow, documented OCR normalizations. Raw OCR is never changed.
                    text = text.replace("ОФ3", "ОФЗ")
                    text = re.sub(r"\b[Цц]ь\b", "ЦБ", text)
                if kind in ("figure", "table"):
                    layout_path = f"layouts/{block_id}.txt"
                    (KB / layout_path).write_text(positioned_text(region["lines"]) + "\n", encoding="utf-8")
                    image_path = f"figures/{block_id}.webp"
                    if not (KB / image_path).is_file():
                        raise ValueError(f"Missing visual source: {image_path}")
                if kind == "table":
                    if block_id in verified:
                        table = verified[block_id]
                        text = markdown_table(table) + "\n\n" + table.get("notes", "")
                        quality = table["verification"]
                        parts = table_parts(table)
                        table_data = table
                    else:
                        text = "```text\n" + positioned_text(region["lines"]) + "\n```"
                        quality = "ocr_layout_requires_cell_review"
                        warnings.append("Структура сохранена по координатам OCR. Перед расчетами проверить соответствие ячеек, значения, знаки и единицы по изображению.")
                        table_data = None
                        # Whole table is retained; don't split away multi-line headers.
                        parts = [text]
                    tables.append({**source, "block_id": block_id, "page": int(page_number),
                                   "title": title, "table": table_data, "layout_path": layout_path,
                                   "image_path": image_path, "quality": quality, "warnings": warnings})
                elif kind == "figure":
                    title = caption(region["lines"], title)
                    sources = [l["text"] for l in region["lines"] if re.search(r"[Ии]сточник", l["text"])]
                    description = spec.get("verified_description", "Сохранены изображение, подписи и расположение OCR-меток. Числовой ряд по графической линии не восстановлен.")
                    text = title + "\n\n" + description
                    if sources:
                        text += "\n\n" + " ".join(sources)
                    quality = "visual_description_checked" if "verified_description" in spec else "chart_labels_only"
                    warnings.append("Числа на осях и отдельные OCR-метки не являются готовым временным рядом. Для точных значений открыть изображение.")
                    figures.append({**source, "block_id": block_id, "page": int(page_number), "title": title,
                                    "description": description, "image_path": image_path,
                                    "ocr_layout_path": layout_path, "quality": quality})
                if confidence < .5:
                    warnings.append("В OCR есть строки с уверенностью ниже 0,5; уверенность OCR не гарантирует правильность остальных строк.")
                block = {**source, "block_id": block_id, "page": int(page_number), "bbox": spec["bbox"],
                         "kind": kind, "title": title, "section_title": section_title, "text": text, "quality": quality,
                         "warnings": warnings, "ocr_min_confidence": confidence,
                         "statement_type": "source_mixed_facts_estimates_forecasts_opinions",
                         "contains_forecast_terms": bool(re.search(r"прогноз|ожида|полага|202[67]П", text, re.I))}
                if kind in ("figure", "table"):
                    block.update(image_path=image_path, layout_path=layout_path)
                blocks.append(block)
                md.extend([f"### {title}", f"<!-- block_id: {block_id}; quality: {quality} -->", text])
                if warnings:
                    md.append("> " + " ".join(warnings))
                if kind in ("figure", "table"):
                    md.append(f"[Изображение источника](../{image_path}) · [OCR с расположением строк](../{layout_path})")
                if kind in ("legal", "cover"):
                    continue
                if not parts:
                    parts = split_text(text)
                for index, part in enumerate(parts, 1):
                    chunks.append({**block, "chunk_id": f"{block_id}-c{index:02d}", "text": part,
                                   "part": index, "parts": len(parts), "page_start": int(page_number),
                                   "page_end": int(page_number), "citation": f"{doc['filename']}, стр. {page_number}",
                                   "oversized_atomic_table": len(part) > MAX_CHARS and kind == "table"})
        document_path = f"documents/{doc['id']}.md"
        (KB / document_path).write_text("\n\n".join(md) + "\n", encoding="utf-8")
        docs.append({**source, "pages": len(doc["pages"]), "markdown_path": document_path,
                     "source_folder_url": config["source_folder_url"], "drive_byte_match_verified_on": config["verified_on"],
                     "date_basis": "publication date printed in source" if doc["published_at"] else "cover month only; exact day unknown"})
    for i, chunk in enumerate(chunks):
        prev = chunks[i-1] if i else None
        nxt = chunks[i+1] if i+1 < len(chunks) else None
        chunk["previous_chunk_id"] = prev["chunk_id"] if prev and prev["document_id"] == chunk["document_id"] else None
        chunk["next_chunk_id"] = nxt["chunk_id"] if nxt and nxt["document_id"] == chunk["document_id"] else None
    for filename, data in [("documents.jsonl", docs), ("blocks.jsonl", blocks), ("chunks.jsonl", chunks), ("tables.jsonl", tables), ("figures.jsonl", figures)]:
        write_jsonl(KB / filename, data)
    report = dict(schema_version=1, documents=len(docs), pages=sum(d["pages"] for d in docs),
                  regions=len(blocks), chunks=len(chunks), tables=len(tables), figures=len(figures),
                  table_quality=dict(Counter(t["quality"] for t in tables)),
                  chart_quality=dict(Counter(f["quality"] for f in figures)),
                  low_confidence_blocks=[b["block_id"] for b in blocks if b["ocr_min_confidence"] < .5],
                  all_drive_files_match_local=all(s["drive_matches_local"] for s in provenance.values()),
                  limitations=["OCR не равен ручной посимвольной проверке всех страниц.",
                               "Таблицы сверены визуально; внешняя достоверность значений источника не проверялась.",
                               "Графические временные ряды не оцифрованы; сохранены изображения и OCR-метки.",
                               "Факты и прогнозы отражают документы кейса, внешняя актуальность не проверялась."])
    (KB / "quality_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("documents", "pages", "regions", "chunks", "tables", "figures", "table_quality")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    build()
