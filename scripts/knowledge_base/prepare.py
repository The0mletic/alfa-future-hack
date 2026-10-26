"""Render configured PDF regions for local OCR; never modify source PDFs.

python scripts/knowledge_base/prepare.py --work-dir /tmp/alfa-kb-ocr
Then run the compiled ocr.swift with WORK_DIR/regions.json and a new output path.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).with_name("layout.json")


def main():
    import pypdfium2 as pdfium

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=3.0)
    parser.add_argument("--export-visuals", action="store_true", help="Save table/chart crops inside the new knowledge_base folder")
    args = parser.parse_args()
    if args.scale <= 0:
        parser.error("scale must be positive")
    config = json.loads(CONFIG.read_text())
    verification = json.loads((ROOT / "applications/knowledge_base/source_verification.json").read_text())
    expected = {r["file"]: r["sha256"] for r in verification}
    args.work_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for spec in config["documents"]:
        source = ROOT / "applications/analytics" / spec["filename"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected[source.name]:
            raise ValueError(f"Source changed: {source}; review layout and provenance before rebuilding")
        pdf = pdfium.PdfDocument(source)
        if len(pdf) != len(spec["pages"]):
            raise ValueError(f"Page count mismatch: {source}")
        for page_number, regions in spec["pages"].items():
            page = pdf[int(page_number) - 1]
            bitmap = page.render(scale=args.scale)
            image = bitmap.to_pil().convert("RGB")
            for region in regions:
                x0, y0, x1, y1 = region["bbox"]
                crop = image.crop((round(x0 * image.width), round(y0 * image.height),
                                   round(x1 * image.width), round(y1 * image.height)))
                path = args.work_dir / (region["id"] + ".png")
                crop.save(path)
                if args.export_visuals and region["kind"] in ("table", "figure"):
                    figures = ROOT / "applications/knowledge_base/figures"
                    figures.mkdir(exist_ok=True)
                    crop.save(figures / (region["id"] + ".webp"), quality=92, method=6)
                manifest.append(dict(source=region["id"], page=int(page_number), image=str(path.resolve()),
                                     width=crop.width, height=crop.height))
            bitmap.close()
            page.close()
        pdf.close()
    (args.work_dir / "regions.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"Prepared {len(manifest)} regions in {args.work_dir}")


if __name__ == "__main__":
    main()
