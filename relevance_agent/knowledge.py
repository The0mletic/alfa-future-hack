"""Read-only adapter for the team's prepared analytics (no OCR)."""
import json
from pathlib import Path
from scripts.knowledge_base.search import retrieve
from .contracts import BankFragment, BankMaterial
from .loaders import ROOT


class BankKnowledge:
    def __init__(self, path=ROOT / 'applications/knowledge_base/chunks.jsonl'):
        self.chunks = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]

    def search(self, query: str, as_of: str, top_k: int = 5) -> tuple[list[BankMaterial], list[dict]]:
        # Strict temporal cut: undated documents and historical steel are excluded.
        hits = retrieve(query, self.chunks, top_k=top_k, as_of=as_of[:10])
        grouped = {}
        for c in hits:
            material = grouped.setdefault(c['document_id'], BankMaterial(
                material_id=c['document_id'], title=c['document_title'],
                source_file=c['source_pdf'], document_date=c['published_at'],
                fragments=[BankFragment(fragment_id=c['chunk_id'], page=c['page_start'], text=c['text'])]))
            if material.fragments[0].fragment_id != c['chunk_id']:
                material.fragments.append(BankFragment(fragment_id=c['chunk_id'], page=c['page_start'], text=c['text']))
        metadata = [{k: c[k] for k in ('chunk_id', 'quality', 'warnings', 'page_start', 'page_end', 'citation')} for c in hits]
        return list(grouped.values()), metadata
