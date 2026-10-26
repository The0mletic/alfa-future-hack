"""Compare real models on the frozen, previously authorized client context."""
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from relevance_agent.contracts import ArticleResult
from relevance_agent.single_contracts import SingleNewsInput, SingleNewsOutput
from relevance_agent.graph import score_news
from relevance_agent.model import load_model

ROOT = Path(__file__).resolve().parents[1]


def run(name, model_id, baseline):
    target = ROOT / 'runs' / f'{name}_first_live.json'
    if target.exists():
        raise ValueError(f'Refusing overwrite: {target.name}')
    request = SingleNewsInput(request_id=baseline.request_id, company_id=baseline.company_id,
                              news=baseline.news, as_of=baseline.as_of)
    cp = SimpleNamespace(collect_context=lambda cid: (baseline.counterparties, baseline.counterparty_profiles))
    kb = SimpleNamespace(search=lambda query, as_of: (baseline.bank_materials, baseline.bank_metadata))

    class FrozenArticle:
        def __init__(self, news, options): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def fetch_article(self, news_id):
            return baseline.article.model_copy(deep=True)

    client = load_model().model_copy(update={'model_name': model_id})
    calls = []

    class ObservedModel:
        model_name = model_id
        def invoke(self, messages):
            start = time.monotonic()
            digest = hashlib.sha256(json.dumps([m.content for m in messages], ensure_ascii=False).encode()).hexdigest()
            try:
                response = client.invoke(messages)
            except Exception as exc:
                calls.append({'elapsed_seconds': round(time.monotonic()-start, 2),
                              'prompt_sha256': digest, 'error_type': type(exc).__name__,
                              'http_status': getattr(exc, 'status_code', None)})
                raise
            calls.append({'elapsed_seconds': round(time.monotonic()-start, 2), 'prompt_sha256': digest,
                          'finish_reason': response.response_metadata.get('finish_reason'),
                          'response_chars': len(response.content), 'usage': response.usage_metadata})
            return response

    start = time.monotonic()
    result = score_news(request, ObservedModel(), counterparties=cp, knowledge=kb,
                        profile_loader=lambda cid: baseline.company_profile,
                        article_factory=FrozenArticle)
    target.write_text(result.model_dump_json(indent=2)+'\n')
    report = {'model': model_id, 'status': result.status, 'score': result.assessment.relevance_score if result.assessment else None,
              'elapsed_seconds': round(time.monotonic()-start, 2), 'calls': calls,
              'assessment': result.assessment.model_dump() if result.assessment else None,
              'last_warning': result.warnings[-1] if result.warnings else None}
    (ROOT / 'runs' / f'{name}_diagnostics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    return report


def main():
    baseline = SingleNewsOutput.model_validate_json((ROOT / 'runs/glm_first_live.json').read_text())
    models = [('gemma', 'google/gemma-4-26b-a4b-it'), ('deepseek', 'deepseek/deepseek-v4.1-flash')]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, name, mid, baseline) for name, mid in models]
        for future in futures:
            print(json.dumps(future.result(), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
