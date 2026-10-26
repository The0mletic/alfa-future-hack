"""Live tests on an explicitly chosen parser snapshot; no bank retrieval.

Default: the three news of live_news/selected.json. --news/--exclude/--max-items run other
items of a saved snapshot; at most 20 paid items per run.
"""
import json
import time
from pathlib import Path
from relevance_agent.graph import build_graph
from relevance_agent.loaders import load_news
from relevance_agent.model import load_model
from relevance_agent.single_contracts import SingleNewsInput

CHECKS = Path(__file__).resolve().parent
SOURCE = CHECKS / 'live_news'
MODEL_ID = 'deepseek/deepseek-v4.1-flash'
# Proxy IDs of the models allowed by the case (page 18), confirmed via GET /v1/models.
ALLOWED_MODELS = ('deepseek/deepseek-v4.1-flash', 'z-ai/glm-5.3-flash', 'qwen/qwen3.6-35b-a3b',
                  'google/gemma-4-26b-a4b-it', 'moonshotai/kimi-k2.6', 'qwen/qwen3-vl-30b-a3b-instruct')


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=SOURCE,
                        help='new directory inside relevance_agent/checks; selected.json is read from live_news')
    parser.add_argument('--model', default=MODEL_ID, choices=ALLOWED_MODELS)
    parser.add_argument('--news', type=Path, default=SOURCE / 'selected.json',
                        help='parser JSON or news list inside relevance_agent/checks')
    parser.add_argument('--exclude', default='', help='comma-separated news ids to skip')
    parser.add_argument('--max-items', type=int, default=3, choices=range(1, 21), metavar='1..20')
    parser.add_argument('--company-id', default='U_57706')
    args = parser.parse_args()
    root = args.output_dir.resolve()
    model_id = args.model
    if not root.is_relative_to(CHECKS):
        parser.error('output-dir must be inside relevance_agent/checks')
    if not args.news.resolve().is_relative_to(CHECKS):
        parser.error('news must be inside relevance_agent/checks')
    skip = {x.strip() for x in args.exclude.split(',') if x.strip()}
    selected = [n for n in load_news(args.news) if n.id not in skip]
    if not 1 <= len(selected) <= args.max_items:
        parser.error(f'{len(selected)} news selected; expected 1..{args.max_items} (raise --max-items explicitly)')
    if (root / 'results.json').exists():
        raise ValueError('Refusing to overwrite a previous paid test')
    root.mkdir(parents=True, exist_ok=True)
    ROOT = root
    client = load_model().model_copy(update={'model_name': model_id})
    calls = []

    class ObservedModel:
        model_name = model_id
        def invoke(self, messages):
            start = time.monotonic()
            try:
                reply = client.invoke(messages)
            except Exception as exc:
                calls.append({'seconds':round(time.monotonic()-start,2),
                              'error_type':type(exc).__name__, 'http_status':getattr(exc,'status_code',None)})
                raise
            calls.append({'seconds':round(time.monotonic()-start,2),
                          'finish_reason':reply.response_metadata.get('finish_reason'),
                          'usage':reply.usage_metadata, 'response':reply.content})
            return reply

    pipeline = build_graph(ObservedModel())
    summary = []
    for news in selected:
        calls.clear()
        req = SingleNewsInput(company_id=args.company_id, news=news)
        start = time.monotonic()
        result = pipeline.invoke({'request':req}, {'recursion_limit':10})['output']
        elapsed = round(time.monotonic()-start,2)
        (ROOT / f'{news.id}.result.json').write_text(result.model_dump_json(indent=2)+'\n')
        (ROOT / f'{news.id}.calls.json').write_text(json.dumps(calls,ensure_ascii=False,indent=2)+'\n')
        item = dict(id=news.id, title=news.title, url=news.link, published=news.published,
                    model=model_id, company_id=args.company_id, status=result.status, seconds=elapsed, model_calls=result.model_calls,
                    score=result.assessment.relevance_score if result.assessment else None,
                    reason=result.assessment.reason if result.assessment else None,
                    channels=[dict(channel=c.channel, shock=c.shock, horizon=c.horizon,
                                   exposure=c.exposure, impact=c.impact)
                              for c in result.assessment.channels] if result.assessment else None,
                    event_confidence=result.assessment.event_confidence if result.assessment else None,
                    total_tokens=sum(u.get('total_tokens',0) for u in result.usage),
                    article_source=result.article.text_source, article_status=result.article.status,
                    last_warning=result.warnings[-1] if result.warnings else None)
        summary.append(item)
        (ROOT / 'results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(item,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
