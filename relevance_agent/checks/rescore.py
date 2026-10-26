"""Re-score saved model answers with the current scoring.py; no network, no paid calls.

Usage: python -B -m relevance_agent.checks.rescore relevance_agent/checks/live_news_channels/deepseek ...
Uses the last model answer of each news (the one the graph kept) and the current profile.
"""
import json
import sys
from pathlib import Path
from relevance_agent.counterparties import CounterpartyTools
from relevance_agent.contracts import ArticleResult, NewsItem
from relevance_agent.graph import news_text, parse_json, score_analysis
from relevance_agent.loaders import load_company_profile, load_news

COMPANY = 'U_57706'  # used for results saved before company_id was recorded
SNAPSHOT = Path(__file__).resolve().parent / 'live_news' / 'rss_full_report.json'


def saved_news_text(directory, summary):
    """News text for shock-quote checks: the saved result, else the RSS snapshot.

    Large *.result.json files are not committed; without them an article the agent fetched
    itself (no upstream full_text) is reduced to title and summary.
    """
    path = directory / f"{summary['id']}.result.json"
    if path.exists():
        saved = json.loads(path.read_text())
        return news_text(NewsItem.model_validate(saved['news']), ArticleResult.model_validate(saved['article']))
    news = next(n for n in load_news(SNAPSHOT) if n.id == summary['id'])
    return news_text(news, ArticleResult(news_id=news.id, status='not_requested', requested_url=news.link))


def main():
    tools, contexts = CounterpartyTools(), {}
    for directory in map(Path, sys.argv[1:]):
        for summary in json.loads((directory / 'results.json').read_text()):
            company = summary.get('company_id', COMPANY)
            if company not in contexts:
                contexts[company] = (load_company_profile(company), *tools.collect_context(company))
            profile, links, profiles = contexts[company]
            calls = json.loads((directory / f"{summary['id']}.calls.json").read_text())
            answers = [c['response'] for c in calls if c.get('response')]
            row = dict(dir=directory.name, title=summary['title'][:40], saved=summary['score'])
            if not answers:
                row.update(rescored=None, note='нет ответа модели')
            else:
                try:
                    text = saved_news_text(directory, summary)
                    result, notes = score_analysis(parse_json(answers[-1]), profile, links, profiles, text)
                    row.update(rescored=result['relevance_score'], notes=notes,
                               channels=[(c.channel, c.shock, c.horizon, c.exposure, c.impact)
                                         for c in result['channels']])
                except ValueError as exc:
                    row.update(rescored=None, note=str(exc))
            print(json.dumps(row, ensure_ascii=False))


if __name__ == '__main__':
    main()
