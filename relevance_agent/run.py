"""CLI: one news from parser JSON -> one client assessment."""
import argparse
from pathlib import Path
from .loaders import load_news
from .model import load_model
from .single_contracts import SingleNewsInput
from .graph import score_news


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--company-id', required=True)
    parser.add_argument('--news', type=Path, required=True)
    parser.add_argument('--news-id', required=True)
    parser.add_argument('--as-of')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    path = args.output.resolve()
    root = Path(__file__).resolve().parent
    if not path.is_relative_to(root) or path.exists():
        parser.error('Нужен новый выходной файл внутри relevance_agent/')
    news = next((n for n in load_news(args.news) if n.id == args.news_id), None)
    if news is None:
        parser.error('news-id отсутствует во входном файле')
    kwargs = {'as_of': args.as_of} if args.as_of else {}
    request = SingleNewsInput(company_id=args.company_id, news=news, **kwargs)
    try:
        model = load_model()
    except ValueError as exc:
        parser.error(str(exc))
    result = score_news(request, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(result.model_dump_json(indent=2) + '\n')
    print(f'{result.status}: {path}')
    if result.status == 'not_scored':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
