"""Экспорт в собственную папку: python -B -m relevance_agent.export_schemas."""
import json
from pathlib import Path
from .contracts import (RelevanceInput, RelevanceOutput, ArticleResult,
                        ProfileResult, CounterpartiesResult, BankSearchResult)


from .single_contracts import SingleNewsInput, SingleNewsOutput


def main():
    destination = Path(__file__).parent / 'schemas'
    destination.mkdir(exist_ok=True)
    for model in (RelevanceInput, RelevanceOutput, ArticleResult, ProfileResult,
                  CounterpartiesResult, BankSearchResult, SingleNewsInput, SingleNewsOutput):
        schema = model.model_json_schema()
        schema['$schema'] = 'https://json-schema.org/draft/2020-12/schema'
        (destination / f'{model.__name__}.json').write_text(
            json.dumps(schema, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
