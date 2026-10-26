import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from relevance_agent.articles import ArticleTools
from relevance_agent.contracts import RunOptions
from relevance_agent.loaders import load_news, InputDataError
from relevance_agent.tests.test_graph import FakeModel, answer, request

BASE = Path(__file__).resolve().parents[1]


class ParserAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=BASE)
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'report.json'
        self.item = dict(title='Учебная новость', link='https://example.org/story', published='',
                         author='', summary_rss='Анонс', full_text='Полный учебный текст. ' * 20,
                         categories=['экономика'])
        self.item['content_for_llm'] = self.item['full_text']
        self.report = dict(generated_at='2026-10-26T12:00:00Z', total_sources=1, total_items=1,
                           sources=[dict(source='Тест', feed_url='https://example.org/rss', items=[self.item])])

    def load(self):
        self.path.write_text(json.dumps(self.report, ensure_ascii=False))
        return load_news(self.path)

    def test_normalize_source_id_and_separate_texts(self):
        news = self.load()[0]
        self.assertEqual(news.summary, 'Анонс')
        self.assertEqual(news.full_text, self.item['full_text'])
        self.assertEqual(news.source, 'Тест')
        self.assertEqual(news.id, hashlib.sha1(self.item['link'].lower().encode()).hexdigest()[:16])
        self.assertEqual(news.published, '')

    def test_full_text_skips_download_even_with_zero_network_budget(self):
        news = self.load()[0]
        with ArticleTools([news], RunOptions(max_article_fetches=0, max_article_chars=25)) as tools:
            with patch.object(tools, '_download', side_effect=AssertionError('Network must not be called')):
                result = tools.fetch_article(news.id)
        self.assertEqual(result.status, 'ok')
        self.assertEqual(result.text_source, 'upstream_parser')
        self.assertIsNone(result.final_url)
        self.assertEqual(result.text, news.full_text[:25])
        self.assertTrue(result.truncated)

    def test_summary_fallback_is_not_claimed_as_full_text(self):
        self.item['full_text'] = ''
        self.item['content_for_llm'] = self.item['summary_rss']
        news = self.load()[0]
        with ArticleTools([news]) as tools:
            with patch.object(tools, '_download', return_value=('https://example.org/final', b'<article><p>' + b'Actual article text. ' * 20 + b'</p></article>', None)) as download:
                result = tools.fetch_article(news.id)
        download.assert_called_once()
        self.assertEqual(result.text_source, 'agent_fetch')
        self.assertEqual(result.final_url, 'https://example.org/final')
        self.assertNotEqual(result.text, news.summary)

    def test_counts_and_dates_rejected(self):
        original = copy.deepcopy(self.report)
        for key, value in [('total_items', 2), ('total_sources', 2), ('generated_at', 'yesterday')]:
            self.report = dict(original, **{key: value})
            with self.assertRaises(InputDataError):
                self.load()

    def test_conflicting_content_rejected(self):
        self.item['content_for_llm'] = 'Несогласованный текст'
        with self.assertRaises(InputDataError):
            self.load()

    def test_duplicate_articles_are_not_silently_scored_twice(self):
        self.report['sources'][0]['items'].append(copy.deepcopy(self.item))
        self.report['total_items'] = 2
        with self.assertRaisesRegex(InputDataError, 'Повторяющиеся'):
            self.load()

    def test_graph_receives_full_text_once_with_provenance(self):
        model = FakeModel([answer()])
        req = request()
        req.news = self.load()[0]
        from types import SimpleNamespace
        from relevance_agent.graph import build_graph
        from relevance_agent.contracts import CompanyProfile, CounterpartiesResult
        cp = SimpleNamespace(collect_context=lambda cid: (CounterpartiesResult(company_id=cid, links=[], complete=True, limitations=[]), []))
        pipeline = build_graph(model, counterparties=cp, knowledge=SimpleNamespace(search=lambda *args: ([], [])),
            profile_loader=lambda cid: CompanyProfile(company_id=cid, industry={'name':'Тест'}))
        with patch.object(ArticleTools, '_download', side_effect=AssertionError('No duplicate download')):
            result = pipeline.invoke({'request':req})['output']
        self.assertEqual(result.status, 'scored')
        payload = json.loads(model.messages[0][1].content)
        self.assertNotIn('full_text', payload['news'])
        self.assertEqual(payload['news']['text'], req.news.full_text)
        self.assertEqual(payload['news']['text_status'], 'full')
        self.assertTrue(any('RSS-парсера' in w for w in result.warnings))


if __name__ == '__main__':
    unittest.main()
