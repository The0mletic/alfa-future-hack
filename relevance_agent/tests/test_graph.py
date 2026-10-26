import json
import unittest
from types import SimpleNamespace
from langchain_core.messages import AIMessage
from relevance_agent.contracts import CompanyProfile, NewsItem, CounterpartiesResult, ArticleResult
from relevance_agent.single_contracts import SingleNewsInput
from relevance_agent.graph import build_graph
from relevance_agent.knowledge import BankKnowledge


class FakeModel:
    model_name = 'test-only'

    def __init__(self, replies):
        self.replies = iter(replies)
        self.messages = []

    def invoke(self, messages):
        self.messages.append(messages)
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return AIMessage(content=reply, usage_metadata={'input_tokens': 10, 'output_tokens': 10, 'total_tokens': 20})


class FakeArticles:
    def __init__(self, news, options):
        self.news = news[0]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def fetch_article(self, news_id):
        return ArticleResult(news_id=news_id, status='unavailable', requested_url=self.news.link, error='test offline')


def answer(**changes):
    value = dict(relevance_score=55, reason='Связь через отрасль клиента',
                 company_evidence=[dict(field_path='/industry/name', kind='source_fact', explanation='Отрасль')],
                 counterparty_evidence=[], bank_evidence=[], limitations=[])
    value.update(changes)
    return json.dumps(value, ensure_ascii=False)


def request(company='C1', news_id='N1'):
    return SingleNewsInput(company_id=company, as_of='2026-10-26T00:00:00+00:00', news=NewsItem(
        id=news_id, title='Ключевая ставка и кредитование', summary='Стоимость кредита',
        link='https://example.org/news', published='', source='test'))


def graph(model, **kwargs):
    cp = SimpleNamespace(collect_context=lambda cid: (CounterpartiesResult(
        company_id=cid, links=[], complete=True, limitations=[]), []))
    return build_graph(model, counterparties=cp, knowledge=BankKnowledge(),
        profile_loader=lambda cid: CompanyProfile(company_id=cid, industry={'name': 'Строительство'}),
        article_factory=FakeArticles, **kwargs)


class GraphTests(unittest.TestCase):
    def test_single_output_and_real_bank(self):
        model = FakeModel([answer()])
        out = graph(model).invoke({'request': request()})['output']
        self.assertEqual(out.status, 'scored')
        self.assertEqual(out.assessment.relevance_score, 55)
        self.assertEqual(out.model_calls, 1)
        self.assertTrue(out.bank_materials)
        self.assertEqual(out.article.status, 'unavailable')
        self.assertNotIn('rank', out.model_dump())
        self.assertNotIn('rank', out.assessment.model_dump())

    def test_invalid_json_repaired_once(self):
        out = graph(FakeModel(['invalid', answer()])).invoke({'request': request()})['output']
        self.assertEqual(out.model_calls, 2)
        self.assertEqual(out.status, 'scored')

    def test_unknown_bank_reference_rejected(self):
        bad = answer(bank_evidence=[{'material_id': 'fake', 'fragment_id': 'fake'}])
        out = graph(FakeModel([bad, bad])).invoke({'request': request()})['output']
        self.assertEqual(out.status, 'not_scored')
        self.assertIsNone(out.assessment)
        self.assertEqual(out.model_calls, 2)

    def test_unknown_company_field_rejected(self):
        bad = answer(company_evidence=[dict(field_path='/missing', kind='source_fact', explanation='wrong')])
        out = graph(FakeModel([bad, bad])).invoke({'request': request()})['output']
        self.assertEqual(out.status, 'not_scored')

    def test_out_of_range_rejected(self):
        out = graph(FakeModel([answer(relevance_score=120)] * 2)).invoke({'request': request()})['output']
        self.assertEqual(out.status, 'not_scored')

    def test_provider_failure_no_fake_zero_or_secret(self):
        out = graph(FakeModel([TimeoutError('SECRET_KEY')])).invoke({'request': request()})['output']
        self.assertEqual(out.status, 'not_scored')
        self.assertEqual(out.model_calls, 1)
        self.assertNotIn('SECRET_KEY', out.model_dump_json())

    def test_retries_can_be_disabled(self):
        req = request()
        req.options.max_refinement_rounds = 0
        out = graph(FakeModel(['bad'])).invoke({'request': req})['output']
        self.assertEqual(out.model_calls, 1)

    def test_no_context_carryover_between_news_or_clients(self):
        model = FakeModel([answer(), answer()])
        compiled = graph(model)
        for company, news_id in [('C1', 'N1'), ('C2', 'N2')]:
            out = compiled.invoke({'request': request(company, news_id)})['output']
            self.assertEqual(out.company_id, company)
            self.assertEqual(out.news.id, news_id)
            self.assertEqual(out.model_calls, 1)
        data = json.loads(model.messages[1][1].content)
        self.assertEqual(data['company_profile']['company_id'], 'C2')
        self.assertNotIn('N1', model.messages[1][1].content)

    def test_large_context_not_silently_dropped(self):
        model = FakeModel([])
        out = graph(model, max_context_chars=1).invoke({'request': request()})['output']
        self.assertEqual(out.status, 'not_scored')
        self.assertEqual(out.model_calls, 0)

    def test_no_future_or_historical_analytics(self):
        kb = BankKnowledge()
        materials, _ = kb.search('ключевая ставка сталь золото', '2026-08-01T00:00:00Z')
        self.assertTrue(materials)
        self.assertTrue(all(m.document_date <= '2026-08-01' for m in materials))
        self.assertTrue(all('2021' not in m.material_id for m in materials))
        self.assertEqual(kb.search('ставка', '2020-01-01T00:00:00Z'), ([], []))


if __name__ == '__main__':
    unittest.main()
