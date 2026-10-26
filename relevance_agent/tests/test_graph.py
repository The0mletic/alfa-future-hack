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
    return build_graph(model, counterparties=cp,
        profile_loader=lambda cid: CompanyProfile(company_id=cid, industry={'name': 'Строительство'}),
        article_factory=FakeArticles, **kwargs)


class GraphTests(unittest.TestCase):
    def test_single_output_without_bank_context(self):
        model = FakeModel([answer()])
        out = graph(model).invoke({'request': request()})['output']
        self.assertEqual(out.status, 'scored')
        self.assertEqual(out.assessment.relevance_score, 55)
        self.assertEqual(out.model_calls, 1)
        self.assertEqual(out.bank_materials, [])
        self.assertEqual(out.bank_metadata, [])
        payload = json.loads(model.messages[0][1].content)
        self.assertNotIn('bank_materials', payload)
        self.assertNotIn('bank_metadata', payload)
        self.assertFalse(any('аналитик' in warning for warning in out.warnings))
        self.assertEqual(out.article.status, 'unavailable')
        self.assertNotIn('rank', out.model_dump())
        self.assertNotIn('rank', out.assessment.model_dump())

    def test_legacy_knowledge_argument_is_never_used(self):
        from unittest.mock import Mock
        knowledge = Mock()
        knowledge.search.side_effect = AssertionError('Ranking must not retrieve bank analytics')
        out = graph(FakeModel([answer()]), knowledge=knowledge).invoke({'request': request()})['output']
        self.assertEqual(out.status, 'scored')
        knowledge.search.assert_not_called()

    def test_invalid_json_repaired_once(self):
        out = graph(FakeModel(['invalid', answer()])).invoke({'request': request()})['output']
        self.assertEqual(out.model_calls, 2)
        self.assertEqual(out.status, 'scored')

    def test_unknown_bank_reference_dropped_without_retry(self):
        bad = answer(bank_evidence=[{'material_id': 'fake', 'fragment_id': 'fake'}])
        out = graph(FakeModel([bad])).invoke({'request': request()})['output']
        self.assertEqual(out.status, 'scored')
        self.assertEqual(out.assessment.bank_evidence, [])
        self.assertEqual(out.model_calls, 1)
        self.assertTrue(any('банковскую аналитику отброшены' in w for w in out.warnings))

    def test_unknown_company_field_rejected(self):
        bad = answer(company_evidence=[dict(field_path='/missing', kind='source_fact', explanation='wrong')])
        out = graph(FakeModel([bad, bad])).invoke({'request': request()})['output']
        self.assertEqual(out.status, 'not_scored')

    def test_partially_invalid_evidence_repaired_without_retry(self):
        good = dict(field_path='/industry/name', kind='source_fact', explanation='Отрасль')
        bad = dict(field_path='/missing', kind='source_fact', explanation='wrong')
        reply = answer(company_evidence=[bad, good, good, good, good], limitations=['a', 'b', 'c'])
        out = graph(FakeModel([reply])).invoke({'request': request()})['output']
        self.assertEqual(out.status, 'scored')
        self.assertEqual(out.model_calls, 1)
        self.assertEqual(len(out.assessment.company_evidence), 3)
        self.assertEqual(len(out.assessment.limitations), 2)
        self.assertTrue(any('несуществующим полем: 1' in w for w in out.warnings))

    def test_markdown_wrapped_json_accepted(self):
        out = graph(FakeModel(['```json\n' + answer() + '\n```'])).invoke({'request': request()})['output']
        self.assertEqual(out.status, 'scored')
        self.assertEqual(out.model_calls, 1)

    def test_counterparty_evidence_checked_and_share_taken_from_tool(self):
        from relevance_agent.contracts import CounterpartyLink, ProfileResult
        links = CounterpartiesResult(company_id='C1', complete=True, limitations=[], links=[
            CounterpartyLink(company_id='P1', direction='outgoing', amount_rub=10, payments_count=1, share=0.123456),
            CounterpartyLink(company_id='P2', direction='incoming', amount_rub=5, payments_count=1, share=0.5)])
        profiles = [ProfileResult(company_id='P1', status='partial', sources=['test'], limitations=['x'],
                                  profile=CompanyProfile(company_id='P1', industry={'okved_code': '46.71'},
                                                         raw={'SECRET_RAW': 1})),
                    ProfileResult(company_id='P2', status='unavailable', profile=None, sources=['test'],
                                  limitations=['нет записи'])]
        feature = [dict(field_path='/industry/okved_code', kind='source_fact', explanation='ОКВЭД')]
        reply = answer(counterparty_evidence=[
            dict(company_id='P1', direction='outgoing', share=0.12, features=feature, connection='отрасль', limitations=[]),
            dict(company_id='P2', direction='incoming', share=0.5, features=feature, connection='нет профиля', limitations=[])])
        model = FakeModel([reply])
        cp = SimpleNamespace(collect_context=lambda cid: (links, profiles))
        out = build_graph(model, counterparties=cp,
            profile_loader=lambda cid: CompanyProfile(company_id=cid, industry={'name': 'Строительство'}),
            article_factory=FakeArticles).invoke({'request': request()})['output']
        self.assertEqual(out.status, 'scored')
        self.assertEqual(out.model_calls, 1)
        self.assertEqual([e.company_id for e in out.assessment.counterparty_evidence], ['P1'])
        self.assertEqual(out.assessment.counterparty_evidence[0].share, 0.123456)
        payload = json.loads(model.messages[0][1].content)
        cards = {c['company_id']: c for c in payload['counterparties']}
        self.assertNotIn('raw', cards['P1']['profile'])
        self.assertIsNone(cards['P2']['profile'])
        self.assertNotIn('counterparty_profiles', payload)

    def test_company_card_is_pointer_compatible_subset(self):
        from relevance_agent.graph import company_card
        from relevance_agent.loaders import load_company_profile
        from relevance_agent.contracts import resolve_pointer
        profile = load_company_profile('U_57706')
        card = company_card(profile)
        for key in ('raw', 'data_quality', 'summary', 'counterparties', 'news_keywords'):
            self.assertNotIn(key, card)
        self.assertNotIn('client_status', card['bank_relationship'])
        full = profile.model_dump()

        def leaves(node, path=''):
            if isinstance(node, dict):
                for k, v in node.items():
                    yield from leaves(v, f'{path}/{k}')
            elif isinstance(node, list):
                for i, v in enumerate(node):
                    yield from leaves(v, f'{path}/{i}')
            else:
                yield path, node
        for path, value in leaves(card):
            self.assertEqual(resolve_pointer(full, path), value)
        self.assertLess(len(json.dumps(card, ensure_ascii=False)), len(json.dumps(full, ensure_ascii=False)) / 2)

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
