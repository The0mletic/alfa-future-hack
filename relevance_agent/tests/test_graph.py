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


def channel(**changes):
    value = dict(channel='customer_demand', shock='notable', direction='negative', horizon='now',
                 mechanism='Ставка → спрос на стройку → отрасль клиента')
    value.update(changes)
    return value


def answer(**changes):
    value = dict(reason='Связь через отрасль клиента', channels=[channel()],
                 event_confidence='high', limitations=[])
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
        # No sensitivities in the fake profile: baseline 0.4 x notable 0.5 x now 1.0.
        self.assertEqual(out.assessment.relevance_score, 20)
        self.assertEqual(out.assessment.scoring_version, 'channels-v3')
        self.assertEqual(out.assessment.channels[0].exposure_kind, 'calculated')
        self.assertEqual(out.model_calls, 1)
        self.assertEqual(out.bank_materials, [])
        self.assertEqual(out.bank_metadata, [])
        payload = json.loads(model.messages[0][1].content)
        self.assertNotIn('bank_materials', payload)
        self.assertNotIn('bank_metadata', payload)
        self.assertNotIn('JSON Schema', model.messages[0][0].content)
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

    def test_model_cannot_set_score_or_bank_evidence(self):
        reply = answer(relevance_score=99, bank_evidence=[{'material_id': 'fake', 'fragment_id': 'fake'}])
        out = graph(FakeModel([reply])).invoke({'request': request()})['output']
        self.assertEqual(out.status, 'scored')
        self.assertEqual(out.assessment.relevance_score, 20)
        self.assertEqual(out.assessment.bank_evidence, [])
        self.assertEqual(out.model_calls, 1)

    def test_all_channels_invalid_retried_not_zeroed(self):
        bad = answer(channels=[channel(shock='huge')])
        out = graph(FakeModel([bad, bad])).invoke({'request': request()})['output']
        self.assertEqual(out.status, 'not_scored')
        self.assertEqual(out.model_calls, 2)

    def test_unknown_channel_rejected(self):
        bad = answer(channels=[channel(channel='weather')])
        out = graph(FakeModel([bad, bad])).invoke({'request': request()})['output']
        self.assertEqual(out.status, 'not_scored')

    def test_no_channels_is_scored_zero(self):
        out = graph(FakeModel([answer(channels=[])])).invoke({'request': request()})['output']
        self.assertEqual(out.status, 'scored')
        self.assertEqual(out.assessment.relevance_score, 0)
        self.assertEqual(out.assessment.company_evidence, [])

    def test_partially_invalid_channels_repaired_without_retry(self):
        reply = answer(channels=[channel(shock='huge'), channel()], limitations=['a', 'b', 'c'])
        out = graph(FakeModel([reply])).invoke({'request': request()})['output']
        self.assertEqual(out.status, 'scored')
        self.assertEqual(out.model_calls, 1)
        self.assertEqual(len(out.assessment.channels), 1)
        self.assertEqual(len(out.assessment.limitations), 2)
        self.assertTrue(any('неверными полями: 1' in w for w in out.warnings))

    def test_duplicate_channel_kept_once_and_top_three_counted(self):
        reply = answer(channels=[channel(shock='weak'), channel(shock='strong'),
                                 channel(channel='market_structure'), channel(channel='regulation'),
                                 channel(channel='logistics', horizon='years')])
        out = graph(FakeModel([reply])).invoke({'request': request()})['output']
        channels = out.assessment.channels
        self.assertEqual([c.channel for c in channels].count('customer_demand'), 1)
        self.assertEqual([c.counted for c in channels], [True, True, True, False])
        from relevance_agent.scoring import combine
        self.assertEqual(out.assessment.relevance_score, combine([c.impact for c in channels]))

    def test_unknown_direction_kept_as_unclear(self):
        out = graph(FakeModel([answer(channels=[channel(direction='neutral')])])).invoke({'request': request()})['output']
        self.assertEqual(out.assessment.channels[0].direction, 'unclear')
        self.assertEqual(out.assessment.relevance_score, 20)

    def test_event_confidence_scales_score(self):
        scores = {}
        for level in ('high', 'medium', 'low'):
            out = graph(FakeModel([answer(event_confidence=level)])).invoke({'request': request()})['output']
            scores[level] = out.assessment.relevance_score
        # baseline 0.4 x notable 0.5 x now 1.0 x {1.0, 0.8, 0.6}
        self.assertEqual(scores, {'high': 20, 'medium': 16, 'low': 12})

    def test_missing_confidence_defaults_to_medium_without_retry(self):
        out = graph(FakeModel([answer(event_confidence='sure')])).invoke({'request': request()})['output']
        self.assertEqual(out.model_calls, 1)
        self.assertEqual(out.assessment.event_confidence, 'medium')
        self.assertTrue(any('принят medium' in w for w in out.warnings))

    def test_second_and_third_channels_decay(self):
        from relevance_agent.scoring import combine
        self.assertEqual(combine([0.5]), 50)
        # 1 - (1 - 0.5)(1 - 0.6 x 0.5) = 0.65; a third equal channel adds 0.4 x 0.5.
        self.assertEqual(combine([0.5, 0.5]), 65)
        self.assertEqual(combine([0.5, 0.5, 0.5]), 72)
        self.assertEqual(combine([0.5, 0.5, 0.5, 0.5]), 72)

    def test_sensitivity_code_mapped_to_channel(self):
        out = graph(FakeModel([answer(channels=[channel(channel='deposit_yield')])])).invoke({'request': request()})['output']
        self.assertEqual(out.assessment.channels[0].channel, 'cost_of_money')
        self.assertEqual(out.model_calls, 1)

    def test_markdown_wrapped_json_accepted(self):
        out = graph(FakeModel(['```json\n' + answer() + '\n```'])).invoke({'request': request()})['output']
        self.assertEqual(out.status, 'scored')
        self.assertEqual(out.model_calls, 1)

    def test_counterparty_channel_checked_and_share_taken_from_tool(self):
        from relevance_agent.contracts import CounterpartyLink, ProfileResult
        links = CounterpartiesResult(company_id='C1', complete=True, limitations=[], links=[
            CounterpartyLink(company_id='P1', direction='outgoing', amount_rub=10, payments_count=1, share=0.5),
            CounterpartyLink(company_id='P2', direction='incoming', amount_rub=5, payments_count=1, share=0.5)])
        profiles = [ProfileResult(company_id='P1', status='partial', sources=['test'], limitations=['x'],
                                  profile=CompanyProfile(company_id='P1', industry={'okved_code': '46.71'},
                                                         raw={'SECRET_RAW': 1})),
                    ProfileResult(company_id='P2', status='unavailable', profile=None, sources=['test'],
                                  limitations=['нет записи'])]
        reply = answer(channels=[
            channel(channel='counterparties', counterparty_id='P1', counterparty_direction='outgoing'),
            channel(channel='counterparties', counterparty_id='P2', counterparty_direction='incoming'),
            channel(channel='counterparties')])
        model = FakeModel([reply])
        cp = SimpleNamespace(collect_context=lambda cid: (links, profiles))
        out = build_graph(model, counterparties=cp,
            profile_loader=lambda cid: CompanyProfile(company_id=cid, industry={'name': 'Строительство'}),
            article_factory=FakeArticles).invoke({'request': request()})['output']
        self.assertEqual(out.status, 'scored')
        self.assertEqual(out.model_calls, 1)
        self.assertEqual([e.company_id for e in out.assessment.counterparty_evidence], ['P1'])
        self.assertEqual(out.assessment.counterparty_evidence[0].share, 0.5)
        # exposure 0.3 + 0.7 x 0.5 = 0.65; x notable 0.5 = 0.325
        self.assertEqual(out.assessment.channels[0].impact, 0.325)
        self.assertEqual(out.assessment.relevance_score, 33)
        self.assertTrue(any('без подтверждённой связи или профиля: 2' in w for w in out.warnings))
        payload = json.loads(model.messages[0][1].content)
        cards = {c['company_id']: c for c in payload['counterparties']}
        self.assertNotIn('raw', cards['P1']['profile'])
        self.assertIsNone(cards['P2']['profile'])
        self.assertNotIn('counterparty_profiles', payload)

    def test_real_profile_exposure_from_sensitivities(self):
        from relevance_agent.loaders import load_company_profile
        from relevance_agent.scoring import company_exposure
        profile = load_company_profile('U_57706').model_dump()
        exposure, basis, kind, _ = company_exposure(profile, 'customer_demand')
        self.assertEqual((exposure, kind), (1.0, 'heuristic'))
        self.assertEqual(profile['sensitivities'][int(basis[0].split('/')[-1])]['code'], 'industrial_demand')
        self.assertEqual(company_exposure(profile, 'market_structure')[:3], (1.0, ['/industry/okved_name'], 'source_fact'))
        # fx comes from FTS countries: 'высокая' 1.0 x discount 0.7.
        exposure, basis, kind, why = company_exposure(profile, 'fx_trade')
        self.assertEqual((exposure, kind), (0.7, 'heuristic'))
        self.assertIn('ФТС', why)
        # key_rate comes only from the leasing flag: 1.0 x 0.8.
        exposure, _, _, why = company_exposure(profile, 'cost_of_money')
        self.assertEqual(exposure, 0.8)
        self.assertIn('продукта банка', why)
        self.assertEqual(company_exposure(profile, 'logistics')[2], 'calculated')

    def test_guess_discount_only_when_every_source_is_a_guess(self):
        from relevance_agent.scoring import guess_discount
        self.assertEqual(guess_discount(['FTS_FLAG', 'FTS_COUNTRY_NAME'])[0], 0.7)
        self.assertEqual(guess_discount(['CL_PRODUCT_LEA_MARK', 'CL_PRODUCT_LOA_MARK'])[0], 0.8)
        self.assertEqual(guess_discount(['CL_PRODUCT_LEA_MARK', 'MAINOKVED_CODE'])[0], 1.0)
        self.assertEqual(guess_discount(['FTS_COUNTRY_NAME', 'CL_PRODUCT_GAR_MARK'])[0], 0.7)
        self.assertEqual(guess_discount([])[0], 1.0)

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

    def test_provider_failure_no_fake_zero_or_secret(self):
        out = graph(FakeModel([ConnectionError('SECRET_KEY')])).invoke({'request': request()})['output']
        self.assertEqual(out.status, 'not_scored')
        self.assertEqual(out.model_calls, 1)
        self.assertNotIn('SECRET_KEY', out.model_dump_json())

    def test_timeout_retried_once(self):
        out = graph(FakeModel([TimeoutError('SECRET_KEY'), answer()])).invoke({'request': request()})['output']
        self.assertEqual(out.status, 'scored')
        self.assertEqual(out.model_calls, 2)
        self.assertTrue(any('Выполнен повторный вызов' in w for w in out.warnings))
        self.assertNotIn('SECRET_KEY', out.model_dump_json())

    def test_second_timeout_not_retried(self):
        class APITimeoutError(Exception):
            pass
        model = FakeModel([APITimeoutError('SECRET_KEY'), APITimeoutError('SECRET_KEY'), answer()])
        out = graph(model).invoke({'request': request()})['output']
        self.assertEqual(out.status, 'not_scored')
        self.assertEqual(out.model_calls, 2)
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
