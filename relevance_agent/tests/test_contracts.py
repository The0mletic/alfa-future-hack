import copy
import json
from pathlib import Path
import unittest
from pydantic import ValidationError
from relevance_agent.contracts import (
    RelevanceInput, RelevanceOutput, ArticleResult, ProfileResult,
    CounterpartiesResult, validate_exchange,
)

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.input = json.loads((EXAMPLES / 'input.json').read_text())
        self.output = json.loads((EXAMPLES / 'output.json').read_text())

    def validate_pair(self):
        validate_exchange(RelevanceInput.model_validate(self.input),
                          RelevanceOutput.model_validate(self.output))

    def test_example_roundtrip(self):
        self.validate_pair()
        request = RelevanceInput.model_validate(self.input)
        self.assertEqual(request, RelevanceInput.model_validate_json(request.model_dump_json()))

    def test_existing_full_profile_is_preserved(self):
        source = Path(__file__).resolve().parents[2] / 'applications/filtered_data/company_profiles_target.jsonl'
        if not source.exists():
            self.skipTest('Исходный датасет не приложен')
        with source.open() as stream:
            profile = json.loads(next(stream))
        self.input['company_profile'] = profile
        request = RelevanceInput.model_validate(self.input)
        self.assertEqual(request.company_profile.model_dump(), profile)

    def test_invalid_scores(self):
        for score in [-1, 101, True, '65', 65.5]:
            with self.subTest(score=score):
                self.output['results'][0]['relevance_score'] = score
                with self.assertRaises(ValidationError):
                    RelevanceOutput.model_validate(self.output)

    def test_unknown_news_and_missing_result(self):
        for results in [[], [dict(self.output['results'][0], news_id='other', article=dict(self.output['results'][0]['article'], news_id='other'))]]:
            self.output['results'] = results
            with self.assertRaises(ValueError):
                self.validate_pair()

    def test_duplicate_input_id(self):
        self.input['news'] *= 2
        with self.assertRaises(ValidationError):
            RelevanceInput.model_validate(self.input)

    def test_dates_require_timezone(self):
        self.input['as_of'] = '2026-10-26T12:00:00'
        with self.assertRaises(ValidationError):
            RelevanceInput.model_validate(self.input)

    def test_not_scored_has_no_fake_zero(self):
        row = self.output['results'][0]
        row.update(status='not_scored', relevance_score=None, rank=None, selected=False)
        self.output['selected_news_ids'] = []
        self.validate_pair()
        row['relevance_score'] = 0
        with self.assertRaises(ValidationError):
            RelevanceOutput.model_validate(self.output)

    def test_unknown_bank_and_company_references(self):
        row = self.output['results'][0]
        row['bank_evidence'] = [{'material_id':'invented','fragment_id':'invented'}]
        with self.assertRaises(ValueError): self.validate_pair()
        row['bank_evidence'] = []
        row['company_evidence'][0]['field_path'] = '/missing'
        with self.assertRaises(ValueError): self.validate_pair()

    def test_rank_selection_and_top_k(self):
        row = copy.deepcopy(self.output['results'][0])
        row.update(news_id='demo_news_02', rank=2)
        row['article']['news_id'] = row['news_id']
        self.output['results'].append(row)
        self.output['selected_news_ids'].append(row['news_id'])
        self.input['news'].append(dict(self.input['news'][0], id=row['news_id']))
        self.validate_pair()
        self.input['options']['top_k'] = 1
        with self.assertRaises(ValueError): self.validate_pair()
        row['relevance_score'] = 100
        with self.assertRaises(ValidationError): RelevanceOutput.model_validate(self.output)

    def test_article_failure_and_empty_success(self):
        row = dict(self.output['results'][0]['article'], status='unavailable', error='Timeout')
        ArticleResult.model_validate(row)
        row.update(status='ok', error=None, final_url=row['requested_url'], text=' ')
        with self.assertRaises(ValidationError): ArticleResult.model_validate(row)

    def test_counterparty_requires_actual_link_and_profile(self):
        row = self.output['results'][0]
        row['counterparty_evidence'] = [{
            'company_id':'CP', 'direction':'incoming', 'share':0.7,
            'features':[{'field_path':'/industry/sector','kind':'source_fact','explanation':'Тест'}],
            'connection':'Учебная косвенная связь', 'limitations':[]}]
        request = RelevanceInput.model_validate(self.input)
        response = RelevanceOutput.model_validate(self.output)
        with self.assertRaises(ValueError): validate_exchange(request, response)
        links = CounterpartiesResult.model_validate({'company_id':'DEMO_COMPANY','complete':True,'limitations':[],
            'links':[{'company_id':'CP','direction':'incoming','amount_rub':70,'payments_count':1,'share':0.7}]})
        profile = ProfileResult.model_validate({'company_id':'CP','status':'partial',
            'profile':{'company_id':'CP','industry':{'sector':'Транспорт'}},'sources':['test.csv'],'limitations':[]})
        validate_exchange(request, response, links, [profile])
        row['counterparty_evidence'][0]['share'] = 0.9
        with self.assertRaises(ValueError):
            validate_exchange(request, RelevanceOutput.model_validate(self.output), links, [profile])

    def test_empty_batch(self):
        self.input['news'] = []
        self.output['results'] = []
        self.output['selected_news_ids'] = []
        self.validate_pair()

    def test_other_company_is_rejected(self):
        self.output['company_id'] = 'OTHER'
        with self.assertRaises(ValueError): self.validate_pair()


if __name__ == '__main__':
    unittest.main()
