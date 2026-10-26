import copy
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO

from relevance_agent.build_input import build_relevance_input, describe_limitations, main, save_input
from relevance_agent.contracts import RelevanceInput
from relevance_agent.loaders import InputDataError, load_bank_materials, load_company_profile, load_news

BASE = Path(__file__).resolve().parents[1]


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=BASE)
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.profile = {'company_id': 'TEST', 'scale': {'revenue_rub': 1}, 'industry': {'sector': 'Транспорт'},
                        'sensitivities': [], 'raw': {'extra': None}}
        self.profiles = self.path / 'profiles.jsonl'
        self.profiles.write_text('\n' + json.dumps(self.profile) + '\n', encoding='utf-8')
        example = json.loads((BASE / 'examples/input.json').read_text())
        self.items = example['news']
        self.news_path = self.write('news.json', {'generated_at':example['as_of'], 'count':len(self.items),'items':self.items})

    def write(self, name, data):
        path = self.path / name
        path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
        return path

    def build(self, **kwargs):
        return build_relevance_input(company_id='TEST', profiles_path=self.profiles, news_path=self.news_path,
            as_of='2026-10-26T12:00:00+03:00', request_id='test', **kwargs)

    def test_build_preserves_source_and_missing_fields(self):
        original = self.profiles.read_bytes()
        request = self.build()
        self.assertEqual(request.company_profile.model_dump(), self.profile)
        self.assertEqual(request.news[0].published, '')
        self.assertEqual(request.bank_materials, [])
        self.assertEqual(self.profiles.read_bytes(), original)
        self.assertTrue(describe_limitations(request))

    def test_list_and_parser_wrapper_match(self):
        raw = self.write('list.json', self.items)
        self.assertEqual(load_news(raw), load_news(self.news_path))

    def test_unknown_company_and_brief(self):
        with self.assertRaisesRegex(InputDataError, 'не найдена'):
            load_company_profile('missing', self.profiles)
        self.profiles.write_text(json.dumps({'company_id':'TEST','facts':{}}))
        with self.assertRaisesRegex(InputDataError, 'полный профиль'):
            self.build()

    def test_duplicate_profiles_even_after_match(self):
        with self.profiles.open('a') as stream:
            stream.write(json.dumps(self.profile) + '\n')
        with self.assertRaisesRegex(InputDataError, 'повтор company_id'):
            self.build()

    def test_bad_json_line_and_nonfinite_number(self):
        self.profiles.write_text(json.dumps(self.profile) + '\n{broken}')
        with self.assertRaisesRegex(InputDataError, 'строка 2'):
            self.build()
        self.profiles.write_text('{"company_id":"TEST","scale":{"n":NaN},"industry":{}}')
        with self.assertRaisesRegex(InputDataError, 'JSON-число'):
            self.build()

    def test_invalid_news_not_silently_dropped(self):
        bad = copy.deepcopy(self.items)
        bad.append(dict(bad[0], id='other', published='yesterday'))
        self.news_path = self.write('bad.json', bad)
        with self.assertRaisesRegex(InputDataError, 'новость #2'):
            self.build()

    def test_duplicate_news_and_wrong_count(self):
        self.news_path = self.write('dup.json', self.items * 2)
        with self.assertRaisesRegex(InputDataError, 'Повторяющиеся'):
            self.build()
        self.news_path = self.write('count.json', {'generated_at':'2026-10-26T12:00:00Z','count':0,'items':self.items})
        with self.assertRaisesRegex(InputDataError, 'count'):
            self.build()

    def test_bank_materials_preserved_and_checked(self):
        materials = [{'material_id':'synthetic','title':'Учебный материал','source_file':'synthetic.pdf','document_date':None,
                      'fragments':[{'fragment_id':'synthetic_p1','page':1,'text':'Учебный текст; не аналитика банка.'}]}]
        path = self.write('bank.json', materials)
        request = self.build(bank_materials_path=path)
        self.assertEqual(request.bank_materials[0].model_dump(), materials[0])
        materials[0]['fragments'][0]['text'] = '   '
        self.write('bank.json', materials)
        with self.assertRaisesRegex(InputDataError, 'пустой текст'):
            load_bank_materials(path)

    def test_missing_file_does_not_mean_no_bank_materials(self):
        with self.assertRaises(InputDataError):
            load_bank_materials(self.path / 'missing.json')

    def test_empty_batch_and_future_date_diagnostics(self):
        self.news_path = self.write('empty.json', [])
        self.assertEqual(self.build().news, [])
        self.items[0]['published'] = '2026-10-27T12:00:00Z'
        self.news_path = self.write('future.json', self.items)
        self.assertTrue(any('после as_of' in text for text in describe_limitations(self.build())))

    def test_save_roundtrip_and_no_overwrite(self):
        request = self.build()
        path = save_input(request, self.path / 'output.json')
        self.assertEqual(request, RelevanceInput.model_validate_json(path.read_text()))
        with self.assertRaises(InputDataError): save_input(request, path)
        with self.assertRaisesRegex(InputDataError, 'внутри'):
            save_input(request, BASE.parent / 'forbidden.json')

    def test_cli_success_and_failure(self):
        args = ['--company-id','TEST','--profiles',str(self.profiles),'--news',str(self.news_path),
                '--output',str(self.path / 'cli.json')]
        with redirect_stderr(StringIO()) as err:
            self.assertEqual(main(args), 0)
            self.assertEqual(main(args), 2)
        self.assertIn('уже существует', err.getvalue())


if __name__ == '__main__':
    unittest.main()
