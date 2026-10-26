import csv
import json
from pathlib import Path
import tempfile
import unittest

from relevance_agent.counterparties import CounterpartyTools
from relevance_agent.loaders import InputDataError

BASE = Path(__file__).resolve().parents[1]


class CounterpartyTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(dir=BASE)
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name)
        self.payments = self.path / 'payments.csv'
        self.companies = self.path / 'companies.csv'
        self.profiles = self.path / 'profiles.jsonl'
        self.profiles.write_text(json.dumps({'company_id':'B','industry':{},'scale':{},'raw':{'x':1}})+'\n')
        self.write(self.companies, ['SYNTHETIC_U','MAINOKVED_CODE','MAINOKVED_NAME','REVENUE_COMPANY'],
                   [['C','01.1','Растениеводство','100'], ['Z','49','Транспорт','']])
        self.set_payments([['A','B','30','1'],['A','B','20','2'],['A','C','50','1'],
                           ['B','A','10','1'],['A','A','999','9'],['C','Z','500','1']])

    def write(self, path, header, rows):
        with path.open('w',newline='') as stream:
            writer=csv.writer(stream,delimiter=';');writer.writerow(header);writer.writerows(rows)

    def set_payments(self, rows):
        self.write(self.payments,['PAYER','BENEFICIARY','PAY_AMT','PAY_CNT'],rows)

    def tools(self):
        return CounterpartyTools(payments_path=self.payments,companies_path=self.companies,profiles_path=self.profiles)

    def test_aggregation_directions_and_one_hop(self):
        links,profiles=self.tools().collect_context('A')
        rows={(x.company_id,x.direction):x for x in links.links}
        self.assertEqual(len(rows),3)
        self.assertEqual(rows['B','outgoing'].amount_rub,50)
        self.assertEqual(rows['B','outgoing'].payments_count,3)
        self.assertEqual(rows['B','outgoing'].share,.5)
        self.assertEqual(rows['B','incoming'].share,1)
        self.assertEqual({p.company_id:p.status for p in profiles},{'B':'full','C':'partial'})
        self.assertTrue(links.complete)

    def test_partial_preserves_okved_and_no_inferred_currency(self):
        p=self.tools().get_company_profile('C').profile.model_dump()
        self.assertEqual(p['industry']['okved_code'],'01.1')
        self.assertEqual(p['scale']['revenue_rub'],100)
        self.assertNotIn('likely_currencies',p['foreign_trade'])

    def test_unavailable_is_not_an_error(self):
        self.assertEqual(self.tools().get_company_profile('unknown').status,'unavailable')

    def test_zero_total_share_unknown(self):
        self.set_payments([['A','B','0','1']])
        self.assertIsNone(self.tools().get_counterparties('A').links[0].share)

    def test_bad_rows_make_coverage_incomplete(self):
        for amount,count in [('NaN','1'),('-1','1'),('1','1.5'),('bad','1')]:
            with self.subTest(amount=amount,count=count):
                self.set_payments([['A','B','10','1'],['A','C',amount,count]])
                result=self.tools().get_counterparties('A')
                self.assertFalse(result.complete)
                self.assertEqual(len(result.links),1)
                self.assertTrue(any('Пропущено' in x for x in result.limitations))

    def test_missing_endpoint_incomplete(self):
        self.set_payments([['','B','10','1']])
        self.assertFalse(self.tools().get_counterparties('A').complete)

    def test_empty_and_self_only(self):
        self.set_payments([['A','A','1','1']])
        result=self.tools().get_counterparties('A')
        self.assertEqual(result.links,[])
        self.assertTrue(result.complete)

    def test_caches_are_isolated_from_caller_mutations(self):
        tools=self.tools()
        result=tools.get_counterparties('A')
        result.links.clear()
        self.set_payments([])
        self.assertEqual(len(tools.get_counterparties('A').links),3)
        p=tools.get_company_profile('C')
        p.profile.industry.clear()
        self.assertEqual(tools.get_company_profile('C').profile.industry['okved_code'],'01.1')

    def test_missing_file_and_conflicting_company_not_silent(self):
        tools=CounterpartyTools(payments_path=self.path/'missing')
        with self.assertRaises(InputDataError):tools.get_counterparties('A')
        with self.companies.open('a') as stream:stream.write('C;49;Other;100\n')
        result=self.tools().get_company_profile('C')
        self.assertEqual(result.status,'unavailable')
        self.assertIn('противоречащие',result.limitations[0])
        self.assertEqual(self.tools().get_company_profile('Z').status,'partial')


if __name__=='__main__':unittest.main()
