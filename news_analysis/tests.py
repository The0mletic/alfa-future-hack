"""Тесты без обращения к API моделей: схемы, разбор ответов, проверка ссылок, извлечение статьи."""
from django.test import SimpleTestCase

from .article import _ParagraphParser
from .llm import extract_json, strict_schema
from pydantic import ValidationError

from .schemas import Action, Evidence, ImpactAnalysis, ImpactItem, NewsAnalysis, NewsEntities, NewsFact
from .validation import MISSING, check, resolve_pointer

PROFILE = {"company_id": "U_1", "foreign_trade": {"countries": ["Китай"]},
           "bank_relationship": {"products": {"loans": {"has": True}}}}


def news(**overrides) -> NewsAnalysis:
    data = dict(language="ru", is_business_news=True, category="macro", event_type="решение ЦБ",
                headline_ru="ЦБ сохранил ставку", summary="…", event_date=None,
                facts=[NewsFact(id="F1", text="Ставка 14%", status="confirmed", attributed_to=None)],
                entities=NewsEntities(companies=[], organizations=[], countries=[], regions=[], currencies=[],
                                      commodities=[]),
                affected_sectors=[], impact_channels=[], alfa_bank_comments=[], uncertainties=[],
                bank_search_queries=[])
    return NewsAnalysis(**(data | overrides))


def item(evidence: list[Evidence], level="medium", source_type="news") -> ImpactItem:
    return ImpactItem(title="t", text="t", level=level, horizon="1–3 мес.", chain=["a", "b"],
                      source_type=source_type, evidence=evidence)


def analysis(**overrides) -> ImpactAnalysis:
    data = dict(link_type="direct", effect_size="moderate", impact_level="medium", significance=50, why_it_matters="…", company_links=[],
                risks=[item([Evidence(kind="news_fact", ref="F1", detail="")])], opportunities=[],
                actions=[Action(title="Проверить", text="…", kind="check", horizon="до 1 мес.",
                                evidence=[Evidence(kind="news_fact", ref="F1", detail="")])],
                questions_to_check=[], bank_view=None, urgent=False, urgent_reason=None,
                confidence="medium", limitations=[])
    return ImpactAnalysis(**(data | overrides))


class SchemaTests(SimpleTestCase):
    def test_strict_schema_requires_all_fields_and_keeps_title_field(self):
        schema = strict_schema(ImpactAnalysis)
        impact_item = schema["$defs"]["ImpactItem"]
        self.assertIn("title", impact_item["properties"])
        self.assertEqual(set(impact_item["required"]), set(impact_item["properties"]))
        self.assertFalse(impact_item["additionalProperties"])

    def test_extract_json_from_markdown_and_chatter(self):
        self.assertEqual(extract_json('Вот ответ:\n```json\n{"a": 1}\n```\nГотово'), '{"a": 1}')
        self.assertEqual(extract_json('Итог: {"a": {"b": 2}} спасибо'), '{"a": {"b": 2}}')


class ValidationTests(SimpleTestCase):
    def test_pointer(self):
        self.assertEqual(resolve_pointer(PROFILE, "/foreign_trade/countries/0"), "Китай")
        self.assertIs(resolve_pointer(PROFILE, "/foreign_trade/currency"), MISSING)

    def test_valid_analysis_has_no_issues(self):
        self.assertEqual(check(analysis(), news(), PROFILE, {}, set()), [])

    def test_actions_are_required_by_schema(self):
        with self.assertRaises(ValidationError):
            analysis(actions=[])

    def test_unknown_company_ids_are_critical(self):
        bad = analysis(why_it_matters="Ключевой поставщик U_999 под ударом")
        problems = check(bad, news(), PROFILE, {"T1": {"result": {"top": [{"company_id": "U_5"}]}}}, set())
        self.assertEqual([p["severity"] for p in problems], ["critical"])
        ok = analysis(why_it_matters="Ключевой поставщик U_5 под ударом")
        self.assertEqual(check(ok, news(), PROFILE, {"T1": {"result": {"top": [{"company_id": "U_5"}]}}}, set()), [])

    def test_unknown_references_are_reported(self):
        bad = analysis(risks=[item([Evidence(kind="news_fact", ref="F9", detail=""),
                                    Evidence(kind="company_field", ref="/nope", detail=""),
                                    Evidence(kind="tool_result", ref="T1", detail=""),
                                    Evidence(kind="bank_fragment", ref="x", detail="")])])
        locations = [p["location"] for p in check(bad, news(), PROFILE, {}, set())]
        for i in range(4):
            self.assertIn(f"risks[0].evidence[{i}]", locations)

    def test_bank_analytics_source_needs_fragment(self):
        bad = analysis(risks=[item([Evidence(kind="news_fact", ref="F1", detail="")], source_type="bank_analytics")])
        problems = check(bad, news(), PROFILE, {}, set())
        self.assertTrue(any("bank_analytics" in p["problem"] for p in problems))

    def test_weak_link_must_not_be_inflated(self):
        ev = [Evidence(kind="news_fact", ref="F1", detail="")]
        bad = analysis(link_type="indirect", impact_level="low", significance=20, risks=[item(ev, "low") for _ in range(4)])
        problems = check(bad, news(), PROFILE, {}, set())
        self.assertTrue(any(p["location"] == "impact_level" for p in problems))

    def test_level_must_follow_rubric(self):
        problems = check(analysis(link_type="indirect", effect_size="minor"), news(), PROFILE, {}, set())
        self.assertTrue(any("по шкале" in p["problem"] for p in problems))

    def test_significance_must_match_level(self):
        problems = check(analysis(link_type="none", impact_level="none", significance=60, risks=[]), news(), PROFILE, {}, set())
        self.assertTrue(any(p["location"] == "significance" for p in problems))


class FrontendTests(SimpleTestCase):
    def test_short_tag(self):
        from .frontend import short_tag
        self.assertEqual(short_tag("Лизинг"), "Лизинг")
        long = "91 покупатель с профилем обработки данных и ИТ (0,8% входящих платежей)"
        self.assertLessEqual(len(short_tag(long)), 41)
        self.assertTrue(short_tag(long).endswith("…"))


    def test_clean_refs(self):
        from .frontend import clean_refs
        self.assertEqual(clean_refs("Цены упали на 5% (F4)."), "Цены упали на 5%.")
        self.assertEqual(clean_refs("оценки противоречивы (F7, F12)"), "оценки противоречивы")
        self.assertEqual(clean_refs("(строительство — 51%, T2)"), "(строительство — 51%)")
        self.assertEqual(clean_refs("ОКВЭД (46.71.2)"), "ОКВЭД (46.71.2)")


class ArticleTests(SimpleTestCase):
    def test_paragraphs_from_p_and_div_blocks(self):
        parser = _ParagraphParser()
        parser.feed('<nav><p>Меню сайта и ссылки на разделы, которые не нужны</p></nav>'
                    '<div class="doc-text-block__paragraph">Первый абзац статьи с достаточно длинным текстом.</div>'
                    '<p>Второй абзац статьи, тоже достаточно длинный для учёта.</p><script>var x;</script>')
        self.assertEqual(len(parser.paragraphs), 2)
        self.assertTrue(parser.paragraphs[0].startswith("Первый"))
