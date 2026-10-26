"""Анализ новости для компании.

    python manage.py analyze_news --news applications/news_samples/yandex_dc_attack.json --company U_44564
    python manage.py analyze_news --cases applications/news_samples/cases.json   # все тестовые пары

Новость — файл в формате RSS-парсера ({generated_at, count, items} или список). Результат сохраняется
в базу (AnalysisRun) и в applications/analysis_runs/<news_id>__<company_id>.json.
"""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import connection

from news_analysis.graph import run_analysis

OUT_DIR = Path(settings.BASE_DIR) / "applications" / "analysis_runs"


def load_news(path: Path, news_id: str | None) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data["items"] if isinstance(data, dict) else data
    if news_id:
        for item in items:
            if item["id"] == news_id:
                return item
        raise CommandError(f"В {path} нет новости {news_id}")
    return items[0]


class Command(BaseCommand):
    help = "Анализ влияния новости на компанию (три агента)"

    def add_arguments(self, parser):
        parser.add_argument("--news", help="файл с новостями в формате парсера")
        parser.add_argument("--news-id", help="id новости в файле (по умолчанию первая)")
        parser.add_argument("--company", action="append", default=[], help="ID компании, можно несколько")
        parser.add_argument("--cases", help="файл тестовых пар (applications/news_samples/cases.json)")
        parser.add_argument("--no-cache", action="store_true", help="не брать разбор новости из кэша")
        parser.add_argument("--parallel", type=int, default=1, help="сколько пар анализировать одновременно")

    def handle(self, *args, **opts):
        pairs = []
        if opts["cases"]:
            cases_path = Path(opts["cases"])
            for case in json.loads(cases_path.read_text(encoding="utf-8"))["cases"]:
                news = load_news(cases_path.parent / case["news_file"], case["news_id"])
                pairs.append((news, case["company_id"], case.get("expected_relevance")))
        if opts["news"]:
            news = load_news(Path(opts["news"]), opts["news_id"])
            pairs += [(news, company, None) for company in opts["company"]]
        if not pairs:
            raise CommandError("Укажите --news и --company или --cases")

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        use_cache = not opts["no_cache"]

        def analyze(pair):
            try:
                return run_analysis(pair[0], pair[1], use_cache=use_cache)
            finally:
                connection.close()  # у каждого потока своё соединение с базой

        if opts["parallel"] > 1:
            with ThreadPoolExecutor(opts["parallel"]) as pool:
                runs = list(pool.map(analyze, pairs))
        else:
            runs = [run_analysis(news, company_id, use_cache=use_cache) for news, company_id, _ in pairs]

        total_cost = 0.0
        for (news, company_id, expected), run in zip(pairs, runs):
            self.stdout.write(f"\n=== {news['title'][:70]} × {company_id}")
            total_cost += run.cost
            out = OUT_DIR / f"{news['id']}__{company_id}.json"
            out.write_text(json.dumps({
                "status": run.status, "error": run.error, "cost": run.cost, "duration_s": run.duration_s,
                "revisions": run.revisions, "news_analysis": run.news_analysis, "plan": run.plan,
                "impact": run.impact, "critic_reports": run.critic_reports, "frontend": run.frontend,
                "trace": run.trace,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            self.report(run, expected, out)
        self.summary(pairs, runs)
        self.stdout.write(self.style.SUCCESS(f"\nВсего стоимость: {total_cost:.3f}"))

    def summary(self, pairs, runs):
        self.stdout.write("\nИтог: новость × компания | ожидание | влияние (значимость) | статус | доработок | "
                          "оценки критика (точн/обосн/персон/непротив/банк) | стоимость | время")
        for (news, company_id, expected), run in zip(pairs, runs):
            impact = run.impact or {}
            scores = run.critic_reports[-1]["report"]["scores"] if run.critic_reports else {}
            self.stdout.write(
                f"{news['id'][:16]} × {company_id} | {expected or '-'} | {impact.get('impact_level', '-')} "
                f"({impact.get('significance', '-')}) | {run.status} | {run.revisions} | "
                f"{'/'.join(str(v) for v in scores.values()) or '-'} | {run.cost:.2f} | {run.duration_s} с")

    def report(self, run, expected, out):
        style = self.style.SUCCESS if run.status == "ok" else self.style.WARNING
        self.stdout.write(style(f"статус: {run.status}, доработок: {run.revisions}, "
                                f"стоимость: {run.cost:.3f}, время: {run.duration_s} с"))
        if run.error:
            self.stdout.write(self.style.ERROR(run.error))
        impact = run.impact
        if not impact:
            return
        exp = f" (ожидалось: {expected})" if expected else ""
        self.stdout.write(f"влияние: {impact['impact_level']}, значимость {impact['significance']}{exp}, "
                          f"уверенность: {impact['confidence']}, срочно: {impact['urgent']}")
        self.stdout.write(f"почему: {impact['why_it_matters']}")
        for kind, label in (("risks", "риск"), ("opportunities", "возможность")):
            for item in impact[kind]:
                self.stdout.write(f"  {label} [{item['level']}, {item['source_type']}]: {item['title']}")
        if impact["bank_view"]:
            self.stdout.write(f"  позиция банка: {impact['bank_view']['text'][:150]}")
        for r in run.critic_reports:
            rep = r["report"]
            self.stdout.write(f"  критик, круг {r['round']}: {rep['verdict']} {rep['scores']} — "
                              f"замечаний {len(r['issues'])}")
        self.stdout.write(f"  файл: {out}")
