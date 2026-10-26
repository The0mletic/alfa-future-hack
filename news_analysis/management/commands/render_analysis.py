"""Страницы новостей с результатами анализа — статичные HTML для просмотра в браузере.

    python manage.py render_analysis            # все прогоны из applications/analysis_runs
    python manage.py render_analysis --run 5c3028edf7b2cb07__U_44564

Результат: applications/analysis_runs/html/<news_id>__<company_id>.html и index.html со списком.
"""
import json
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from news_analysis.render import LEVELS, render_index, render_run

RUNS_DIR = Path(settings.BASE_DIR) / "applications" / "analysis_runs"
OUT_DIR = RUNS_DIR / "html"


class Command(BaseCommand):
    help = "Собирает статичные HTML-страницы новостей с результатами анализа"

    def add_arguments(self, parser):
        parser.add_argument("--run", action="append", default=[], help="имя прогона <news_id>__<company_id>")

    def handle(self, *args, **opts):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        selected = {f"{name}.json" for name in opts["run"]}
        entries, rendered = [], 0
        # Список строится по всем прогонам; страницы пересобираются только для выбранных (или для всех).
        for path in sorted(RUNS_DIR.glob("*.json")):
            run = json.loads(path.read_text(encoding="utf-8"))
            if not run.get("frontend"):
                self.stdout.write(self.style.WARNING(f"{path.name}: нет результата ({run.get('error')})"))
                continue
            company_id = path.stem.split("__", 1)[1]
            out = OUT_DIR / f"{path.stem}.html"
            if not selected or path.name in selected:
                out.write_text(render_run(run, company_id), encoding="utf-8")
                rendered += 1
                self.stdout.write(f"{out}")
            entries.append({"file": out.name, "title": run["frontend"]["title"], "company": company_id,
                            "level": run["impact"]["impact_level"], "significance": run["impact"]["significance"],
                            "status": run["status"]})
        (OUT_DIR / "index.html").write_text(render_index(entries), encoding="utf-8")
        self.stdout.write(self.style.SUCCESS(f"Готово: {rendered} страниц, в списке {len(entries)} — {OUT_DIR / 'index.html'}"))
