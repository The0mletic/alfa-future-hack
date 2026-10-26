"""Статичная HTML-страница с результатом анализа — чтобы посмотреть его глазами без запуска сервера.

Основная часть — шаблон фронтенда base/news_detail.html с данными модели. Ниже добавляется блок
«Как получен анализ»: всё, чего пока нет в шаблоне (действия, цепочки, ссылки на источники, разбор
новости, план, вызовы инструментов, отчёты критика, шаги агентов). Стили и скрипт встраиваются в файл.
"""
import json
import re
from html import escape
from pathlib import Path

from django.conf import settings
from django.template.loader import render_to_string

from base.mock_data import USER
from companies.models import Company

from .frontend import SOURCE_LABELS, clean_refs, short_tag

STATIC_DIR = Path(settings.BASE_DIR) / "base" / "static" / "base"
LEVELS = {"high": "высокое", "medium": "среднее", "low": "низкое", "none": "нет влияния"}
CONFIDENCE = {"high": "высокая", "medium": "средняя", "low": "низкая"}
VERDICTS = {"pass": "принято", "revise": "на доработку"}
LINKS = {"direct": "прямая", "indirect": "косвенная", "none": "нет связи"}
EFFECTS = {"significant": "существенный", "moderate": "умеренный", "minor": "слабый / неопределённый"}
FACT_STATUS = {"confirmed": "факт", "claimed": "заявление", "opinion": "оценка", "forecast": "прогноз"}
ACTION_KINDS = {"check": "проверить", "decision": "решение", "monitor": "отслеживать", "bank_product": "продукт банка"}
EVIDENCE_KINDS = {"news_fact": "факт новости", "news_channel": "канал влияния", "company_field": "поле профиля",
                  "tool_result": "вызов инструмента", "bank_fragment": "аналитика банка",
                  "bank_comment_in_news": "комментарий банка в новости"}
STEP_NAMES = {"fetch_article": "загрузка статьи", "news_analyst": "аналитик новости", "bank_retrieval": "поиск по аналитике банка",
              "impact_plan": "аналитик влияния: план", "impact_research": "аналитик влияния: сбор данных",
              "impact_write": "аналитик влияния: итог", "validate": "проверка кодом", "critic": "критик", "tool": "инструмент"}

EXTRA_CSS = """
.trace { margin: 32px auto 48px; display: grid; grid-template-columns: minmax(0, 1fr); gap: 16px; }
.trace td, .trace p, .trace li, .trace summary { overflow-wrap: anywhere; }
.trace .debug__body { grid-template-columns: minmax(0, 1fr); }
.trace table { table-layout: fixed; }
.trace .card { padding: 24px; }
.trace h2 { font-size: 22px; line-height: 28px; margin: 0 0 4px; }
.trace h3 { font-size: 17px; line-height: 24px; margin: 16px 0 8px; }
.trace p, .trace li, .trace td, .trace th { font-size: 14px; line-height: 20px; }
.trace .muted { color: #86868a; }
.trace table { width: 100%; border-collapse: collapse; }
.trace th, .trace td { text-align: left; vertical-align: top; padding: 8px 10px; border-bottom: 1px solid #ececed; }
.trace th { color: #86868a; font-weight: 500; }
.trace ul { margin: 4px 0 0; padding-left: 18px; }
.trace .chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }
.trace .chip { padding: 4px 10px; border-radius: 8px; background: #f3f4f5; font-size: 13px; }
.trace .chip--red { background: #feeceb; color: #c41f12; } .trace .chip--green { background: #e8f6ec; color: #0f7a35; }
.trace .chain { color: #3c3c3f; margin: 6px 0; }
.trace .item { padding: 12px 0; border-bottom: 1px solid #ececed; } .trace .item:last-child { border-bottom: 0; }
.trace pre { white-space: pre-wrap; word-break: break-word; font-size: 12px; background: #f7f7f8; padding: 8px; border-radius: 8px; margin: 6px 0 0; }
.trace details summary { cursor: pointer; font-weight: 500; }
.trace .debug { background: #fff; border-radius: 24px; padding: 16px 24px; }
.trace .debug > summary { font-size: 17px; line-height: 24px; padding: 8px 0; }
.trace .debug > summary .muted { font-size: 13px; font-weight: 400; margin-left: 8px; }
.trace .debug__body { display: grid; gap: 16px; margin-top: 12px; }
.trace .debug__body .card { box-shadow: none; border: 1px solid #ececed; }
"""


def e(value) -> str:
    return escape(str(value if value is not None else "—"))


def company_context(company_id: str) -> dict:
    c = Company.objects.get(pk=company_id)
    p = c.profile or {}
    return {
        "name": f"компании {company_id}", "initials": "U", "inn": "обезличено", "client_since": "—",
        "industry": (p.get("industry") or {}).get("sector", c.sector), "okved": c.okved_code, "region": c.region,
        "revenue": (p.get("scale") or {}).get("revenue_label", ""),
        "employees": (p.get("scale") or {}).get("employees_label", ""),
        "foreign_trade": ", ".join((p.get("foreign_trade") or {}).get("countries", [])) or "нет",
        "products": (p.get("bank_relationship") or {}).get("active_products", []),
    }


def inline_assets(html: str) -> str:
    css = (STATIC_DIR / "css" / "app.css").read_text(encoding="utf-8")
    js = (STATIC_DIR / "js" / "app.js").read_text(encoding="utf-8")
    html = re.sub(r'<link rel="stylesheet" href="[^"]*app\.css">', lambda _: f"<style>{css}\n{EXTRA_CSS}</style>", html)
    html = re.sub(r'<script src="[^"]*app\.js" defer></script>', lambda _: f"<script>{js}</script>", html)
    html = re.sub(r'<link rel="icon"[^>]*>', "", html)
    return html.replace('href="/"', 'href="index.html"')


def evidence_list(evidence: list[dict]) -> str:
    """«Основано на» для пользователя: тип источника и что он подтверждает, без технических ссылок."""
    rows = "".join(f"<li><b>{e(EVIDENCE_KINDS.get(ev['kind'], ev['kind']))}</b> — {e(ev['detail'])}</li>"
                   for ev in evidence)
    return f"<p class='muted'>Основано на:</p><ul>{rows}</ul>" if rows else ""


def impact_items(items: list[dict], kind: str) -> str:
    if not items:
        return '<p class="muted">Нет</p>'
    out = []
    for x in items:
        chain = " → ".join(s.strip(" →-—") for s in x["chain"])
        out.append(
            f'<div class="item"><b>{e(x["title"])}</b>'
            f'<div class="chips"><span class="chip chip--{"red" if kind == "risk" else "green"}">{e(LEVELS.get(x["level"], x["level"]))}</span>'
            f'<span class="chip">{e(x["horizon"])}</span><span class="chip">источник: {e(SOURCE_LABELS[x["source_type"]])}</span></div>'
            f'<p>{e(clean_refs(x["text"]))}</p><p class="chain">Цепочка: {e(clean_refs(chain))}</p>{evidence_list(x["evidence"])}</div>')
    return "".join(out)


def trace_section(run: dict) -> str:
    impact, na, plan = run["impact"], run["news_analysis"], run.get("plan") or {}
    parts = []

    parts.append(f"""<section class="card"><h2>Итог для компании</h2>
<div class="chips"><span class="chip">Влияние: {e(LEVELS[impact['impact_level']])} ({impact['significance']}/100)</span>
<span class="chip">Уверенность: {e(CONFIDENCE[impact['confidence']])}</span><span class="chip">Срочно: {'да — ' + e(impact['urgent_reason']) if impact['urgent'] else 'нет'}</span></div>
<h3>Почему важно компании</h3><p>{e(clean_refs(impact['why_it_matters']))}</p>
<h3>Признаки компании, на которых держится вывод</h3>
<table><tr><th>Признак</th><th>Почему важно</th></tr>
{''.join(f"<tr><td>{e(l['signal'])}</td><td>{e(l['why_relevant'])}</td></tr>" for l in impact['company_links'])}</table>
</section>""")

    parts.append(f"""<section class="card"><h2>Риски</h2>{impact_items(impact['risks'], 'risk')}</section>
<section class="card"><h2>Возможности</h2>{impact_items(impact['opportunities'], 'opp')}</section>""")

    actions = "".join(
        f'<div class="item"><b>{e(a["title"])}</b><div class="chips"><span class="chip">{e(ACTION_KINDS.get(a["kind"], a["kind"]))}</span>'
        f'<span class="chip">{e(a["horizon"])}</span></div><p>{e(clean_refs(a["text"]))}</p>{evidence_list(a["evidence"])}</div>'
        for a in impact["actions"])
    questions = "".join(f"<li>{e(clean_refs(q))}</li>" for q in impact["questions_to_check"])
    limits = "".join(f"<li>{e(q)}</li>" for q in impact["limitations"])
    parts.append(f"""<section class="card"><h2>Что делать</h2>{actions}
<h3>Что выяснить, чтобы уточнить оценку</h3><ul>{questions}</ul>
<h3>Ограничения анализа</h3><ul>{limits}</ul></section>""")

    # Дальше — только отладка: как агенты пришли к результату. Позиция банка уже есть на странице новости.
    debug: list[str] = []
    bank = impact.get("bank_view")
    bank_html = (f"<p>{e(bank['text'])}</p><p class='muted'>{e(bank['author'])} · «{e(bank['document'])}» · {e(bank['date'])} · "
                 f"источник: {e(SOURCE_LABELS[bank['source_type']])} · ссылки: {e(', '.join(bank['refs']))}</p>") if bank else \
        "<p class='muted'>Подходящей по теме аналитики Альфа-Банка не найдено — позиция банка не показывается.</p>"
    retrieval = next((s for s in run["trace"] if s.get("step") == "bank_retrieval"), {})
    fragments = "".join(f"<li>{e(f['document'])} — <span class='muted'>{e(f['id'])}, score {f['score']}</span></li>"
                        for f in retrieval.get("fragments", []))
    debug.append(f"""<section class="card"><h2>Позиция банка: как искали</h2>{bank_html}
<h3>Что нашёл обязательный поиск по аналитике банка</h3>
<p class="muted">Запросы: {e('; '.join(retrieval.get('queries', [])))} · документы не позже {e(retrieval.get('as_of'))}</p><ul>{fragments}</ul></section>""")

    facts = "".join(f"<tr><td>{e(f['id'])}</td><td>{e(FACT_STATUS[f['status']])}</td><td>{e(f['attributed_to'] or '')}</td><td>{e(f['text'])}</td></tr>"
                    for f in na["facts"])
    channels = "".join(f"<tr><td>{e(c['id'])}</td><td>{e(c['mechanism'])}</td><td>{e(c['who_is_affected'])}</td><td>{e(c['direction'])}, {e(c['horizon'])}</td></tr>"
                       for c in na["impact_channels"])
    comments = "".join(f"<li><b>{e(c['id'])} {e(c['speaker'])}</b>: {e(c['position'])}</li>" for c in na["alfa_bank_comments"]) or "<li class='muted'>нет</li>"
    uncertain = "".join(f"<li>{e(u)}</li>" for u in na["uncertainties"])
    debug.append(f"""<section class="card"><h2>Агент 1 — разбор новости</h2>
<p class="muted">Категория: {e(na['category'])} · тип события: {e(na['event_type'])} · язык: {e(na['language'])}</p>
<h3>Факты</h3><table><tr><th>ID</th><th>Тип</th><th>Кто сказал</th><th>Утверждение</th></tr>{facts}</table>
<h3>Каналы влияния</h3><table><tr><th>ID</th><th>Механизм</th><th>Кого касается</th><th>Направление, горизонт</th></tr>{channels}</table>
<h3>Комментарии Альфа-Банка в новости</h3><ul>{comments}</ul>
<h3>Что неизвестно</h3><ul>{uncertain}</ul></section>""")

    hyps = "".join(f"<li>{e(h['hypothesis'])} <span class='muted'>({e(', '.join(h['channel_ids']))})</span></li>"
                   for h in plan.get("hypotheses", []))
    calls = "".join(f"<li><b>{e(c['tool'])}</b> — {e(c['purpose'])}</li>" for c in plan.get("data_needed", []))
    tools = "".join(
        f"<details class='item'><summary>{e(s['call_id'])} {e(s['name'])}({e(json.dumps(s['arguments'], ensure_ascii=False))})</summary>"
        f"<pre>{e(s.get('error') or s.get('result_preview'))}</pre></details>"
        for s in run["trace"] if s["kind"] == "tool")
    debug.append(f"""<section class="card"><h2>Агент 2 — план и сбор данных</h2>
<p class="muted">Предварительная сила связи: {e(plan.get('link_strength'))}</p>
<h3>Гипотезы</h3><ul>{hyps}</ul><h3>Что решил запросить</h3><ul>{calls}</ul>
<h3>Вызовы инструментов (начало ответа)</h3>{tools or "<p class='muted'>вызовов не было</p>"}</section>""")

    rounds = []
    for r in run["critic_reports"]:
        rep = r["report"]
        issues = "".join(f"<li><b>{e(i['severity'])}</b> [{e(i.get('source', 'critic'))}] {e(i['location'])}: {e(i['problem'])} <span class='muted'>→ {e(i['fix'])}</span></li>"
                         for i in r["issues"]) or "<li class='muted'>замечаний нет</li>"
        scores = ", ".join(f"{k} {v}" for k, v in rep["scores"].items())
        rounds.append(f"<div class='item'><b>Круг {r['round']}: {e(VERDICTS[rep['verdict']])}</b> <span class='muted'>({e(scores)})</span>"
                      f"<p>{e(rep['summary'])}</p><ul>{issues}</ul></div>")
    debug.append(f"<section class='card'><h2>Агент 3 — критик</h2>{''.join(rounds)}</section>")

    rows = []
    for s in run["trace"]:
        name = STEP_NAMES.get(s.get("step", "").split(":")[0].rstrip("0123456789"), s.get("step"))
        if s["kind"] == "llm":
            rows.append(f"<tr><td>{s['n']}</td><td>{e(name)} <span class='muted'>{e(s['step'])}</span></td><td>{e(s['model'])}</td>"
                        f"<td>{e(s['latency_s'])} с</td><td>{e(s['prompt_tokens'])} / {e(s['completion_tokens'])}</td><td>{(s['cost'] or 0):.3f}</td></tr>")
        elif s["kind"] == "tool":
            rows.append(f"<tr><td>{s['n']}</td><td>инструмент {e(s['call_id'])}: {e(s['name'])}</td><td>—</td><td>{e(s['latency_s'])} с</td><td></td><td></td></tr>")
        else:
            rows.append(f"<tr><td>{s['n']}</td><td>{e(name)}</td><td>{e(s['kind'])}</td><td></td><td></td><td></td></tr>")
    debug.append(f"""<section class="card"><h2>Все шаги</h2>
<table><tr><th>#</th><th>Шаг</th><th>Модель</th><th>Время</th><th>Токены вход / выход</th><th>Стоимость</th></tr>{''.join(rows)}</table></section>""")

    summary = (f"статус: {e(run['status'])} · связь: {e(LINKS[impact['link_type']])} · эффект: {e(EFFECTS[impact['effect_size']])} · "
               f"доработок: {e(run['revisions'])} · {run['cost']:.2f} ед. API · {e(run['duration_s'])} с")
    debug_html = (f'<details class="debug"><summary>Отладка: агенты и шаги <span class="muted">{summary}</span></summary>'
                  f'<div class="debug__body">{"".join(debug)}</div></details>')
    return f'<div class="container trace"><h1 class="article__title">Подробнее об анализе</h1>{"".join(parts)}{debug_html}</div>'


def render_run(run: dict, company_id: str) -> str:
    item = dict(run["frontend"])
    item["basis"] = [short_tag(b) for b in item["basis"]]  # прогоны до появления короткой метки tag
    item["why"] = clean_refs(item["why"])
    if item.get("bank_view"):
        item["bank_view"] = {**item["bank_view"], "text": clean_refs(item["bank_view"]["text"])}
    for key in ("risks", "opportunities"):
        item[key] = [{**x, "title": clean_refs(x["title"]), "text": clean_refs(x["text"])} for x in item[key]]
    context = {"user": USER, "company": company_context(company_id), "item": item, "related": []}
    html = render_to_string("base/news_detail.html", context)
    html = html.replace("</main>", trace_section(run) + "</main>", 1)
    return inline_assets(html)


def render_index(entries: list[dict]) -> str:
    rows = "".join(
        f"<tr><td><a href='{e(x['file'])}'>{e(x['title'])}</a></td><td>{e(x['company'])}</td>"
        f"<td>{e(LEVELS.get(x['level'], x['level']))} ({e(x['significance'])})</td><td>{e(x['status'])}</td></tr>"
        for x in entries)
    css = (STATIC_DIR / "css" / "app.css").read_text(encoding="utf-8")
    return f"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Прогоны анализа</title><style>{css}{EXTRA_CSS}</style></head><body><main class="page"><div class="container trace">
<h1 class="article__title">Прогоны анализа «новость × компания»</h1><section class="card">
<table><tr><th>Новость</th><th>Компания</th><th>Влияние</th><th>Статус</th></tr>{rows}</table></section></div></main></body></html>"""
