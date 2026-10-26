"""Детерминированная проверка анализа до критика.

Код проверяет то, что можно проверить без модели: существуют ли ссылки на факты, поля профиля,
вызовы инструментов и фрагменты аналитики, и согласованы ли уровни. Найденное передаётся критику
как замечания уровня major — модель не может «договориться» с кодом.
"""
import json
import re
from typing import Any

from .schemas import ImpactAnalysis, NewsAnalysis

MISSING = object()
COMPANY_ID = re.compile(r"\bU_\d+\b")
SIGNIFICANCE_RANGES = {"high": (65, 100), "medium": (35, 79), "low": (10, 49), "none": (0, 20)}
# Шкала уровня влияния из промпта: (тип связи, масштаб эффекта) -> уровень.
LEVEL_RUBRIC = {
    ("direct", "significant"): "high", ("direct", "moderate"): "medium", ("direct", "minor"): "low",
    ("indirect", "significant"): "medium", ("indirect", "moderate"): "low", ("indirect", "minor"): "low",
}


def expected_level(link_type: str, effect_size: str) -> str:
    return "none" if link_type == "none" else LEVEL_RUBRIC[(link_type, effect_size)]


def resolve_pointer(doc: Any, pointer: str) -> Any:
    """JSON Pointer (/a/b/0) -> значение или MISSING."""
    node = doc
    for part in pointer.strip().strip("/").split("/"):
        if part == "":
            continue
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            return MISSING
    return node


def issue(location: str, problem: str, fix: str, severity: str = "major") -> dict:
    return {"severity": severity, "location": location, "problem": problem, "fix": fix, "source": "code_check"}


def check(analysis: ImpactAnalysis, news: NewsAnalysis, profile: dict, tool_calls: dict,
          bank_fragment_ids: set[str]) -> list[dict]:
    fact_ids = {f.id for f in news.facts}
    channel_ids = {c.id for c in news.impact_channels}
    comment_ids = {c.id for c in news.alfa_bank_comments}
    problems: list[dict] = []

    def check_evidence(location: str, evidence: list) -> None:
        for i, ev in enumerate(evidence):
            where = f"{location}.evidence[{i}]"
            ref = ev.ref.strip()
            if ev.kind == "news_fact" and ref not in fact_ids:
                problems.append(issue(where, f"Нет факта {ref} в разборе новости", f"Ссылайся на один из {sorted(fact_ids)}"))
            elif ev.kind == "news_channel" and ref not in channel_ids:
                problems.append(issue(where, f"Нет канала влияния {ref} в разборе новости", f"Ссылайся на один из {sorted(channel_ids)}"))
            elif ev.kind == "company_field" and resolve_pointer(profile, ref) is MISSING:
                problems.append(issue(where, f"В профиле компании нет поля {ref}",
                                      "Укажи существующий JSON-путь (например /foreign_trade/countries) или убери ссылку"))
            elif ev.kind == "tool_result" and ref not in tool_calls:
                problems.append(issue(where, f"Вызова {ref} не было", f"Доступные вызовы: {sorted(tool_calls)}"))
            elif ev.kind == "bank_fragment" and ref not in bank_fragment_ids:
                problems.append(issue(where, f"Фрагмент аналитики {ref} не был получен",
                                      "Ссылайся только на фрагменты из контекста или из search_bank_analytics"))
            elif ev.kind == "bank_comment_in_news" and ref not in comment_ids:
                problems.append(issue(where, f"Нет комментария {ref} в новости", f"Доступные: {sorted(comment_ids)}"))

    for group in ("risks", "opportunities"):
        for i, item in enumerate(getattr(analysis, group)):
            location = f"{group}[{i}]"
            if not item.evidence:
                problems.append(issue(location, "Вывод без единой ссылки на источник", "Добавь evidence или убери пункт"))
            check_evidence(location, item.evidence)
            kinds = {e.kind for e in item.evidence}
            if item.source_type == "bank_analytics" and "bank_fragment" not in kinds:
                problems.append(issue(location, "source_type=bank_analytics, но нет ссылки на фрагмент аналитики",
                                      "Добавь bank_fragment или смени source_type"))
            if item.source_type == "bank_comment_in_news" and "bank_comment_in_news" not in kinds:
                problems.append(issue(location, "source_type=bank_comment_in_news, но нет ссылки на комментарий Q",
                                      "Добавь ссылку Q или смени source_type"))
    for i, action in enumerate(analysis.actions):
        check_evidence(f"actions[{i}]", action.evidence)

    view = analysis.bank_view
    if view:
        allowed = bank_fragment_ids if view.source_type == "bank_analytics" else comment_ids
        if not view.refs:
            problems.append(issue("bank_view", "Позиция банка без ссылок на источник",
                                  "Укажи id фрагментов или Q-номер комментария, либо bank_view = null"))
        for ref in view.refs:
            if ref not in allowed:
                problems.append(issue("bank_view", f"Ссылка {ref} не найдена среди источников типа {view.source_type}",
                                      "Исправь ссылку или тип источника"))
    elif news.alfa_bank_comments:
        problems.append(issue("bank_view", "В новости есть комментарий Альфа-Банка, но bank_view пуст",
                              "Проверь, относится ли комментарий к влиянию на компанию; если да — заполни bank_view",
                              severity="minor"))

    # ID компаний в тексте анализа должны встречаться в профиле или результатах инструментов.
    known_ids = set(COMPANY_ID.findall(json.dumps(profile, ensure_ascii=False, default=str)))
    known_ids |= set(COMPANY_ID.findall(json.dumps(tool_calls, ensure_ascii=False, default=str)))
    unknown = sorted(set(COMPANY_ID.findall(analysis.model_dump_json())) - known_ids)
    if unknown:
        problems.append(issue("analysis", f"Упомянуты компании, которых нет в данных: {', '.join(unknown)}",
                              "Используй только ID из профиля и результатов инструментов или убери их", "critical"))

    items = len(analysis.risks) + len(analysis.opportunities)
    if analysis.impact_level != "none" and not analysis.actions:
        problems.append(issue("actions", "Нет ни одного действия", "Добавь 1–4 конкретных действия: что проверить, "
                                                                  "какое решение принять или что отслеживать"))
    if analysis.impact_level in ("low", "none") and items > 3:
        problems.append(issue("impact_level", f"Влияние «{analysis.impact_level}», но {items} рисков и возможностей",
                              "При слабой связи оставь 0–3 самых обоснованных пункта или пересмотри уровень влияния"))
    if analysis.impact_level in ("high", "medium") and not items:
        problems.append(issue("impact_level", f"Влияние «{analysis.impact_level}», но нет ни рисков, ни возможностей",
                              "Опиши, в чём именно влияние, или понизь уровень"))
    if analysis.urgent and not analysis.urgent_reason:
        problems.append(issue("urgent_reason", "urgent=true без причины", "Укажи, какое решение нужно принять и когда"))
    expected = expected_level(analysis.link_type, analysis.effect_size)
    if analysis.impact_level != expected:
        problems.append(issue("impact_level", f"impact_level={analysis.impact_level}, но по шкале для связи "
                                              f"«{analysis.link_type}» и эффекта «{analysis.effect_size}» должно быть "
                                              f"«{expected}»", "Согласуй уровень с типом связи и масштабом эффекта"))
    low, high = SIGNIFICANCE_RANGES[analysis.impact_level]
    if not low <= analysis.significance <= high:
        problems.append(issue("significance", f"significance={analysis.significance} не соответствует "
                                              f"impact_level={analysis.impact_level} (ожидается {low}–{high})",
                              "Согласуй оценку значимости с уровнем влияния"))
    if analysis.impact_level == "none" and any(x.level == "high" for x in analysis.risks + analysis.opportunities):
        problems.append(issue("impact_level", "Влияние «none», но есть пункты высокого уровня",
                              "Либо понизь уровни пунктов, либо пересмотри impact_level"))
    return problems
