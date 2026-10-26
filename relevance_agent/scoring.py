"""Балл релевантности по каналам влияния (scoring_version = channels-v3).

ВОССТАНОВЛЕНО: оригинал модуля не был закоммичен автором агента релевантности. Код собран по
описанию формулы в CONTRACT.md (channels-v1…v3) и проверкам в tests/test_graph.py — все они проходят.
Если у автора есть оригинал, его версию нужно положить поверх этой.

impact = exposure × SHOCK[shock] × HORIZON[horizon] × CONFIDENCE[event_confidence]
relevance_score = 100 × (1 − Π(1 − w_k × impact_k)) по трём сильнейшим каналам, w = (1.0, 0.6, 0.4),
округление вверх от 0,5. exposure — сильнейшая sensitivity профиля для канала (со скидкой за догадку)
или базовая подверженность канала; для конкретного контрагента — 0.3 + 0.7 × доля платежей.
"""
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCORING_VERSION = 'channels-v3'
TOP_CHANNELS = 3
WEIGHTS = (1.0, 0.6, 0.4)

Channel = Literal['input_costs', 'customer_demand', 'market_structure', 'cost_of_money', 'fx_trade',
                  'regulation', 'logistics', 'counterparties']
Shock = Literal['weak', 'notable', 'strong']
Horizon = Literal['now', 'months', 'years']
Direction = Literal['positive', 'negative', 'mixed', 'unclear']
Confidence = Literal['high', 'medium', 'low']

DIRECTIONS = {'positive', 'negative', 'mixed', 'unclear'}
SHOCK = {'weak': 0.15, 'notable': 0.5, 'strong': 0.8}
HORIZON = {'now': 1.0, 'months': 0.85, 'years': 0.6}
CONFIDENCE = {'high': 1.0, 'medium': 0.8, 'low': 0.6}
STRENGTH = {'высокая': 1.0, 'средняя': 0.7, 'низкая': 0.4}

# Подверженность без признака в профиле: отсутствие данных — не нулевая подверженность.
BASELINE = {'input_costs': 0.3, 'customer_demand': 0.4, 'market_structure': 1.0, 'cost_of_money': 0.3,
            'fx_trade': 0.2, 'regulation': 0.4, 'logistics': 0.3, 'counterparties': 0.3}

CHANNEL_LABELS = {
    'input_costs': 'Издержки и цены на входе', 'customer_demand': 'Спрос покупателей',
    'market_structure': 'Структура рынка', 'cost_of_money': 'Стоимость денег',
    'fx_trade': 'Курс, ВЭД и санкции', 'regulation': 'Регулирование',
    'logistics': 'Логистика', 'counterparties': 'Контрагенты',
}

# Коды sensitivities профиля (scripts/reference_data.py) → канал.
SENSITIVITY_CHANNEL = {
    'key_rate': 'cost_of_money', 'deposit_yield': 'cost_of_money',
    'fx': 'fx_trade', 'cny': 'fx_trade', 'sanctions': 'fx_trade',
    'foreign_owner': 'regulation', 'regulation': 'regulation',
    'commodity': 'input_costs', 'energy_tariffs': 'input_costs', 'labor_market': 'input_costs',
    'consumer_demand': 'customer_demand', 'construction': 'customer_demand',
    'gov_contracts': 'customer_demand', 'industrial_demand': 'customer_demand',
    'it_budgets': 'customer_demand', 'ad_budgets': 'customer_demand', 'tourism': 'customer_demand',
    'logistics': 'logistics',
    'supplier_concentration': 'counterparties', 'customer_concentration': 'counterparties',
    'holding': 'counterparties',
}

# Признак — догадка, если ВСЕ его source_fields из этих групп (channels-v3, GUESS_DISCOUNT).
GUESS_DISCOUNT = {
    'FTS_': (0.7, 'выведено по странам ФТС: валюта и направление торговли — догадка'),
    'CL_PRODUCT_': (0.8, 'выведено по наличию продукта банка: объём долга или депозита неизвестен'),
}


class ChannelDraft(BaseModel):
    """Разбор канала моделью: балл она не ставит."""
    model_config = ConfigDict(extra='ignore')

    channel: Channel
    shock: Shock
    direction: Direction
    horizon: Horizon
    mechanism: str = Field(min_length=1)
    counterparty_id: str | None = None
    counterparty_direction: Literal['outgoing', 'incoming'] | None = None


class ChannelImpact(ChannelDraft):
    exposure: float
    exposure_basis: list[str]
    exposure_kind: Literal['source_fact', 'calculated', 'heuristic']
    impact: float
    counted: bool


def guess_discount(source_fields: list[str]) -> tuple[float, str]:
    """Скидка, только если все источники признака — догадка; иначе 1.0."""
    if not source_fields:
        return 1.0, ''
    found = []
    for field in source_fields:
        match = next((v for prefix, v in GUESS_DISCOUNT.items() if field.startswith(prefix)), None)
        if match is None:
            return 1.0, ''
        found.append(match)
    return min(found, key=lambda x: x[0])


def _pointer_exists(doc: Any, pointer: str) -> bool:
    node = doc
    for part in pointer.strip('/').split('/'):
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            return False
    return True


def _basis(doc: dict) -> str:
    for pointer in ('/industry/okved_name', '/industry/okved_code', '/industry', '/company_id'):
        if _pointer_exists(doc, pointer):
            return pointer
    return '/company_id'


def company_exposure(profile: dict, channel: str) -> tuple[float, list[str], str, str]:
    """(exposure, exposure_basis, exposure_kind, объяснение) для канала по профилю компании."""
    best = None
    for index, tag in enumerate(profile.get('sensitivities') or []):
        if SENSITIVITY_CHANNEL.get(tag.get('code')) != channel:
            continue
        factor, note = guess_discount(tag.get('source_fields') or [])
        exposure = round(STRENGTH.get(tag.get('strength'), 0.4) * factor, 4)
        if best is None or exposure > best[0]:
            reason = (tag.get('reasons') or [tag.get('label', '')])[0]
            why = f"{tag.get('label', tag.get('code'))}: {reason}" + (f' ({note})' if note else '')
            best = (exposure, [f'/sensitivities/{index}'], 'heuristic', why)
    if best:
        return best
    if channel == 'market_structure' and _pointer_exists(profile, '/industry/okved_name'):
        return 1.0, ['/industry/okved_name'], 'source_fact', 'основной рынок компании по ОКВЭД'
    return BASELINE[channel], [_basis(profile)], 'calculated', 'базовая подверженность канала: признака в профиле нет'


def counterparty_exposure(link: Any, profile: dict) -> tuple[float, list[str], str, str]:
    """Подверженность через прямого контрагента: 0.3 + 0.7 × доля платежей (не доля выручки)."""
    share = link.share or 0.0
    exposure = round(0.3 + 0.7 * share, 4)
    role = 'поставщик' if link.direction == 'outgoing' else 'покупатель'
    why = f'{role} с долей {share:.1%} в платежах через банк'
    return exposure, [_basis(profile)], 'source_fact', why


def channel_impact(exposure: float, shock: str, horizon: str, confidence: str) -> float:
    return round(exposure * SHOCK[shock] * HORIZON[horizon] * CONFIDENCE[confidence], 4)


def combine(impacts: list[float]) -> int:
    """Noisy-OR по трём сильнейшим каналам с затуханием для второго и третьего."""
    remaining = 1.0
    for weight, impact in zip(WEIGHTS, sorted(impacts, reverse=True)[:TOP_CHANNELS]):
        remaining *= 1 - weight * impact
    score = Decimal(str(round(100 * (1 - remaining), 6))).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
    return max(0, min(100, int(score)))
