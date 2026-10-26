"""Channel-based relevance: the model describes the event, this code computes the score.

relevance = how strongly the event can change the company's money through a transmission
channel, not how close the event is to the company's industry. For each channel the model
names: impact = exposure (from the profile) x shock x horizon x event confidence (from the
model). Channels are combined with noisy-OR over the three strongest impacts, the second
and third with decaying weights so that one driver seen through two channels is not
counted twice.
All numbers below are working defaults, not calibrated business weights.
"""
from typing import Literal
from pydantic import Field
from .contracts import Contract, Text, resolve_pointer

SCORING_VERSION = 'channels-v4'

Channel = Literal['input_costs', 'customer_demand', 'market_structure', 'cost_of_money',
                  'fx_trade', 'regulation', 'logistics', 'counterparties']
Shock = Literal['weak', 'moderate', 'notable', 'strong']
Direction = Literal['negative', 'positive', 'mixed', 'unclear']
Horizon = Literal['now', 'months', 'years']
Confidence = Literal['high', 'medium', 'low']
DIRECTIONS = Direction.__args__

CHANNEL_LABELS = {
    'input_costs': 'Цены на входе (сырьё, товар, энергия, труд)',
    'customer_demand': 'Спрос клиентов (отрасли-покупатели)',
    'market_structure': 'Конкуренты и структура рынка',
    'cost_of_money': 'Стоимость денег (ставка, кредит, лизинг, депозиты)',
    'fx_trade': 'Валюта, ВЭД, санкции, платежи',
    'regulation': 'Регулирование и налоги',
    'logistics': 'Логистика и география',
    'counterparties': 'Контрагенты и группа',
}

# weak is a noise floor: one weak indirect chain should stay below 20 («связь не установлена»).
# moderate sits between weak and notable so that a borderline call moves the score less.
SHOCK = {'weak': 0.15, 'moderate': 0.3, 'notable': 0.5, 'strong': 0.8}
# A shock above weak needs a verbatim quote from the news; otherwise it is capped at weak.
UNSUPPORTED_SHOCK = 'weak'
MAX_BASIS_CHARS = 200
HORIZON = {'now': 1.0, 'months': 0.85, 'years': 0.6}
STRENGTH = {'высокая': 1.0, 'средняя': 0.7, 'низкая': 0.4}
# high: completed fact or official decision; medium: statement, lawsuit, plan; low: forecast, rumour.
CONFIDENCE = {'high': 1.0, 'medium': 0.8, 'low': 0.6}
# Weights of the 1st, 2nd and 3rd strongest channel in noisy-OR.
CHANNEL_WEIGHTS = (1.0, 0.6, 0.4)
TOP_CHANNELS = len(CHANNEL_WEIGHTS)
# Exposure guessed from a weak source is discounted. Applied only when every source field of
# the sensitivity is such a guess; an industry (OKVED) or payment basis keeps full weight.
# FTS: countries are known, currency and trade direction are not.
# CL_PRODUCT_*: the bank product exists, its size (debt, deposit, guarantees) is unknown.
GUESS_DISCOUNT = {'FTS_': 0.7, 'CL_PRODUCT_': 0.8}
GUESS_NOTES = {'FTS_': 'выведено по странам ФТС, валюта и направление не известны',
               'CL_PRODUCT_': 'выведено по наличию продукта банка, объём не известен'}


def guess_discount(fields):
    """Returns (multiplier, note) for a sensitivity's source_fields."""
    fields = list(fields or [])
    prefixes = {next((p for p in GUESS_DISCOUNT if f.startswith(p)), None) for f in fields}
    if not fields or None in prefixes:
        return 1.0, ''
    prefix = min(prefixes, key=GUESS_DISCOUNT.get)
    return GUESS_DISCOUNT[prefix], GUESS_NOTES[prefix]

# Profile sensitivity codes (scripts/reference_data.SENSITIVITY_TAGS) per channel.
SENSITIVITY_CHANNEL = {
    'commodity': 'input_costs', 'energy_tariffs': 'input_costs', 'labor_market': 'input_costs',
    'consumer_demand': 'customer_demand', 'construction': 'customer_demand',
    'industrial_demand': 'customer_demand', 'gov_contracts': 'customer_demand',
    'it_budgets': 'customer_demand', 'ad_budgets': 'customer_demand', 'tourism': 'customer_demand',
    'key_rate': 'cost_of_money', 'deposit_yield': 'cost_of_money',
    'fx': 'fx_trade', 'cny': 'fx_trade', 'sanctions': 'fx_trade', 'foreign_owner': 'fx_trade',
    'regulation': 'regulation', 'logistics': 'logistics',
    'supplier_concentration': 'counterparties', 'customer_concentration': 'counterparties',
    'holding': 'counterparties',
}

# Exposure when the profile has no sensitivity for the channel. Missing data is not
# proof of zero exposure (e.g. no FTS record does not prove no foreign trade).
BASELINE = {'input_costs': 0.3, 'customer_demand': 0.4, 'market_structure': 1.0,
            'cost_of_money': 0.3, 'fx_trade': 0.2, 'regulation': 0.4, 'logistics': 0.3,
            'counterparties': 0.3}

# Profile fields that justify a baseline exposure; the first existing one is cited.
FALLBACK_FIELDS = {
    'input_costs': ['/industry/commodity_exposure', '/industry/okved_name', '/industry'],
    'customer_demand': ['/industry/demand_drivers', '/industry/okved_name', '/industry'],
    'market_structure': ['/industry/okved_name', '/industry'],
    'cost_of_money': ['/bank_relationship/debt_instruments', '/bank_relationship'],
    'fx_trade': ['/foreign_trade/is_international', '/foreign_trade'],
    'regulation': ['/industry/regulation', '/industry/okved_name', '/industry'],
    'logistics': ['/geography/region', '/geography'],
    'counterparties': ['/group/group_size_label', '/group'],
}


class ChannelDraft(Contract):
    """One channel as named by the model; exposure is never taken from the model."""
    channel: Channel
    shock: Shock
    direction: Direction
    horizon: Horizon
    mechanism: Text
    shock_basis: str | None = None  # verbatim quote from the news that justifies the shock
    counterparty_id: str | None = None
    counterparty_direction: Literal['outgoing', 'incoming'] | None = None


class EventAnalysis(Contract):
    reason: Text
    channels: list[ChannelDraft]
    event_confidence: Confidence
    limitations: list[Text]


class ChannelImpact(Contract):
    channel: Channel
    shock: Shock
    direction: Direction
    horizon: Horizon
    mechanism: Text
    exposure: float = Field(ge=0, le=1)
    exposure_basis: list[Text]  # JSON Pointers into the company or counterparty profile
    exposure_kind: Literal['source_fact', 'calculated', 'heuristic']
    impact: float = Field(ge=0, le=1)
    counted: bool  # False when outside the TOP_CHANNELS strongest impacts
    shock_basis: str | None = None
    basis_verified: bool = False  # shock_basis found in the news text
    model_shock: Shock | None = None  # the model's level when it was capped for lack of basis
    counterparty_id: str | None = None
    counterparty_direction: Literal['outgoing', 'incoming'] | None = None


def _normalize(text: str) -> str:
    import re
    text = text.lower().replace('ё', 'е')
    text = re.sub(r'[«»"“”„\'`]', '', text)
    text = re.sub(r'[\u2010-\u2015]', '-', text)
    return re.sub(r'\s+', ' ', text).strip(' .…')


def basis_in_text(basis: str | None, text: str) -> bool:
    """True when the quote occurs in the news: exactly after normalisation, or when at least
    80% of its words of 4+ letters occur (the model may drop a word or reorder punctuation)."""
    if not basis or not basis.strip():
        return False
    quote, source = _normalize(basis[:MAX_BASIS_CHARS]), _normalize(text)
    if len(quote) < 8:
        return False
    if quote in source:
        return True
    import re
    words = [w for w in re.findall(r'\w+', quote) if len(w) >= 4]
    if len(words) < 3:
        return False
    present = set(re.findall(r'\w+', source))
    return sum(w in present for w in words) / len(words) >= 0.8


def _first_field(document, paths):
    for path in paths:
        try:
            resolve_pointer(document, path)
            return path
        except ValueError:
            continue
    return '/company_id'


def company_exposure(profile: dict, channel: str):
    """Returns (exposure, basis pointers, kind, explanation) for the client itself."""
    best = None
    for index, item in enumerate(profile.get('sensitivities') or []):
        if not isinstance(item, dict) or SENSITIVITY_CHANNEL.get(item.get('code')) != channel:
            continue
        value = STRENGTH.get(item.get('strength'), STRENGTH['низкая'])
        discount, note = guess_discount(item.get('source_fields'))
        value = round(value * discount, 3)
        note = f'; {note}' if note else ''
        if best is None or value > best[0]:
            best = (value, [f'/sensitivities/{index}'], 'heuristic',
                    f"{item.get('label', item.get('code'))} ({item.get('strength', 'н/д')}){note}")
    if best:
        return best
    path = _first_field(profile, FALLBACK_FIELDS[channel])
    if channel == 'market_structure' and path.startswith('/industry'):
        return (BASELINE[channel], [path], 'source_fact', 'основной рынок компании по ОКВЭД')
    return (BASELINE[channel], [path], 'calculated',
            'подверженность по умолчанию: в профиле нет признака для этого канала')


def counterparty_exposure(link, cp_profile: dict):
    """Exposure through a direct counterparty: grows with its payment share."""
    share = link.share or 0.0
    path = _first_field(cp_profile, ['/industry/okved_name', '/industry', '/company_id'])
    return (round(0.3 + 0.7 * share, 3), [path], 'calculated',
            f'доля платежей {share:.1%} ({link.direction}); не равна доле выручки')


def channel_impact(exposure: float, shock: str, horizon: str, confidence: str) -> float:
    return round(exposure * SHOCK[shock] * HORIZON[horizon] * CONFIDENCE[confidence], 3)


def combine(impacts: list[float]) -> int:
    remaining = 1.0
    for weight, value in zip(CHANNEL_WEIGHTS, sorted(impacts, reverse=True)):
        remaining *= 1 - weight * value
    # Half-up, not banker's rounding; round(…, 6) absorbs float noise like 32.4999999.
    return int(round(100 * (1 - remaining), 6) + 0.5)
