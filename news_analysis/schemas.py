"""Схемы обмена между агентами. Они же — JSON Schema для строгого ответа моделей.

Все поля обязательны (строгий режим API): «нет значения» передаётся как null или пустой список.
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- Агент 1: аналитик новости -------------------------------------------------------

Category = Literal["macro", "markets", "commodities", "regulation", "technology", "industry", "other"]
Direction = Literal["negative", "positive", "mixed"]


class NewsFact(Strict):
    id: str = Field(description="F1, F2, …")
    text: str = Field(description="Одно утверждение с цифрами, датами и участниками, если они есть в тексте")
    status: Literal["confirmed", "claimed", "opinion", "forecast"] = Field(
        description="confirmed — подано как произошедшее событие или официальные данные; "
                    "claimed — заявление участника; opinion — оценка эксперта; forecast — прогноз")
    attributed_to: str | None = Field(description="Кто сказал или чьи данные; null для фактов от издания")


class NewsEntities(Strict):
    companies: list[str]
    organizations: list[str] = Field(description="Госорганы, регуляторы, ассоциации")
    countries: list[str]
    regions: list[str] = Field(description="Регионы и города России")
    currencies: list[str]
    commodities: list[str] = Field(description="Сырьё и товары")


class AffectedSector(Strict):
    sector: str
    effect: str = Field(description="Как именно затронута отрасль, одной фразой")
    direction: Direction


class ImpactChannel(Strict):
    id: str = Field(description="C1, C2, …")
    mechanism: str = Field(description="Цепочка: событие → изменение → последствие для бизнеса")
    who_is_affected: str = Field(description="Какие компании затрагивает: отрасль, модель бизнеса, признаки")
    direction: Direction
    horizon: str = Field(description="сейчас / до 1 мес. / 1–3 мес. / 3–6 мес. / 6–12 мес. / более года")


class AlfaComment(Strict):
    id: str = Field(description="Q1, Q2, …")
    speaker: str = Field(description="Имя и должность аналитика или представителя Альфа-Банка")
    position: str = Field(description="Позиция близко к тексту новости")


class NewsAnalysis(Strict):
    language: str = Field(description="Язык исходной новости: ru, en, …")
    is_business_news: bool
    category: Category
    event_type: str = Field(description="Тип события коротко: решение регулятора, атака на инфраструктуру, …")
    headline_ru: str = Field(description="Заголовок по-русски")
    summary: str = Field(description="2–3 нейтральных предложения: что произошло")
    event_date: str | None = Field(description="Дата события YYYY-MM-DD, если указана")
    facts: list[NewsFact]
    entities: NewsEntities
    affected_sectors: list[AffectedSector]
    impact_channels: list[ImpactChannel]
    alfa_bank_comments: list[AlfaComment]
    uncertainties: list[str] = Field(description="Что пока неизвестно или спорно")
    bank_search_queries: list[str] = Field(description="3–5 запросов по 2–5 слов для поиска в аналитике банка")


# --- Агент 2: аналитик влияния -------------------------------------------------------

ToolName = Literal["get_company_section", "get_counterparties", "find_counterparties_by_keywords",
                   "get_company_card", "get_group_companies", "search_bank_analytics"]


class Hypothesis(Strict):
    hypothesis: str = Field(description="Как новость может затронуть именно эту компанию")
    channel_ids: list[str] = Field(description="Каналы влияния из разбора новости: C1, …")
    company_signals: list[str] = Field(description="Признаки компании, на которых держится гипотеза")


class PlannedCall(Strict):
    tool: ToolName
    purpose: str = Field(description="Что хотим проверить этим вызовом")


class AnalysisPlan(Strict):
    link_strength: Literal["direct", "indirect", "weak", "none"] = Field(
        description="Предварительная оценка силы связи новости с компанией")
    hypotheses: list[Hypothesis]
    data_needed: list[PlannedCall] = Field(description="Какие инструменты вызвать и зачем, не более 6")


EvidenceKind = Literal["news_fact", "news_channel", "company_field", "tool_result", "bank_fragment",
                       "bank_comment_in_news"]
SourceType = Literal["news", "company_profile", "bank_analytics", "bank_comment_in_news", "model_inference"]
Level = Literal["high", "medium", "low"]


class Evidence(Strict):
    kind: EvidenceKind
    ref: str = Field(description="F2 / C1 / JSON-путь в профиле (/foreign_trade/countries) / T3 / id фрагмента / Q1")
    detail: str = Field(description="Что именно подтверждает эта ссылка")


class ImpactItem(Strict):
    title: str
    text: str = Field(description="2–3 предложения про эту компанию")
    level: Level
    horizon: str
    chain: list[str] = Field(description="2–4 звена по порядку, каждое — короткая фраза без стрелок")
    source_type: SourceType = Field(description="Основной источник вывода")
    evidence: list[Evidence] = Field(min_length=1)


class Action(Strict):
    title: str
    text: str
    kind: Literal["check", "decision", "monitor", "bank_product"]
    horizon: str
    evidence: list[Evidence] = Field(min_length=1)


class CompanyLink(Strict):
    tag: str = Field(description="Короткая метка для бейджа, 1–4 слова: «Импорт из Китая», «Лизинг», «ОКВЭД 46.51»")
    signal: str = Field(description="Признак компании подробно, с цифрами")
    ref: str = Field(description="JSON-путь в профиле или T-номер вызова инструмента")
    why_relevant: str


class BankView(Strict):
    text: str = Field(description="Позиция банка близко к источнику, с сохранением «ожидаем/прогноз»")
    source_type: Literal["bank_analytics", "bank_comment_in_news"]
    author: str
    document: str = Field(description="Название документа или издания, где прозвучал комментарий")
    date: str | None
    refs: list[str] = Field(description="id фрагментов аналитики или Q-номера комментариев")


class ImpactAnalysis(Strict):
    link_type: Literal["direct", "indirect", "none"] = Field(
        description="direct — компания или её отрасль, финансирование, ВЭД непосредственно затронуты; "
                    "indirect — через контрагентов, цены, спрос, общую макроэкономику; none — пересечений нет")
    effect_size: Literal["significant", "moderate", "minor"] = Field(
        description="Масштаб эффекта для компании: существенный / умеренный / слабый или неопределённый")
    impact_level: Literal["high", "medium", "low", "none"]
    significance: int = Field(ge=0, le=100, description="Значимость новости для компании, 0–100")
    why_it_matters: str = Field(description="1–2 предложения: почему это важно именно этой компании")
    company_links: list[CompanyLink]
    risks: list[ImpactItem]
    opportunities: list[ImpactItem]
    actions: list[Action] = Field(min_length=1, max_length=5,
                                  description="1–5 действий; при слабой связи — что отслеживать и при каком сигнале реагировать")
    questions_to_check: list[str] = Field(description="Что выяснить, чтобы уточнить оценку")
    bank_view: BankView | None
    urgent: bool
    urgent_reason: str | None = Field(description="Короткая фраза до 12 слов для бейджа «Срочно»; null, если не срочно")
    confidence: Literal["high", "medium", "low"]
    limitations: list[str]


# --- Агент 3: критик ------------------------------------------------------------------


class CriticIssue(Strict):
    severity: Literal["critical", "major", "minor"]
    location: str = Field(description="Где проблема: risks[1], bank_view, why_it_matters, …")
    problem: str
    fix: str = Field(description="Как исправить")


class CriticScores(Strict):
    accuracy: int = Field(ge=1, le=5, description="Факты и цифры совпадают с источниками")
    grounding: int = Field(ge=1, le=5, description="Выводы подтверждены ссылками")
    personalization: int = Field(ge=1, le=5, description="Выводы про эту компанию, а не общие")
    consistency: int = Field(ge=1, le=5, description="Нет противоречий с профилем и между пунктами")
    bank_separation: int = Field(ge=1, le=5, description="Позиция банка отделена и не искажена")


class CriticReport(Strict):
    verdict: Literal["pass", "revise"]
    scores: CriticScores
    issues: list[CriticIssue]
    summary: str
