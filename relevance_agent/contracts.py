"""Контракт v1.0. Здесь нет загрузки данных, сетевых вызовов или LLM."""
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Text = Annotated[str, Field(min_length=1)]
Score = Annotated[int, Field(strict=True, ge=0, le=100)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def check_time(value: str, allow_empty: bool = False) -> str:
    if allow_empty and value == "":
        return value
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Дата должна содержать часовой пояс")
    return value


def unique(values: list[str], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"Повторяющиеся идентификаторы: {label}")


class CompanyProfile(Contract):
    # Существующие блоки сохраняются без потери данных; их внутренние правила
    # будут проверяться адаптером, а не копироваться в этот контракт.
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)
    company_id: Text


class NewsItem(Contract):
    id: Text
    title: Text
    summary: str
    link: str
    published: str
    source: Text
    author: str = ""
    categories: list[str] = Field(default_factory=list)
    language: str = ""

    @field_validator("published")
    @classmethod
    def published_date(cls, value: str) -> str:
        return check_time(value, allow_empty=True)


class BankFragment(Contract):
    fragment_id: Text
    page: PositiveInt
    text: Text


class BankMaterial(Contract):
    material_id: Text
    title: Text
    source_file: Text
    document_date: str | None  # YYYY-MM-DD, null при неизвестной дате
    fragments: list[BankFragment] = Field(min_length=1)

    @field_validator("document_date")
    @classmethod
    def date_format(cls, value: str | None) -> str | None:
        if value is not None:
            from datetime import date
            if date.fromisoformat(value).isoformat() != value:
                raise ValueError("Ожидается YYYY-MM-DD")
        return value


class RunOptions(Contract):
    # Стартовые технические defaults; это не согласованные бизнес-пороги.
    top_k: PositiveInt = 5
    max_refinement_rounds: Annotated[int, Field(strict=True, ge=0)] = 1
    max_article_fetches: Annotated[int, Field(strict=True, ge=0)] = 15
    max_article_chars: PositiveInt = 20000
    counterparty_depth: Literal[1] = 1


class RelevanceInput(Contract):
    schema_version: Literal["1.0"] = "1.0"
    request_id: Text
    as_of: str
    company_profile: CompanyProfile
    news: list[NewsItem]
    bank_materials: list[BankMaterial]
    options: RunOptions = Field(default_factory=RunOptions)

    @field_validator("as_of")
    @classmethod
    def timestamp(cls, value: str) -> str:
        return check_time(value)

    @model_validator(mode="after")
    def ids(self):
        unique([n.id for n in self.news], "news")
        unique([m.material_id for m in self.bank_materials], "bank_materials")
        unique([f.fragment_id for m in self.bank_materials for f in m.fragments], "fragments")
        return self


class ArticleResult(Contract):
    news_id: Text
    status: Literal["ok", "unavailable", "not_requested"]
    requested_url: str
    final_url: str | None = None
    text: str | None = None
    truncated: bool = False
    error: str | None = None

    @model_validator(mode="after")
    def status_fields(self):
        if self.status == "ok":
            if not self.text or not self.text.strip() or not self.final_url or self.error:
                raise ValueError("ok требует text, final_url и отсутствия error")
        else:
            if self.text is not None or self.truncated:
                raise ValueError("Без успешной загрузки text=null, truncated=false")
            if self.status == "unavailable" and not self.error:
                raise ValueError("unavailable требует причины в error")
            if self.status == "not_requested" and self.error:
                raise ValueError("not_requested не является ошибкой загрузки")
        return self


class ProfileResult(Contract):
    company_id: Text
    status: Literal["full", "partial", "unavailable"]
    profile: CompanyProfile | None
    sources: list[Text]
    limitations: list[Text]

    @model_validator(mode="after")
    def profile_status(self):
        if self.status == "unavailable":
            if self.profile is not None or not self.limitations:
                raise ValueError("unavailable требует profile=null и limitations")
        elif self.profile is None or self.profile.company_id != self.company_id:
            raise ValueError("Профиль должен соответствовать company_id")
        return self


class CounterpartyLink(Contract):
    company_id: Text
    direction: Literal["outgoing", "incoming"]
    amount_rub: Annotated[float, Field(ge=0)]
    payments_count: Annotated[int, Field(strict=True, ge=0)]
    share: Annotated[float, Field(ge=0, le=1)] | None
    source: Text = "applications/csv/df_Uid_payment_hack.csv"


class CounterpartiesResult(Contract):
    company_id: Text
    links: list[CounterpartyLink]
    complete: bool  # Полнота обхода доступного набора, не полнота бизнеса.
    limitations: list[Text]

    @model_validator(mode="after")
    def links_unique(self):
        unique([f"{x.company_id}:{x.direction}" for x in self.links], "links")
        if any(x.company_id == self.company_id for x in self.links):
            raise ValueError("Самоплатежи исключаются")
        if not self.complete and not self.limitations:
            raise ValueError("Неполный обход требует объяснения")
        return self


class BankReference(Contract):
    material_id: Text
    fragment_id: Text


class BankSearchResult(Contract):
    query: Text
    hits: list[BankReference]
    limitations: list[Text]


class FeatureEvidence(Contract):
    field_path: Text  # JSON Pointer, например /bank_relationship/products/leasing/has
    kind: Literal["source_fact", "calculated", "heuristic"]
    explanation: Text


class CounterpartyEvidence(Contract):
    company_id: Text
    direction: Literal["outgoing", "incoming"]
    share: Annotated[float, Field(ge=0, le=1)] | None
    features: list[FeatureEvidence] = Field(min_length=1)
    connection: Text
    limitations: list[Text]


class NewsAssessment(Contract):
    news_id: Text
    status: Literal["scored", "not_scored"]
    relevance_score: Score | None
    rank: PositiveInt | None
    selected: bool
    reason: Text
    company_evidence: list[FeatureEvidence]
    counterparty_evidence: list[CounterpartyEvidence]
    bank_evidence: list[BankReference]
    article: ArticleResult
    limitations: list[Text]

    @model_validator(mode="after")
    def score_status(self):
        if self.article.news_id != self.news_id:
            raise ValueError("Статья относится к другой новости")
        if self.status == "scored":
            if self.relevance_score is None or self.rank is None:
                raise ValueError("scored требует score и rank")
        elif self.relevance_score is not None or self.rank is not None or self.selected:
            raise ValueError("not_scored: score=null, rank=null, selected=false")
        return self


class RelevanceOutput(Contract):
    schema_version: Literal["1.0"] = "1.0"
    request_id: Text
    company_id: Text
    as_of: str
    results: list[NewsAssessment]
    selected_news_ids: list[Text]
    warnings: list[Text]

    @field_validator("as_of")
    @classmethod
    def timestamp(cls, value: str) -> str:
        return check_time(value)

    @model_validator(mode="after")
    def ranking(self):
        unique([n.news_id for n in self.results], "results")
        scored = sorted((n for n in self.results if n.status == "scored"), key=lambda n: n.rank)
        if [n.rank for n in scored] != list(range(1, len(scored) + 1)):
            raise ValueError("Ранги должны идти подряд от 1")
        # При равном балле сортируем по строковому news_id: воспроизводимый tie-breaker.
        expected = sorted(scored, key=lambda n: (-n.relevance_score, n.news_id))
        if [n.news_id for n in scored] != [n.news_id for n in expected]:
            raise ValueError("Порядок: score по убыванию, затем news_id по возрастанию")
        if self.selected_news_ids != [n.news_id for n in scored if n.selected]:
            raise ValueError("selected_news_ids не соответствует выбранным новостям")
        return self


def resolve_pointer(document: dict[str, Any], pointer: str) -> Any:
    if not pointer.startswith("/"):
        raise ValueError("field_path должен быть JSON Pointer, начинающимся с /")
    current: Any = document
    try:
        for part in pointer[1:].split("/"):
            key = part.replace("~1", "/").replace("~0", "~")
            current = current[int(key)] if isinstance(current, list) else current[key]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"Отсутствующее поле: {pointer}") from exc
    return current


def validate_exchange(request: RelevanceInput, response: RelevanceOutput,
                      links: CounterpartiesResult | None = None,
                      profiles: list[ProfileResult] | None = None) -> None:
    """Проверяет ссылки и соответствие запуска. Не доказывает истинность объяснений."""
    if (response.request_id, response.company_id, response.as_of) != (
        request.request_id, request.company_profile.company_id, request.as_of
    ):
        raise ValueError("Результат относится к другому запуску/компании/моменту")
    news = {n.id: n for n in request.news}
    if set(news) != {n.news_id for n in response.results}:
        raise ValueError("Нужен ровно один результат для каждой входной новости")
    if len(response.selected_news_ids) > request.options.top_k:
        raise ValueError("Превышен top_k")
    bank = {(m.material_id, f.fragment_id) for m in request.bank_materials for f in m.fragments}
    profile_map = {p.company_id: p for p in profiles or []}
    if links and links.company_id != request.company_profile.company_id:
        raise ValueError("Связи относятся к другой компании")
    link_map = {(x.company_id, x.direction): x for x in links.links} if links else {}
    for item in response.results:
        if item.article.requested_url != news[item.news_id].link:
            raise ValueError("URL статьи не соответствует входу")
        for feature in item.company_evidence:
            resolve_pointer(request.company_profile.model_dump(), feature.field_path)
        for ref in item.bank_evidence:
            if (ref.material_id, ref.fragment_id) not in bank:
                raise ValueError("Неизвестный фрагмент банковской аналитики")
        for evidence in item.counterparty_evidence:
            link = link_map.get((evidence.company_id, evidence.direction))
            profile = profile_map.get(evidence.company_id)
            if not link or not profile or profile.profile is None:
                raise ValueError("Нет подтверждённой связи или профиля контрагента")
            if evidence.share != link.share:
                raise ValueError("Доля контрагента не совпадает с инструментом")
            for feature in evidence.features:
                resolve_pointer(profile.profile.model_dump(), feature.field_path)
