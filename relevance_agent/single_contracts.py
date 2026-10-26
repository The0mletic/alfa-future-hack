"""Primary event contract: one news item for one client; no feed ranking."""
from datetime import datetime, timezone
from uuid import uuid4
from typing import Literal
from pydantic import Field, field_validator, model_validator
from .contracts import (Contract, Text, Score, NewsItem, RunOptions, check_time,
    FeatureEvidence, CounterpartyEvidence, BankReference, ArticleResult,
    CompanyProfile, CounterpartiesResult, ProfileResult, BankMaterial)


class SingleNewsInput(Contract):
    schema_version: Literal['1.0-single'] = '1.0-single'
    request_id: Text = Field(default_factory=lambda: str(uuid4()))
    company_id: Text
    news: NewsItem
    as_of: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    options: RunOptions = Field(default_factory=RunOptions)

    @field_validator('as_of')
    @classmethod
    def timestamp(cls, value):
        return check_time(value)


class ModelAssessment(Contract):
    relevance_score: Score
    reason: Text
    company_evidence: list[FeatureEvidence]
    counterparty_evidence: list[CounterpartyEvidence]
    bank_evidence: list[BankReference]
    limitations: list[Text]

    @model_validator(mode='after')
    def grounded(self):
        if self.relevance_score > 0 and not (self.company_evidence or self.counterparty_evidence):
            raise ValueError('Ненулевая оценка требует признака компании или контрагента')
        return self


class SingleNewsOutput(Contract):
    schema_version: Literal['1.0-single'] = '1.0-single'
    request_id: Text
    company_id: Text
    news: NewsItem
    as_of: str
    status: Literal['scored', 'not_scored']
    assessment: ModelAssessment | None
    article: ArticleResult
    company_profile: CompanyProfile
    counterparties: CounterpartiesResult
    counterparty_profiles: list[ProfileResult]
    bank_materials: list[BankMaterial]
    bank_metadata: list[dict]
    warnings: list[Text]
    model: str
    model_calls: int
    usage: list[dict]

    @model_validator(mode='after')
    def consistent(self):
        check_time(self.as_of)
        if (self.status == 'scored') != (self.assessment is not None):
            raise ValueError('Оценка допустима только для scored')
        if self.article.news_id != self.news.id or self.article.requested_url != self.news.link:
            raise ValueError('Статья не соответствует новости')
        if self.company_profile.company_id != self.company_id or self.counterparties.company_id != self.company_id:
            raise ValueError('Контекст относится к другой компании')
        return self
