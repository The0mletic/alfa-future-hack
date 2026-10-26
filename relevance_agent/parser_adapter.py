"""Normalize the team's grouped rss_full_parser report without running the parser."""
import hashlib
from typing import Annotated
from pydantic import Field, field_validator, model_validator
from .contracts import Contract, Text, NewsItem, check_time

Count = Annotated[int, Field(strict=True, ge=0)]


class FullParserItem(Contract):
    title: Text
    link: Text
    published: str
    author: str
    summary_rss: str
    full_text: str
    content_for_llm: str
    categories: list[str]

    @model_validator(mode='after')
    def content_consistent(self):
        if self.content_for_llm != (self.full_text or self.summary_rss):
            raise ValueError('content_for_llm не соответствует full_text/summary_rss')
        return self


class FullParserSource(Contract):
    source: Text
    feed_url: Text
    items: list[FullParserItem]


class FullParserReport(Contract):
    generated_at: str
    total_sources: Count
    total_items: Count
    sources: list[FullParserSource]

    @field_validator('generated_at')
    @classmethod
    def timestamp(cls, value):
        return check_time(value)

    @model_validator(mode='after')
    def counts(self):
        if self.total_sources != len(self.sources):
            raise ValueError('total_sources не соответствует sources')
        if self.total_items != sum(len(source.items) for source in self.sources):
            raise ValueError('total_items не соответствует items')
        return self


def normalize_report(data: dict) -> list[dict]:
    report = FullParserReport.model_validate(data)
    items = []
    for source in report.sources:
        for item in source.items:
            # Same ID rule as scripts/rss_news_parser.py; no heavy parser imports.
            uid = hashlib.sha1(item.link.strip().lower().encode('utf-8')).hexdigest()[:16]
            news = NewsItem(id=uid, title=item.title, link=item.link,
                summary=item.summary_rss, full_text=item.full_text, published=item.published,
                source=source.source, author=item.author, categories=item.categories)
            items.append(news.model_dump())
    return items
