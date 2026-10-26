"""Explicit configuration for an OpenAI-compatible Chat Completions endpoint."""
import os
from pathlib import Path
from urllib.parse import urlsplit
from dotenv import dotenv_values
from langchain_openai import ChatOpenAI


def load_model():
    local = dotenv_values(Path(__file__).parent / '.env')
    values = {name: os.environ.get(name) or local.get(name) for name in (
        'RELEVANCE_API_KEY', 'RELEVANCE_BASE_URL', 'RELEVANCE_MODEL')}
    # The team keeps its key in the root .env; read it without modifying/copying it.
    if not values['RELEVANCE_API_KEY']:
        shared = dotenv_values(Path(__file__).resolve().parents[1] / '.env')
        values['RELEVANCE_API_KEY'] = os.environ.get('LLM_API_KEY') or shared.get('LLM_API_KEY')
    missing = [k for k, v in values.items() if not v]
    if missing:
        raise ValueError('Заполните relevance_agent/.env: ' + ', '.join(missing))
    url = urlsplit(values['RELEVANCE_BASE_URL'])
    if url.scheme != 'https' or not url.hostname or url.username or url.password:
        raise ValueError('RELEVANCE_BASE_URL должен быть HTTPS URL без credentials')
    return ChatOpenAI(
        model=values['RELEVANCE_MODEL'], api_key=values['RELEVANCE_API_KEY'],
        base_url=values['RELEVANCE_BASE_URL'], temperature=0,
        max_tokens=3000, timeout=60, max_retries=0, use_responses_api=False,
    )
