"""Тонкий клиент к OpenAI-совместимому API хакатона.

Свой клиент вместо обёрток LangChain: нужен полный контроль над вызовом инструментов,
строгим JSON, учётом стоимости (`usage.cost`) и починкой ответов.
"""
import json
import os
import re
import ssl
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import certifi
import httpx
from django.conf import settings
from pydantic import BaseModel, ValidationError

DEFAULT_BASE_URL = "https://changellenge.up.railway.app/v1"
RETRY_STATUSES = {429, 500, 502, 503, 504}
# Обрыв соединения у провайдера обычно происходит через ~60 с, поэтому повторов немного:
# дальше выручает запасная модель (см. structured(fallback=…)).
MAX_HTTP_ATTEMPTS = 2
MAX_REPAIR_ATTEMPTS = 2

T = TypeVar("T", bound=BaseModel)


def load_env() -> None:
    """Подхватывает .env из корня проекта, не перезаписывая уже заданные переменные."""
    path = Path(settings.BASE_DIR) / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    model: str
    content: str
    tool_calls: list[dict]
    message: dict
    usage: dict
    cost: float
    latency: float
    finish_reason: str | None = None


@dataclass
class Trace:
    """Журнал шагов одного прогона: вызовы моделей, инструментов и проверки кода."""

    steps: list[dict] = field(default_factory=list)
    cost: float = 0.0

    def add(self, **entry: Any) -> dict:
        entry.setdefault("n", len(self.steps) + 1)
        self.steps.append(entry)
        self.cost += entry.get("cost") or 0.0
        return entry


def strict_schema(model: type[BaseModel]) -> dict:
    """JSON Schema из Pydantic в строгом виде: все поля обязательны, лишние запрещены."""
    schema = model.model_json_schema()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            node.pop("default", None)
            # Ключевое слово title — строка; поле модели с именем title внутри properties — словарь.
            if isinstance(node.get("title"), str):
                node.pop("title")
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(schema)
    return schema


def extract_json(text: str) -> str:
    """Достаёт JSON из ответа: убирает markdown-обёртку и текст вокруг объекта."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start:end + 1]
    return text


class LLMClient:
    def __init__(self, trace: Trace, base_url: str | None = None, api_key: str | None = None):
        load_env()
        self.trace = trace
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY")
        if not self.api_key:
            raise LLMError("Не задан LLM_API_KEY (в .env или окружении)")
        # python.org-сборка Python на macOS не видит системные сертификаты.
        self.http = httpx.Client(timeout=180, verify=ssl.create_default_context(cafile=certifi.where()))

    def chat(self, *, step: str, model: str, messages: list[dict], tools: list[dict] | None = None,
             response_format: dict | None = None, max_tokens: int = 4000,
             temperature: float = 0.2, extra: dict | None = None,
             fallback: tuple[str, dict | None] | None = None) -> LLMResponse:
        """Один запрос к модели. fallback — (модель, параметры), если основная модель недоступна."""
        def build(model_id: str, params: dict | None) -> dict:
            body: dict[str, Any] = {"model": model_id, "messages": messages, "max_tokens": max_tokens,
                                    "temperature": temperature}
            if tools:
                body["tools"] = tools
            if response_format:
                body["response_format"] = response_format
            if params:
                body.update(params)
            return body

        started = time.monotonic()
        try:
            data = self._post(build(model, extra))
        except LLMError as exc:
            if not fallback or model == fallback[0]:
                raise
            self.trace.add(kind="fallback", step=step, failed_model=model, error=str(exc)[:300],
                           fallback_model=fallback[0])
            model = fallback[0]
            data = self._post(build(*fallback))
        latency = time.monotonic() - started

        message = data["choices"][0]["message"]
        usage = data.get("usage") or {}
        response = LLMResponse(
            model=model,
            content=message.get("content") or "",
            tool_calls=message.get("tool_calls") or [],
            message=message,
            usage=usage,
            cost=float(usage.get("cost") or 0.0),
            latency=latency,
            finish_reason=data["choices"][0].get("finish_reason"),
        )
        self.trace.add(
            kind="llm", step=step, model=model, latency_s=round(latency, 2), cost=response.cost,
            prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
            reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            tool_calls=[{"name": c["function"]["name"], "arguments": c["function"]["arguments"]}
                        for c in response.tool_calls],
            finish_reason=data["choices"][0].get("finish_reason"),
        )
        return response

    def structured(self, *, step: str, model: str, messages: list[dict], schema: type[T],
                   max_tokens: int = 8000, temperature: float = 0.2, extra: dict | None = None,
                   fallback: tuple[str, dict | None] | None = None) -> T:
        """Ответ строго по Pydantic-схеме; при ошибке валидации просит модель исправиться.

        fallback — (модель, параметры), на которую переключаемся, если основная модель недоступна.
        """
        response_format = {"type": "json_schema", "json_schema": {
            "name": schema.__name__, "strict": True, "schema": strict_schema(schema)}}
        messages = list(messages)
        last_error = ""
        for attempt in range(MAX_REPAIR_ATTEMPTS + 1):
            kwargs = dict(messages=messages, response_format=response_format, max_tokens=max_tokens,
                          temperature=temperature)
            step_name = step if not attempt else f"{step}:repair{attempt}"
            response = self.chat(step=step_name, model=model, extra=extra, fallback=fallback, **kwargs)
            if response.model != model:  # основная модель недоступна — дальше работаем с запасной
                model, extra = fallback
            if not response.content.strip() or response.finish_reason == "length":
                # Ответ упёрся в лимит токенов (или весь ушёл на рассуждения) — повторяем с запасом,
                # просить модель «исправить» обрезанный JSON бессмысленно.
                last_error = "ответ обрезан по лимиту токенов"
                max_tokens = int(max_tokens * 1.6)
                continue
            try:
                return schema.model_validate_json(extract_json(response.content))
            except ValidationError as exc:
                last_error = exc.json(include_url=False)[:3000]
                messages = messages + [
                    {"role": "assistant", "content": response.content},
                    {"role": "user", "content": "Ответ не прошёл проверку схемы. Ошибки:\n"
                                                f"{last_error}\nВерни исправленный JSON целиком, без пояснений."},
                ]
        raise LLMError(f"{step}: модель не вернула валидный JSON за {MAX_REPAIR_ATTEMPTS + 1} попыток: {last_error}")

    def _post(self, body: dict) -> dict:
        last = ""
        for attempt in range(MAX_HTTP_ATTEMPTS):
            try:
                resp = self.http.post(f"{self.base_url}/chat/completions", json=body,
                                      headers={"Authorization": f"Bearer {self.api_key}"})
            except httpx.HTTPError as exc:
                last = str(exc)
            else:
                if resp.status_code == 200:
                    return resp.json()
                last = f"HTTP {resp.status_code}: {resp.text[:500]}"
                if resp.status_code not in RETRY_STATUSES:
                    break
            time.sleep(2 ** attempt)
        raise LLMError(f"Запрос к модели {body['model']} не удался: {last}")


def tool_spec(name: str, description: str, parameters: dict) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)
