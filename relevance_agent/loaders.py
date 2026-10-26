"""Чтение существующих файлов без их изменения и без сетевых запросов."""
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .contracts import BankMaterial, CompanyProfile, NewsItem, check_time, unique

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILES = ROOT / 'applications/filtered_data/company_profiles.jsonl'


class InputDataError(ValueError):
    """Ошибка входных данных с указанием файла/записи, без вывода всего содержимого."""


def _reject_constant(value: str):
    raise ValueError(f'Недопустимое JSON-число: {value}')


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Повторяющийся JSON-ключ: {key}')
        result[key] = value
    return result


def parse_json(text: str, location: str) -> Any:
    try:
        return json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_unique_keys)
    except ValueError as exc:
        raise InputDataError(f'{location}: некорректный JSON ({exc})') from exc


def read_json(path: str | Path) -> Any:
    path = Path(path)
    try:
        text = path.read_text(encoding='utf-8-sig')
    except (OSError, UnicodeError) as exc:
        raise InputDataError(f'Не удалось прочитать {path}: {exc}') from exc
    return parse_json(text, str(path))


def _validate(model, value: Any, location: str):
    try:
        return model.model_validate(value)
    except ValidationError as exc:
        # Не печатаем исходные значения: в профиле могут быть чувствительные данные.
        details = '; '.join(
            f"{'.'.join(map(str, e['loc'])) or '<объект>'}: {e['msg']}"
            for e in exc.errors(include_input=False, include_url=False)
        )
        raise InputDataError(f'{location}: {details}') from exc


def load_company_profile(company_id: str, path: str | Path = DEFAULT_PROFILES) -> CompanyProfile:
    """Выбирает полный профиль из JSONL; brief намеренно не подставляется молча."""
    if not isinstance(company_id, str) or not company_id.strip():
        raise InputDataError('company_id должен быть непустой строкой')
    path = Path(path)
    found = None
    seen = set()
    try:
        with path.open(encoding='utf-8-sig') as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                location = f'{path}, строка {number}'
                data = parse_json(line, location)
                if not isinstance(data, dict) or not isinstance(data.get('company_id'), str) or not data['company_id'].strip():
                    raise InputDataError(f'{location}: отсутствует корректный company_id')
                uid = data['company_id']
                if uid in seen:
                    raise InputDataError(f'{location}: повтор company_id={uid}')
                seen.add(uid)
                if uid != company_id:
                    continue
                if not isinstance(data.get('industry'), dict) or not isinstance(data.get('scale'), dict):
                    raise InputDataError(f'{location}: ожидается полный профиль с блоками industry и scale, а не brief')
                object_blocks = ['legal', 'geography', 'foreign_trade', 'bank_relationship',
                                 'group', 'counterparties', 'related_individuals', 'data_quality', 'raw']
                for key in object_blocks:
                    if key in data and not isinstance(data[key], dict):
                        raise InputDataError(f'{location}: {key} должен быть объектом')
                for key in ['sensitivities', 'news_keywords']:
                    if key in data and not isinstance(data[key], list):
                        raise InputDataError(f'{location}: {key} должен быть списком')
                found = _validate(CompanyProfile, data, location)
    except (OSError, UnicodeError) as exc:
        raise InputDataError(f'Не удалось прочитать {path}: {exc}') from exc
    if found is None:
        raise InputDataError(f'Компания {company_id} не найдена в {path}')
    return found


def load_news(path: str | Path) -> list[NewsItem]:
    """Принимает JSON RSS-парсера {generated_at,count,items} либо список NewsItem."""
    data = read_json(path)
    if isinstance(data, dict):
        if set(data) != {'generated_at', 'count', 'items'}:
            raise InputDataError(f'{path}: ожидается обёртка RSS generated_at/count/items либо список новостей')
        items = data['items']
        if not isinstance(items, list):
            raise InputDataError(f'{path}: items должен быть списком')
        if type(data['count']) is not int or data['count'] != len(items):
            raise InputDataError(f'{path}: count не соответствует числу items')
        try:
            if not isinstance(data['generated_at'], str):
                raise ValueError('Ожидается строка')
            check_time(data['generated_at'])
        except ValueError as exc:
            raise InputDataError(f'{path}: некорректная дата generated_at') from exc
    elif isinstance(data, list):
        items = data
    else:
        raise InputDataError(f'{path}: ожидается список или обёртка RSS')
    result = [_validate(NewsItem, item, f'{path}, новость #{i}') for i, item in enumerate(items, 1)]
    try:
        unique([item.id for item in result], 'news')
    except ValueError as exc:
        raise InputDataError(f'{path}: {exc}') from exc
    return result


def load_bank_materials(path: str | Path | None) -> list[BankMaterial]:
    """JSON-список подготовленных материалов. None означает, что текстов ещё нет."""
    if path is None:
        return []
    data = read_json(path)
    if not isinstance(data, list):
        raise InputDataError(f'{path}: аналитика должна быть JSON-списком BankMaterial')
    result = [_validate(BankMaterial, item, f'{path}, материал #{i}') for i, item in enumerate(data, 1)]
    try:
        unique([m.material_id for m in result], 'bank_materials')
        unique([f.fragment_id for m in result for f in m.fragments], 'fragments')
    except ValueError as exc:
        raise InputDataError(f'{path}: {exc}') from exc
    for material in result:
        for fragment in material.fragments:
            if not fragment.text.strip():
                raise InputDataError(f'{path}: пустой текст фрагмента {fragment.fragment_id}')
    return result
