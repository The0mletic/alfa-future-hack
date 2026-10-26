"""Сборка проверенного входа; это ещё не запуск ранжировщика."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from uuid import uuid4

from .contracts import RelevanceInput, RunOptions
from .loaders import (DEFAULT_PROFILES, InputDataError, _validate, load_bank_materials,
                      load_company_profile, load_news)

MODULE_DIR = Path(__file__).resolve().parent


def build_relevance_input(*, company_id: str, news_path: str | Path,
                          profiles_path: str | Path = DEFAULT_PROFILES,
                          bank_materials_path: str | Path | None = None,
                          request_id: str | None = None, as_of: str | None = None,
                          options: RunOptions | dict | None = None) -> RelevanceInput:
    """Читает файлы и возвращает модель. Ничего не записывает, не вызывает LLM."""
    return _validate(RelevanceInput, {
        'request_id': str(uuid4()) if request_id is None else request_id,
        'as_of': datetime.now(timezone.utc).isoformat() if as_of is None else as_of,
        'company_profile': load_company_profile(company_id, profiles_path),
        'news': load_news(news_path),
        'bank_materials': load_bank_materials(bank_materials_path),
        'options': RunOptions() if options is None else options,
    }, 'Вход агента')


def describe_limitations(request: RelevanceInput) -> list[str]:
    """Диагностика сборки; не меняет контракт и не отбрасывает новости."""
    messages = []
    if not request.news:
        messages.append('Список новостей пуст.')
    if not request.bank_materials:
        messages.append('Аналитика банка не предоставлена; сквозной анализ с ней пока невозможен.')
    for field, label in [('published', 'дата публикации'), ('summary', 'описание'), ('link', 'ссылка')]:
        count = sum(not getattr(n, field).strip() for n in request.news)
        if count:
            messages.append(f'У {count} новостей отсутствует {label}.')
    cutoff = datetime.fromisoformat(request.as_of.replace('Z', '+00:00'))
    future = sum(bool(n.published) and datetime.fromisoformat(n.published.replace('Z', '+00:00')) > cutoff for n in request.news)
    if future:
        messages.append(f'{future} новостей опубликованы после as_of; перед оценкой нужна проверка временного окна.')
    if not request.company_profile.model_dump().get('sensitivities'):
        messages.append('В профиле нет заполненного списка sensitivities.')
    return messages


def save_input(request: RelevanceInput, destination: str | Path) -> Path:
    """Пишет только внутрь relevance_agent; существующие файлы не перезаписывает."""
    path = Path(destination).resolve()
    if not path.is_relative_to(MODULE_DIR):
        raise InputDataError('Выходной файл должен находиться внутри relevance_agent/')
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open('x', encoding='utf-8') as stream:
            stream.write(request.model_dump_json(indent=2) + '\n')
    except FileExistsError as exc:
        raise InputDataError(f'Файл уже существует: {path}. Выберите новое имя.') from exc
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Собрать вход агента из существующих файлов; без LLM и сети')
    parser.add_argument('--company-id', required=True)
    parser.add_argument('--news', required=True, type=Path)
    parser.add_argument('--profiles', type=Path, default=DEFAULT_PROFILES)
    parser.add_argument('--bank-materials', type=Path)
    parser.add_argument('--request-id')
    parser.add_argument('--as-of', help='ISO-8601 с часовым поясом; по умолчанию текущее UTC')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--output', type=Path, help='Новый файл внутри relevance_agent; иначе JSON в stdout')
    args = parser.parse_args(argv)
    try:
        request = build_relevance_input(company_id=args.company_id, news_path=args.news,
            profiles_path=args.profiles, bank_materials_path=args.bank_materials,
            request_id=args.request_id, as_of=args.as_of, options={'top_k': args.top_k})
        for message in describe_limitations(request):
            print(f'Предупреждение: {message}', file=sys.stderr)
        if args.output:
            path = save_input(request, args.output)
            print(f'Вход сохранён: {path}; компания={request.company_profile.company_id}, '
                  f'новостей={len(request.news)}, материалов банка={len(request.bank_materials)}', file=sys.stderr)
        else:
            print(request.model_dump_json(indent=2))
        return 0
    except (InputDataError, OSError) as exc:
        print(f'Ошибка: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
