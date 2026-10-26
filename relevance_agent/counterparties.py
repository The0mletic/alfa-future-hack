"""Инструменты прямых платёжных связей. Без LLM, сети и изменения датасета."""
import argparse
from collections import defaultdict
import csv
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import sys

from .contracts import CompanyProfile, CounterpartiesResult, CounterpartyLink, ProfileResult
from .loaders import DEFAULT_PROFILES, ROOT, InputDataError, _validate, parse_json

CSV_DIR = ROOT / 'applications/csv'


def _identifier(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputDataError('company_id должен быть непустой строкой')
    return value


def _rows(path: Path, required: set[str]):
    try:
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream, delimiter=';')
            if not required.issubset(reader.fieldnames or []):
                raise InputDataError(f'{path}: отсутствуют обязательные столбцы {sorted(required)}')
            for row in reader:
                yield reader.line_num, row
    except (OSError, UnicodeError, csv.Error) as exc:
        raise InputDataError(f'Не удалось прочитать {path}: {exc}') from exc


def _number(value: str | None, integer=False):
    try:
        number = Decimal(value or '')
        if not number.is_finite() or number < 0:
            raise ValueError()
        if integer and number != number.to_integral_value():
            raise ValueError()
        return int(number) if integer else number
    except (ValueError, InvalidOperation) as exc:
        raise ValueError('Ожидается конечное неотрицательное число') from exc


class CounterpartyTools:
    """Один экземпляр на запуск. Индексы профилей и связи кэшируются в памяти.

    Платежи читаются один раз для каждого запрошенного target. Профили
    контрагентов не запускают обход их платежей: глубина всегда один уровень.
    Для обновлённых файлов создайте новый экземпляр.
    """
    def __init__(self, *, payments_path=CSV_DIR / 'df_Uid_payment_hack.csv',
                 companies_path=CSV_DIR / 'df_Uid_data_hack.csv', profiles_path=DEFAULT_PROFILES):
        self.payments_path = Path(payments_path)
        self.companies_path = Path(companies_path)
        self.profiles_path = Path(profiles_path)
        self._profiles = None
        self._companies = None
        self._conflicting_companies = set()
        self._links = {}
        self._results = {}

    def get_counterparties(self, company_id: str) -> CounterpartiesResult:
        company_id = _identifier(company_id)
        if company_id in self._links:
            return self._links[company_id].model_copy(deep=True)
        amounts = defaultdict(Decimal)
        counts = defaultdict(int)
        invalid = 0
        ambiguous = 0
        for _, row in _rows(self.payments_path, {'PAYER', 'BENEFICIARY', 'PAY_AMT', 'PAY_CNT'}):
            payer, beneficiary = row.get('PAYER'), row.get('BENEFICIARY')
            if not payer or not beneficiary:
                # При неизвестном конце ребра полноту обхода доказать нельзя.
                ambiguous += 1
                continue
            if company_id not in (payer, beneficiary) or payer == beneficiary:
                continue
            direction = 'outgoing' if payer == company_id else 'incoming'
            cp_id = beneficiary if direction == 'outgoing' else payer
            try:
                if None in row or any(row.get(k) is None for k in ('PAY_AMT', 'PAY_CNT')):
                    raise ValueError()
                amount = _number(row['PAY_AMT'])
                count = _number(row['PAY_CNT'], integer=True)
                # Контракт использует конечный float; огромные числа недопустимы.
                if amount > Decimal('1e300'):
                    raise ValueError()
            except ValueError:
                invalid += 1
                continue
            key = (cp_id, direction)
            amounts[key] += amount
            counts[key] += count
        totals = {d: sum((a for (_, side), a in amounts.items() if side == d), Decimal(0))
                  for d in ('incoming', 'outgoing')}
        links = [CounterpartyLink(company_id=cp, direction=direction,
                     amount_rub=float(amount), payments_count=counts[(cp, direction)],
                     share=float(amount / totals[direction]) if totals[direction] else None,
                     source=str(self.payments_path))
                 for (cp, direction), amount in amounts.items()]
        links.sort(key=lambda link: (link.direction, -link.amount_rub, link.company_id))
        limitations = [
            'Видны только платежи предоставленного набора; период не указан.',
            'Направление платежа не доказывает роль поставщика или покупателя.',
            'Доли рассчитаны отдельно по входящим и исходящим суммам доступных корректных записей.'
        ]
        if invalid:
            limitations.append(f'Пропущено некорректных платежных записей целевой компании: {invalid}. Доли неполные.')
        if ambiguous:
            limitations.append(f'Записей с неизвестным участником платежа: {ambiguous}. Полнота связей не гарантируется.')
        if not links:
            limitations.append('Прямых связей в корректных записях не найдено; это не доказывает отсутствие контрагентов.')
        if any(value == 0 for value in totals.values()) and links:
            limitations.append('При нулевой сумме направления доля не определена (null).')
        result = CounterpartiesResult(company_id=company_id, links=links,
                                     complete=not (invalid or ambiguous), limitations=limitations)
        self._links[company_id] = result
        return result.model_copy(deep=True)

    def _load_profiles(self):
        if self._profiles is not None:
            return
        profiles = {}
        try:
            with self.profiles_path.open(encoding='utf-8-sig') as stream:
                for line_no, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    location = f'{self.profiles_path}, строка {line_no}'
                    data = parse_json(line, location)
                    profile = _validate(CompanyProfile, data, location)
                    if not profile.company_id.strip() or not isinstance(data.get('industry'), dict) or not isinstance(data.get('scale'), dict):
                        raise InputDataError(f'{location}: ожидается полный профиль')
                    if profile.company_id in profiles:
                        raise InputDataError(f'{location}: повтор company_id')
                    profiles[profile.company_id] = profile
        except (OSError, UnicodeError) as exc:
            raise InputDataError(f'Не удалось прочитать {self.profiles_path}: {exc}') from exc
        self._profiles = profiles

    def _load_companies(self):
        if self._companies is not None:
            return
        companies = {}
        conflicts = set()
        for line_no, row in _rows(self.companies_path, {'SYNTHETIC_U', 'MAINOKVED_CODE', 'MAINOKVED_NAME'}):
            uid = row['SYNTHETIC_U']
            if not uid or not uid.strip() or None in row or any(v is None for v in row.values()):
                raise InputDataError(f'{self.companies_path}, строка {line_no}: некорректная запись компании')
            if uid in companies and companies[uid] != row:
                conflicts.add(uid)
            companies[uid] = row
        self._companies = companies
        self._conflicting_companies = conflicts

    def get_company_profile(self, company_id: str) -> ProfileResult:
        company_id = _identifier(company_id)
        if company_id in self._results:
            return self._results[company_id].model_copy(deep=True)
        self._load_profiles()
        if company_id in self._profiles:
            result = ProfileResult(company_id=company_id, status='full', profile=self._profiles[company_id],
                sources=[str(self.profiles_path)],
                limitations=['Полный подготовленный профиль содержит эвристики; учитывать notes и data_quality.'])
        else:
            self._load_companies()
            row = self._companies.get(company_id)
            if company_id in self._conflicting_companies:
                result = ProfileResult(company_id=company_id, status='unavailable', profile=None,
                    sources=[str(self.companies_path)],
                    limitations=['В CSV обнаружены противоречащие записи этой компании. Профиль не выбран; требуется разрешить конфликт.'])
            elif row is None:
                result = ProfileResult(company_id=company_id, status='unavailable', profile=None,
                    sources=[str(self.profiles_path), str(self.companies_path)],
                    limitations=['В таблицах профилей и компаний нет записи; платёжная связь при этом сохраняется.'])
            else:
                # Только исходные сведения; валюты, чувствительности и бизнес-модель не выдумываются.
                raw = {k: v if v != '' else None for k, v in row.items()}
                limits = ['Сокращённая карточка из CSV: связи и чувствительности контрагента не рассчитывались.',
                          'Регион регистрации не обязательно совпадает с регионом деятельности.',
                          'ОКВЭД отражает основной вид деятельности; страны ФТС не определяют валюты и направление торговли.']
                revenue = None
                if raw.get('REVENUE_COMPANY') is not None:
                    try:
                        number = _number(raw['REVENUE_COMPANY'])
                        if number > Decimal('1e300'): raise ValueError()
                        revenue = float(number)
                    except ValueError:
                        limits.append('Выручка не распознана; исходное значение сохранено в raw.')
                profile = CompanyProfile(company_id=company_id,
                    industry={'okved_code': raw.get('MAINOKVED_CODE'), 'okved_name': raw.get('MAINOKVED_NAME')},
                    scale={'revenue_rub': revenue, 'employees_raw':raw.get('AMOUNT_WORKERS_FNS_GRADATION')},
                    geography={'city':raw.get('CITY'), 'region':raw.get('REGION')},
                    foreign_trade={'countries_raw':raw.get('FTS_COUNTRY_NAME'), 'fts_flag_raw':raw.get('FTS_FLAG')},
                    group={'group_id':raw.get('GROUP_ID')}, raw=raw,
                    data_quality={'notes':limits, 'missing_fields':[k for k,v in raw.items() if v is None]})
                result = ProfileResult(company_id=company_id, status='partial', profile=profile,
                                       sources=[str(self.companies_path)], limitations=limits)
        self._results[company_id] = result
        return result.model_copy(deep=True)

    def collect_context(self, company_id: str) -> tuple[CounterpartiesResult, list[ProfileResult]]:
        links = self.get_counterparties(company_id)
        profiles = [self.get_company_profile(uid) for uid in sorted({link.company_id for link in links.links})]
        return links, profiles


def main(argv=None):
    parser = argparse.ArgumentParser(description='Прямые контрагенты и доступные профили; без LLM')
    parser.add_argument('--company-id', required=True)
    parser.add_argument('--payments', type=Path, default=CSV_DIR / 'df_Uid_payment_hack.csv')
    parser.add_argument('--companies', type=Path, default=CSV_DIR / 'df_Uid_data_hack.csv')
    parser.add_argument('--profiles', type=Path, default=DEFAULT_PROFILES)
    args = parser.parse_args(argv)
    try:
        tools = CounterpartyTools(payments_path=args.payments, companies_path=args.companies, profiles_path=args.profiles)
        links, profiles = tools.collect_context(args.company_id)
        print(json.dumps({'links':links.model_dump(), 'profiles':[p.model_dump() for p in profiles]}, ensure_ascii=False, indent=2))
        print(f'Связей: {len(links.links)}, уникальных контрагентов: {len(profiles)}, '
              f'full: {sum(p.status == "full" for p in profiles)}, '
              f'partial: {sum(p.status == "partial" for p in profiles)}, '
              f'unavailable: {sum(p.status == "unavailable" for p in profiles)}', file=sys.stderr)
        return 0
    except (InputDataError, ValueError) as exc:
        print(f'Ошибка: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
