"""Загрузка данных кейса в базу.

    python manage.py import_data                 # облегчённая база (~40 МБ) — её храним в git
    python manage.py import_data --full          # все 85 тыс. компаний и 1,4 млн платежей (~170 МБ)
    python manage.py import_data --skip-payments # без платежей

Облегчённая база содержит то, что нужно агентам анализа: компании рабочего набора (выручка от 10 млрд ₽),
их платежи, контрагентов и компании их групп. Полная не влезает в лимит GitHub на файл (100 МБ).

Перед запуском профили должны быть собраны:
    python scripts/clean_data.py && python scripts/build_profiles.py

Команда идемпотентна: каждая таблица очищается и загружается заново.
"""
import csv
import json
import sys
import time
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand
from django.db import connection, transaction

from companies.models import Company, CompanySensitivity, Payment, PersonLink

DATA_DIR = Path(settings.BASE_DIR) / "applications"
SRC_DIR = DATA_DIR / "csv"
FILTERED_DIR = DATA_DIR / "filtered_data"

# Справочник секторов живёт рядом со скриптами сборки профилей.
sys.path.insert(0, str(Path(settings.BASE_DIR) / "scripts"))
from reference_data import okved_section  # noqa: E402

TARGET_REVENUE_RUB = 15_000_000_000
BATCH_SIZE = 5000


def read_csv(path: Path):
    with open(path, encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f, delimiter=";")


def read_jsonl(path: Path) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        return {row["company_id"]: row for row in map(json.loads, f)}


def as_bool(value: str) -> bool:
    return value not in ("", "0", "0.0")


def as_int(value: str) -> int:
    return int(float(value)) if value else 0


class Command(BaseCommand):
    help = "Загружает компании, профили, платежи и связи с физлицами из applications/ в базу"

    def add_arguments(self, parser):
        parser.add_argument("--skip-payments", action="store_true", help="не загружать платежи")
        parser.add_argument("--full", action="store_true",
                            help="загрузить все компании и платежи, а не только нужные рабочему набору")

    def handle(self, *args, **options):
        started = time.monotonic()
        with transaction.atomic():
            self.import_companies()
            self.import_links()
            if not options["skip_payments"]:
                self.import_payments()
            if not options["full"]:
                self.trim_to_working_set()
        with connection.cursor() as cursor:
            cursor.execute("VACUUM")  # возвращает место на диске после очистки таблиц
        self.stdout.write(self.style.SUCCESS(f"Готово за {time.monotonic() - started:.0f} с"))

    def trim_to_working_set(self) -> None:
        """Оставляет платежи компаний рабочего набора, их контрагентов и компании их групп.

        Удаление сырым SQL: через ORM Django собирал бы каскады по десяткам тысяч компаний.
        """
        working = f"SELECT company_id FROM {Company._meta.db_table} WHERE in_working_set = 1"
        keep = f"""{working}
            UNION SELECT payer_id FROM {Payment._meta.db_table}
            UNION SELECT beneficiary_id FROM {Payment._meta.db_table}
            UNION SELECT company_id FROM {Company._meta.db_table}
                  WHERE group_id IN (SELECT group_id FROM {Company._meta.db_table} WHERE in_working_set = 1)"""
        with connection.cursor() as cursor:
            cursor.execute(f"DELETE FROM {Payment._meta.db_table} "
                           f"WHERE payer_id NOT IN ({working}) AND beneficiary_id NOT IN ({working})")
            cursor.execute(f"DELETE FROM {Company._meta.db_table} WHERE company_id NOT IN ({keep})")
            cursor.execute(f"DELETE FROM {PersonLink._meta.db_table} WHERE company_id NOT IN "
                           f"(SELECT company_id FROM {Company._meta.db_table})")
        self.stdout.write(f"Облегчённая база: компаний {Company.objects.count()}, платежей {Payment.objects.count()}, "
                          f"связей с физлицами {PersonLink.objects.count()}")

    def import_companies(self) -> None:
        working_ids = {row["SYNTHETIC_U"] for row in read_csv(FILTERED_DIR / "df_Uid_data_hack.csv")}
        profiles = read_jsonl(FILTERED_DIR / "company_profiles.jsonl")
        briefs = read_jsonl(FILTERED_DIR / "company_briefs.jsonl")

        Company.objects.all().delete()  # каскадом удаляет и чувствительности
        companies, seen = [], set()
        for row in read_csv(SRC_DIR / "df_Uid_data_hack.csv"):
            uid = row["SYNTHETIC_U"]
            if uid in seen:  # в исходном файле есть 11 дублей ID — берём первую запись
                continue
            seen.add(uid)
            revenue = as_int(row["REVENUE_COMPANY"])
            profile = profiles.get(uid)
            companies.append(Company(
                company_id=uid,
                in_working_set=uid in working_ids,
                is_target=uid in working_ids and revenue >= TARGET_REVENUE_RUB,
                revenue=revenue,
                employees_fns=row["AMOUNT_WORKERS_FNS_GRADATION"],
                employees_spark=row["AMOUNT_WORKERS_SPARK_GRADATION"],
                okved_code=row["MAINOKVED_CODE"],
                okved_name=row["MAINOKVED_NAME"],
                sector=okved_section(row["MAINOKVED_CODE"])[0],
                org_type=row["ORG_TYPE"],
                ownership=row["OKFS_NAME"],
                age_years=row["COMPANY_AGE_YEARS_GRADATION"],
                development_stage=row["ECO_MARK_LABEL"],
                negative_mark=as_bool(row["NEGATIVEMARK"]),
                city=row["CITY"],
                region=row["REGION"],
                has_foreign_trade=as_bool(row["FTS_FLAG"]),
                trade_countries=row["FTS_COUNTRY_NAME"],
                client_type=row["CL_TYPE"],
                has_rko=as_bool(row["CL_PRODUCT_RKO_MARK"]),
                has_guarantees=as_bool(row["CL_PRODUCT_GAR_MARK"]),
                has_loans=as_bool(row["CL_PRODUCT_LOA_MARK"]),
                has_leasing=as_bool(row["CL_PRODUCT_LEA_MARK"]),
                has_deposits=as_bool(row["CL_PRODUCT_DEP_MARK"]),
                has_payroll=as_bool(row["CL_PRODUCT_ZPP_MARK"]),
                has_factoring=as_bool(row["FACTORING_MARK"]),
                group_id=row["GROUP_ID"],
                summary=profile["summary"] if profile else "",
                brief=briefs.get(uid),
                profile=profile,
            ))
        Company.objects.bulk_create(companies, batch_size=BATCH_SIZE)

        sensitivities = [
            CompanySensitivity(
                company_id=uid,
                code=tag["code"],
                label=tag["label"],
                strength=tag["strength"],
                reasons=tag["reasons"],
                source_fields=tag["source_fields"],
                news_keywords=tag["news_keywords"],
            )
            for uid, profile in profiles.items()
            for tag in profile["sensitivities"]
        ]
        CompanySensitivity.objects.bulk_create(sensitivities, batch_size=BATCH_SIZE)

        missing = working_ids - profiles.keys()
        self.stdout.write(
            f"Компании: {len(companies)}, из них рабочий набор {len(working_ids)}, "
            f"с профилем {len(profiles)}; чувствительностей {len(sensitivities)}"
        )
        if missing:
            self.stdout.write(self.style.WARNING(
                f"У {len(missing)} компаний рабочего набора нет профиля — пересоберите scripts/build_profiles.py"
            ))

    def import_links(self) -> None:
        PersonLink.objects.all().delete()
        links = [
            PersonLink(person_id=r["SYNTHETIC_F"], company_id=r["SYNTHETIC_U"],
                       city=r["CITY_NAME"], region=r["RC_NAME"])
            for r in read_csv(SRC_DIR / "df_Fid_Uid_link_hack.csv")
        ]
        PersonLink.objects.bulk_create(links, batch_size=BATCH_SIZE)
        self.stdout.write(f"Связи с физлицами: {len(links)}")

    def import_payments(self) -> None:
        Payment.objects.all().delete()
        batch, total = [], 0
        for r in read_csv(SRC_DIR / "df_Uid_payment_hack.csv"):
            batch.append(Payment(payer_id=r["PAYER"], beneficiary_id=r["BENEFICIARY"],
                                 amount=as_int(r["PAY_AMT"]), count=as_int(r["PAY_CNT"])))
            if len(batch) >= BATCH_SIZE:
                Payment.objects.bulk_create(batch)
                total += len(batch)
                batch = []
        Payment.objects.bulk_create(batch)
        total += len(batch)
        self.stdout.write(f"Платежи: {total}")
