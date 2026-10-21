"""Очистка исходных CSV кейса под целевую аудиторию (крупный бизнес).

Запуск из корня репозитория:
    python scripts/clean_data.py

Читает applications/csv, пишет результат в applications/filtered_data.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "applications" / "csv"
OUT_DIR = ROOT / "applications" / "filtered_data"

SEP = ";"

# Рабочий набор: на нём прогоняем и обобщаем сервис.
# Порог ниже кейсового, чтобы не терять компании около границы.
MIN_REVENUE_RUB = 10_000_000_000
# Целевая аудитория по определению кейса (выручка свыше 15 млрд ₽) — для демо.
TARGET_REVENUE_RUB = 15_000_000_000


def filter_by_revenue(df: pd.DataFrame, min_revenue: int) -> pd.DataFrame:
    df = df[df["REVENUE_COMPANY"] >= min_revenue]
    return df.reset_index(drop=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    companies = pd.read_csv(SRC_DIR / "df_Uid_data_hack.csv", sep=SEP, low_memory=False)
    outputs = {
        "df_Uid_data_hack.csv": MIN_REVENUE_RUB,
        "df_Uid_data_hack_target.csv": TARGET_REVENUE_RUB,
    }
    for name, min_revenue in outputs.items():
        cleaned = filter_by_revenue(companies, min_revenue)
        cleaned.to_csv(OUT_DIR / name, sep=SEP, index=False)
        print(f"{name}: {len(companies)} -> {len(cleaned)} строк")


if __name__ == "__main__":
    main()
