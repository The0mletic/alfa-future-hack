"""Аналитика по рабочему набору компаний (выручка от 10 млрд ₽).

Запуск из корня репозитория (после scripts/clean_data.py):
    python scripts/analyze_companies.py

Компании берутся из applications/filtered_data, платежи и связи с физлицами —
из исходных applications/csv, чтобы видеть всех контрагентов, а не только крупных.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "applications" / "csv"
FILTERED_DIR = ROOT / "applications" / "filtered_data"

SEP = ";"
TARGET_REVENUE_RUB = 15_000_000_000
BN = 1_000_000_000

PRODUCT_COLUMNS = {
    "CL_PRODUCT_RKO_MARK": "РКО",
    "CL_PRODUCT_GAR_MARK": "Гарантии",
    "CL_PRODUCT_LOA_MARK": "Кредиты",
    "CL_PRODUCT_LEA_MARK": "Лизинг",
    "CL_PRODUCT_DEP_MARK": "Депозиты",
    "CL_PRODUCT_ZPP_MARK": "Зарплатный проект",
    "FACTORING_MARK": "Факторинг",
}

pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 60)
pd.set_option("display.max_rows", 200)


def section(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def counts(series: pd.Series, top: int | None = None) -> pd.DataFrame:
    vc = series.value_counts(dropna=False)
    if top:
        vc = vc.head(top)
    return pd.DataFrame({"n": vc, "%": (vc / len(series) * 100).round(1)})


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    companies = pd.read_csv(FILTERED_DIR / "df_Uid_data_hack.csv", sep=SEP)
    all_companies = pd.read_csv(SRC_DIR / "df_Uid_data_hack.csv", sep=SEP, low_memory=False)
    links = pd.read_csv(SRC_DIR / "df_Fid_Uid_link_hack.csv", sep=SEP)
    payments = pd.read_csv(SRC_DIR / "df_Uid_payment_hack.csv", sep=SEP)

    companies["SEGMENT"] = (companies["REVENUE_COMPANY"] >= TARGET_REVENUE_RUB).map(
        {True: "target (>=15)", False: "10-15"}
    )
    companies["OKVED2"] = companies["MAINOKVED_CODE"].astype(str).str.split(".").str[0]
    all_companies = all_companies.drop_duplicates("SYNTHETIC_U")
    return companies, all_companies, links, payments


def overview(c: pd.DataFrame) -> None:
    section("1. Общая картина")
    print(f"Компаний: {len(c)}")
    print(counts(c["SEGMENT"]).to_string())
    print("\nВыручка, млрд ₽:")
    revenue = c.groupby("SEGMENT")["REVENUE_COMPANY"]
    print(pd.concat(
        [revenue.size().rename("n"), (revenue.describe()[["min", "50%", "mean", "max"]] / BN).round(1)],
        axis=1,
    ).to_string())
    print("\nПропуски, %:")
    nulls = (c.isna().mean() * 100).round(1)
    print(nulls[nulls > 0].to_string())


def profile(c: pd.DataFrame) -> None:
    section("2. Профиль компаний")
    for col, title in [
        ("CL_TYPE", "Тип клиента"),
        ("OKFS_NAME", "Форма собственности"),
        ("COMPANY_AGE_YEARS_GRADATION", "Возраст, лет"),
        ("ECO_MARK_LABEL", "Стадия развития"),
        ("AMOUNT_WORKERS_FNS_GRADATION", "Сотрудники (ФНС)"),
        ("NEGATIVEMARK", "NEGATIVEMARK"),
    ]:
        print(f"\n-- {title}")
        print(pd.crosstab(c[col].fillna("<пусто>"), c["SEGMENT"], margins=True, margins_name="всего").to_string())


def industries(c: pd.DataFrame) -> None:
    section("3. Отрасли")
    print("-- Раздел ОКВЭД (2 знака)")
    by2 = c.groupby("OKVED2").agg(
        n=("SYNTHETIC_U", "size"),
        target=("SEGMENT", lambda s: (s == "target (>=15)").sum()),
        revenue_bn=("REVENUE_COMPANY", lambda s: round(s.sum() / BN, 1)),
    ).sort_values("n", ascending=False)
    print(by2.to_string())
    print("\n-- Топ-20 конкретных видов деятельности")
    print(counts(c["MAINOKVED_CODE"].astype(str) + " " + c["MAINOKVED_NAME"], top=20).to_string())


def geography(c: pd.DataFrame) -> None:
    section("4. География и ВЭД")
    print("-- Регион регистрации")
    print(counts(c["REGION"], top=15).to_string())
    print(f"\n-- ВЭД: есть данные ФТС у {int(c['FTS_FLAG'].sum())} из {len(c)}")
    countries = (
        c["FTS_COUNTRY_NAME"].dropna().str.split(",").explode().str.strip()
    )
    print("Страны (компания может торговать с несколькими):")
    print(countries.value_counts().head(20).to_string())


def products(c: pd.DataFrame) -> None:
    section("5. Продукты Альфа-Банка")
    share = c.groupby("SEGMENT")[list(PRODUCT_COLUMNS)].mean().T.rename(index=PRODUCT_COLUMNS)
    share["всего"] = c[list(PRODUCT_COLUMNS)].mean().rename(index=PRODUCT_COLUMNS)
    print("Доля компаний с продуктом, %:")
    print((share * 100).round(1).to_string())
    n_products = c[list(PRODUCT_COLUMNS)].sum(axis=1)
    print("\nКоличество продуктов у компании:")
    print(counts(n_products.astype(int)).sort_index().to_string())
    print("\nКредиты + лизинг (зависимость от заёмного финансирования):",
          int(((c["CL_PRODUCT_LOA_MARK"] == 1) | (c["CL_PRODUCT_LEA_MARK"] == 1)).sum()))


def groups(c: pd.DataFrame, all_c: pd.DataFrame) -> None:
    section("6. Группы компаний")
    group_size = all_c["GROUP_ID"].value_counts()
    c = c.assign(GROUP_SIZE=c["GROUP_ID"].map(group_size))
    print("Размер группы (по всему датасету):")
    print(counts(pd.cut(c["GROUP_SIZE"], [0, 1, 5, 20, 100, 10_000],
                        labels=["1 (одиночка)", "2-5", "6-20", "21-100", ">100"])).sort_index().to_string())
    shared = c["GROUP_ID"].value_counts()
    shared = shared[shared > 1]
    print(f"\nГрупп, где несколько компаний из набора: {len(shared)}")
    for gid, n in shared.items():
        members = c[c["GROUP_ID"] == gid]
        print(f"  {gid}: {n} шт., размер группы {group_size[gid]} —",
              "; ".join(members["MAINOKVED_NAME"].str[:40]))


def related_people(c: pd.DataFrame, links: pd.DataFrame) -> None:
    section("7. Связи через физлиц")
    ids = set(c["SYNTHETIC_U"])
    own = links[links["SYNTHETIC_U"].isin(ids)]
    print(f"Компаний со связанными физлицами: {own['SYNTHETIC_U'].nunique()} из {len(c)}")
    print(f"Связанных физлиц: {own['SYNTHETIC_F'].nunique()}")
    people = own["SYNTHETIC_F"].unique()
    other = links[links["SYNTHETIC_F"].isin(people) & ~links["SYNTHETIC_U"].isin(ids)]
    print(f"Через этих физлиц выходим ещё на {other['SYNTHETIC_U'].nunique()} компаний вне набора")
    inner = own.groupby("SYNTHETIC_F")["SYNTHETIC_U"].nunique()
    print(f"Физлиц, связывающих 2+ компании из набора: {(inner > 1).sum()}")


def payment_graph(c: pd.DataFrame, all_c: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    section("8. Платёжный граф (контрагенты)")
    p = p[p["PAYER"] != p["BENEFICIARY"]]
    ids = set(c["SYNTHETIC_U"])

    outgoing = p[p["PAYER"].isin(ids)].rename(columns={"PAYER": "U", "BENEFICIARY": "CP"})
    incoming = p[p["BENEFICIARY"].isin(ids)].rename(columns={"BENEFICIARY": "U", "PAYER": "CP"})
    print(f"Рёбер: исходящих {len(outgoing)}, входящих {len(incoming)}")

    stats = pd.DataFrame(index=c["SYNTHETIC_U"])
    stats["suppliers"] = outgoing.groupby("U")["CP"].nunique()
    stats["customers"] = incoming.groupby("U")["CP"].nunique()
    stats["out_bn"] = outgoing.groupby("U")["PAY_AMT"].sum() / BN
    stats["in_bn"] = incoming.groupby("U")["PAY_AMT"].sum() / BN
    stats = stats.fillna(0)
    print(f"Компаний без платежей вообще: {((stats['suppliers'] + stats['customers']) == 0).sum()}")
    print("\nКонтрагентов на компанию:")
    print(stats[["suppliers", "customers"]].describe(percentiles=[.25, .5, .75]).round(1).to_string())

    # Концентрация: доля топ-5 поставщиков/покупателей в объёме платежей.
    def top5_share(edges: pd.DataFrame) -> pd.Series:
        def share(g: pd.DataFrame) -> float:
            return g["PAY_AMT"].nlargest(5).sum() / g["PAY_AMT"].sum()
        return edges.groupby("U")[["PAY_AMT"]].apply(share)

    stats["top5_supplier_share"] = top5_share(outgoing)
    stats["top5_customer_share"] = top5_share(incoming)
    print("\nДоля топ-5 контрагентов в объёме платежей (медиана):")
    print(stats[["top5_supplier_share", "top5_customer_share"]].median().round(2).to_string())

    known = set(all_c["SYNTHETIC_U"])
    cps = pd.concat([outgoing["CP"], incoming["CP"]]).drop_duplicates()
    print(f"\nУникальных контрагентов: {len(cps)}, из них есть в датасете компаний: "
          f"{cps.isin(known).mean() * 100:.1f}%")
    inner = outgoing[outgoing["CP"].isin(ids)]
    print(f"Платежей внутри набора (крупная -> крупная): {len(inner)}")

    okved = all_c.set_index("SYNTHETIC_U")["MAINOKVED_NAME"]
    print("\n-- Отрасли поставщиков (по объёму платежей, млрд ₽)")
    sup = outgoing.assign(OKVED=outgoing["CP"].map(okved)).dropna(subset=["OKVED"])
    print((sup.groupby("OKVED")["PAY_AMT"].sum() / BN).nlargest(15).round(1).to_string())
    print("\n-- Отрасли покупателей (по объёму платежей, млрд ₽)")
    cus = incoming.assign(OKVED=incoming["CP"].map(okved)).dropna(subset=["OKVED"])
    print((cus.groupby("OKVED")["PAY_AMT"].sum() / BN).nlargest(15).round(1).to_string())
    return stats


def demo_candidates(c: pd.DataFrame, stats: pd.DataFrame) -> None:
    section("9. Насколько богат профиль (кандидаты для демо из target)")
    c = c.set_index("SYNTHETIC_U").join(stats)
    # Чем больше признаков заполнено, тем больше поводов для персонализации.
    c["richness"] = (
        c[list(PRODUCT_COLUMNS)].sum(axis=1)
        + c["FTS_FLAG"] * 2
        + (c["suppliers"] > 0).astype(int)
        + (c["customers"] > 0).astype(int)
    )
    cols = ["SEGMENT", "MAINOKVED_NAME", "REVENUE_COMPANY", "CL_TYPE", "REGION",
            "FTS_COUNTRY_NAME", "suppliers", "customers", "richness"]
    top = c[c["SEGMENT"] == "target (>=15)"].sort_values(
        ["richness", "suppliers"], ascending=False
    )[cols].head(20)
    top["REVENUE_COMPANY"] = (top["REVENUE_COMPANY"] / BN).round(1)
    print(top.to_string())


def main() -> None:
    companies, all_companies, links, payments = load()
    overview(companies)
    profile(companies)
    industries(companies)
    geography(companies)
    products(companies)
    groups(companies, all_companies)
    related_people(companies, links)
    stats = payment_graph(companies, all_companies, payments)
    demo_candidates(companies, stats)


if __name__ == "__main__":
    main()
