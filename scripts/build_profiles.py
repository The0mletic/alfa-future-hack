"""Сборка профилей компаний в понятном для LLM виде.

Запуск из корня репозитория (после scripts/clean_data.py):
    python scripts/build_profiles.py

Результат в applications/filtered_data:
    company_profiles.jsonl / company_profiles_target.jsonl — по одному JSON-профилю на строку,
        их отдаёт инструмент агента get_company_profile;
    company_profiles.csv / company_profiles_target.csv — плоская таблица для людей и отладки.

Все признаки считаются детерминированно из исходных CSV. У каждого вывода
(тега чувствительности, возможности продаж) есть поле reasons/source_fields —
из каких данных он получен.
"""
import json
from pathlib import Path

import pandas as pd

import reference_data as ref

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "applications" / "csv"
FILTERED_DIR = ROOT / "applications" / "filtered_data"

SEP = ";"
TARGET_REVENUE_RUB = 15_000_000_000
BN = 1_000_000_000
TOP_COUNTERPARTIES = 5

# Пороги концентрации платежей на топ-5 контрагентах.
CONCENTRATION_LEVELS = [(0.8, "очень высокая"), (0.5, "высокая"), (0.3, "умеренная"), (0.0, "низкая")]
REVENUE_BANDS = [(100, "более 100 млрд ₽"), (50, "50–100 млрд ₽"), (25, "25–50 млрд ₽"),
                 (15, "15–25 млрд ₽"), (10, "10–15 млрд ₽"), (0, "менее 10 млрд ₽")]


# --- Форматирование -----------------------------------------------------------------

def fmt_rub(amount: float) -> str:
    if amount >= BN:
        return f"{amount / BN:.1f} млрд ₽".replace(".", ",")
    if amount >= 1_000_000:
        return f"{amount / 1_000_000:.1f} млн ₽".replace(".", ",")
    return f"{amount / 1000:.0f} тыс. ₽"


def fmt_share(share: float) -> str:
    return f"{share * 100:.0f}%"


def level(value: float, levels: list[tuple[float, str]]) -> str:
    return next(label for threshold, label in levels if value >= threshold)


def clean(value):
    """NaN -> None, numpy-типы -> обычные Python-типы, чтобы JSON был чистым."""
    if isinstance(value, float) and pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def flag(value) -> bool:
    return bool(value == 1)


def lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]


def upper_first(text: str) -> str:
    return text[:1].upper() + text[1:]


def country_name(raw: str) -> str:
    """'ЮЖНАЯ КОРЕЯ' -> 'Южная Корея', аббревиатуры (ОАЭ, США) оставляем как есть."""
    if raw in ("ОАЭ", "США"):
        return raw
    return " ".join(w.capitalize() for w in raw.split())


# --- Загрузка -------------------------------------------------------------------------

def load():
    companies = pd.read_csv(FILTERED_DIR / "df_Uid_data_hack.csv", sep=SEP, dtype={"MAINOKVED_CODE": str})
    all_companies = (
        pd.read_csv(SRC_DIR / "df_Uid_data_hack.csv", sep=SEP, dtype={"MAINOKVED_CODE": str}, low_memory=False)
        .drop_duplicates("SYNTHETIC_U")
        .set_index("SYNTHETIC_U")
    )
    links = pd.read_csv(SRC_DIR / "df_Fid_Uid_link_hack.csv", sep=SEP)
    payments = pd.read_csv(SRC_DIR / "df_Uid_payment_hack.csv", sep=SEP)
    payments = payments[payments["PAYER"] != payments["BENEFICIARY"]]
    return companies, all_companies, links, payments


# --- Блоки профиля ------------------------------------------------------------------

def build_scale(row, rank: int, total: int) -> dict:
    revenue = row["REVENUE_COMPANY"]
    workers_raw = row["AMOUNT_WORKERS_FNS_GRADATION"]
    workers_label, workers_max = ref.EMPLOYEE_LABELS.get(workers_raw, ("нет данных", None))
    revenue_per_employee_min = revenue / workers_max if workers_max else None

    anomaly = None
    if workers_raw == "Менее 15":
        anomaly = ("Очень высокая выручка при штате менее 15 человек: вероятно, торговая или агентская "
                   "модель, либо компания — часть холдинга, а сотрудники оформлены в других юрлицах.")
    elif workers_raw == "16-100":
        anomaly = ("Высокая выручка при небольшом штате (до 100 человек): вероятно, торговая модель "
                   "с низкой долей собственного производства.")

    return {
        "revenue_rub": revenue,
        "revenue_label": fmt_rub(revenue),
        "revenue_band": level(revenue / BN, REVENUE_BANDS),
        "revenue_rank_in_dataset": f"{rank} из {total} по выручке",
        "is_target_audience": revenue >= TARGET_REVENUE_RUB,
        "target_audience_note": (
            "крупный бизнес по определению кейса (выручка свыше 15 млрд ₽)"
            if revenue >= TARGET_REVENUE_RUB
            else "близко к крупному бизнесу (10–15 млрд ₽), вне целевой аудитории кейса"
        ),
        "employees_label": workers_label,
        "employees_raw": workers_raw,
        "is_large_employer": workers_raw == "Более 251",
        "revenue_per_employee_min_label": fmt_rub(revenue_per_employee_min) if revenue_per_employee_min else None,
        "staff_anomaly": anomaly,
    }


def build_industry(row) -> dict:
    code = str(row["MAINOKVED_CODE"])
    sector, model = ref.okved_section(code)
    section = int(code.split(".")[0]) if code.split(".")[0].isdigit() else None
    return {
        "okved_code": code,
        "okved_name": row["MAINOKVED_NAME"],
        "okved_section": code.split(".")[0],
        "sector": sector,
        "business_model": model,
        "is_capital_intensive": section in ref.CAPITAL_INTENSIVE_SECTIONS,
        "commodity_exposure": ref.match_prefix(code, ref.COMMODITY_RULES),
        "demand_drivers": ref.match_prefix(code, ref.DEMAND_RULES),
        "regulation": ref.match_prefix(code, ref.REGULATION_RULES),
        "note": "ОКВЭД отражает только основной вид деятельности; реальный бизнес может быть шире.",
    }


def build_legal(row) -> dict:
    ownership_raw = row["OKFS_NAME"]
    ownership, foreign, state = ref.OWNERSHIP.get(ownership_raw, (str(ownership_raw).lower(), False, False))
    age_raw = row["COMPANY_AGE_YEARS_GRADATION"]
    age_label, age_meaning = ref.AGE_LABELS.get(age_raw, ("нет данных", None))
    stage = clean(row["ECO_MARK_LABEL"])
    return {
        "ownership": ownership,
        "ownership_raw": ownership_raw,
        "has_foreign_capital": foreign,
        "has_state_capital": state,
        "age_label": age_label,
        "age_meaning": age_meaning,
        "development_stage": stage,
        "development_stage_meaning": ref.ECO_STAGE_MEANINGS.get(stage) if stage else None,
        "development_stage_note": "внутренняя метка банка, трактовка предположительная",
        "key_owner_flag_raw": clean(row["NEGATIVEMARK"]),
        "key_owner_flag_note": "NEGATIVEMARK: смысл не подтверждён организаторами, не использовать в выводах",
    }


def build_geography(row) -> dict:
    region = row["REGION"]
    district = ref.REGION_TO_DISTRICT.get(region)
    return {
        "city": row["CITY"],
        "region": region,
        "federal_district": f"{district} федеральный округ" if district else "нет данных",
        "is_capital": region in ("Москва", "Санкт-Петербург"),
        "note": "регион регистрации может не совпадать с регионом реальной деятельности",
    }


def build_foreign_trade(row) -> dict:
    is_international = flag(row["FTS_FLAG"])
    raw = clean(row["FTS_COUNTRY_NAME"])
    countries = [c.strip() for c in raw.split(",")] if raw else []
    known = [c for c in countries if c in ref.COUNTRIES]

    groups = list(dict.fromkeys(ref.COUNTRIES[c][0] for c in known))
    currencies = list(dict.fromkeys(ref.COUNTRIES[c][1] for c in known if ref.COUNTRIES[c][1] != "RUB"))
    unfriendly = [c for c in known if ref.COUNTRIES[c][2]]
    hubs = [c for c in known if ref.COUNTRIES[c][3]]

    return {
        "is_international": is_international,
        "countries": [country_name(c) for c in countries],
        "countries_raw": raw,
        "countries_count": len(countries),
        "country_groups": groups,
        "likely_currencies": [
            {"code": cur, "name": ref.CURRENCY_NAMES.get(cur, cur)} for cur in currencies
        ],
        "trades_with_china": "КИТАЙ" in countries or "ГОНКОНГ" in countries,
        "unfriendly_countries": [country_name(c) for c in unfriendly],
        "transit_hub_countries": [country_name(c) for c in hubs],
        "unknown_countries": [country_name(c) for c in countries if c not in ref.COUNTRIES],
        "note": (
            "Страны из таможенной базы ФТС; направление (импорт или экспорт) и валюта расчётов "
            "в данных не указаны — валюты выведены по странам предположительно."
            if is_international else "Компания не найдена в таможенной базе: внешнеторговой деятельности не видно."
        ),
    }


def build_bank(row, scale: dict, industry: dict, trade: dict, cps: dict) -> dict:
    status_raw = row["CL_TYPE"]
    status, is_active = ref.CLIENT_STATUS.get(status_raw, (status_raw, False))

    products, active, missing = {}, [], []
    for col, (key, name, meaning) in ref.PRODUCTS.items():
        has = flag(row[col])
        products[key] = {"has": has, "name": name, "meaning": meaning}
        (active if has else missing).append(name)

    has = {key: p["has"] for key, p in products.items()}
    debt = [products[k]["name"] for k in ("loans", "leasing") if has[k]]

    # Возможности продаж: простые правила, у каждого есть причина.
    opportunities = []
    if not has["rko"]:
        opportunities.append({"product": "РКО", "reason": "у компании нет расчётного счёта в банке"})
    if not has["factoring"] and cps["customers_count"] >= 10 and "B2C" not in industry["business_model"]:
        opportunities.append({"product": "Факторинг",
                              "reason": f"B2B-модель и {cps['customers_count']} покупателей: можно ускорить "
                                        "получение денег по отсрочкам"})
    if not has["guarantees"] and industry["okved_section"] in ("41", "42", "43", "71"):
        opportunities.append({"product": "Банковские гарантии",
                              "reason": "строительная отрасль: гарантии нужны для тендеров и контрактов"})
    if not has["leasing"] and industry["is_capital_intensive"]:
        opportunities.append({"product": "Лизинг",
                              "reason": "капиталоёмкая отрасль: техника и оборудование"})
    if not has["deposits"] and cps["incoming_rub"] >= BN:
        opportunities.append({"product": "Депозиты и управление ликвидностью",
                              "reason": f"входящие платежи {fmt_rub(cps['incoming_rub'])}: "
                                        "есть потенциал размещать свободные остатки"})
    if not has["payroll"] and row["AMOUNT_WORKERS_FNS_GRADATION"] in ("101-250", "Более 251"):
        opportunities.append({"product": "Зарплатный проект",
                              "reason": f"в компании {scale['employees_label']}"})
    if trade["is_international"]:
        opportunities.append({"product": "ВЭД-сервисы: валютный контроль, конверсия, хеджирование",
                              "reason": f"страны ВЭД: {', '.join(trade['countries'])}; "
                                        "использование этих услуг в данных не видно"})

    return {
        "client_status_raw": status_raw,
        "client_status": status,
        "is_active_client": is_active,
        "products": products,
        "active_products": active,
        "missing_products": missing,
        "products_count": len(active),
        "uses_debt_financing": bool(debt),
        "debt_instruments": debt,
        "has_free_cash": has["deposits"],
        "likely_participates_in_tenders": has["guarantees"],
        "cross_sell_opportunities": opportunities,
        "note": "Видны только продукты Альфа-Банка; продукты других банков в данных отсутствуют.",
    }


def build_group(row, all_companies: pd.DataFrame, group_stats: dict, working_ids: set) -> dict:
    gid = row["GROUP_ID"]
    stats = group_stats.get(gid)
    size = stats["size"] if stats else 1
    members_in_set = [m for m in (stats["members"] if stats else []) if m in working_ids and m != row["SYNTHETIC_U"]]
    return {
        "group_id": gid,
        "group_size": size,
        "is_holding_member": size > 1,
        "group_size_label": (
            "самостоятельная компания" if size == 1
            else f"входит в группу из {size} компаний" + (" (крупный холдинг)" if size > 100 else "")
        ),
        "group_total_revenue_label": fmt_rub(stats["revenue"]) if stats else fmt_rub(row["REVENUE_COMPANY"]),
        "group_top_sectors": stats["sectors"] if stats else [],
        "group_members_in_working_set": members_in_set,
    }


def counterparty_card(cp_id: str, amount: float, count: int, total: float,
                      all_companies: pd.DataFrame, own_group: str, working_ids: set) -> dict:
    card = {
        "company_id": cp_id,
        "amount_label": fmt_rub(amount),
        "payments_count": int(count),
        "share": round(amount / total, 3) if total else 0,
        "share_label": fmt_share(amount / total) if total else "0%",
        "in_dataset": cp_id in all_companies.index,
        "in_working_set": cp_id in working_ids,
    }
    if card["in_dataset"]:
        info = all_companies.loc[cp_id]
        sector, _ = ref.okved_section(str(info["MAINOKVED_CODE"]))
        card.update({
            "sector": sector,
            "okved_name": info["MAINOKVED_NAME"],
            "region": info["REGION"],
            "revenue_label": fmt_rub(info["REVENUE_COMPANY"]),
            "same_group": info["GROUP_ID"] == own_group,
        })
    else:
        card["sector"] = "нет данных (компании нет в датасете)"
    return card


def sector_shares(edges: pd.DataFrame, all_companies: pd.DataFrame) -> list[dict]:
    total = edges["PAY_AMT"].sum()
    if not total:
        return []
    sectors = edges["CP"].map(all_companies["SECTOR"]).fillna("нет данных")
    shares = edges.groupby(sectors)["PAY_AMT"].sum().sort_values(ascending=False) / total
    shares = shares[shares >= 0.01]
    return [{"sector": s, "share": round(v, 3), "share_label": fmt_share(v)} for s, v in shares.head(6).items()]


def build_counterparties(uid: str, own_group: str, outgoing: pd.DataFrame, incoming: pd.DataFrame,
                         all_companies: pd.DataFrame, working_ids: set) -> dict:
    out = outgoing[outgoing["U"] == uid]
    inc = incoming[incoming["U"] == uid]

    def side(edges: pd.DataFrame, role: str) -> dict:
        total = edges["PAY_AMT"].sum()
        by_cp = edges.groupby("CP").agg(amount=("PAY_AMT", "sum"), count=("PAY_CNT", "sum"))
        by_cp = by_cp.sort_values("amount", ascending=False)
        top5 = by_cp.head(TOP_COUNTERPARTIES)["amount"].sum() / total if total else 0
        top1 = by_cp["amount"].iloc[0] / total if total else 0
        same_group = edges["CP"].map(all_companies["GROUP_ID"]) == own_group
        return {
            f"{role}_count": len(by_cp),
            f"{role}_volume_rub": float(total),
            f"{role}_volume_label": fmt_rub(total) if total else "нет платежей",
            f"{role}_payments_count": int(edges["PAY_CNT"].sum()),
            f"top1_{role}_share": round(top1, 3),
            f"top5_{role}_share": round(top5, 3),
            f"{role}_concentration": level(top5, CONCENTRATION_LEVELS) if total else "нет данных",
            f"intra_group_{role}_share": round(edges.loc[same_group, "PAY_AMT"].sum() / total, 3) if total else 0,
            f"top_{role}": [
                counterparty_card(cp, r["amount"], r["count"], total, all_companies, own_group, working_ids)
                for cp, r in by_cp.head(TOP_COUNTERPARTIES).iterrows()
            ],
            f"{role}_sectors": sector_shares(edges, all_companies),
        }

    result = {"has_payment_data": len(out) + len(inc) > 0}
    result.update(side(out, "suppliers"))
    result.update(side(inc, "customers"))
    # Совместимые с остальным кодом короткие имена.
    result["incoming_rub"] = result["customers_volume_rub"]
    balance = result["customers_volume_rub"] - result["suppliers_volume_rub"]
    result["net_flow_label"] = (
        f"входящих платежей больше на {fmt_rub(balance)}" if balance > 0
        else f"исходящих платежей больше на {fmt_rub(-balance)}" if balance < 0 else "нет данных"
    )
    result["note"] = ("Платежи агрегированы за период датасета (период не указан). Поставщики — кому компания "
                      "платит, покупатели — кто платит компании. Видны только платежи, прошедшие через банк.")
    return result


def build_people(uid: str, links: pd.DataFrame) -> dict:
    own = links[links["SYNTHETIC_U"] == uid]
    people = set(own["SYNTHETIC_F"])
    other = links[links["SYNTHETIC_F"].isin(people) & (links["SYNTHETIC_U"] != uid)]
    return {
        "linked_individuals_count": len(people),
        "companies_linked_via_individuals": sorted(other["SYNTHETIC_U"].unique().tolist()),
        "note": "Связи с физлицами есть лишь у малой части компаний; отсутствие данных не означает отсутствия связей.",
    }


def add_tag(tags: dict, code: str, strength: str, reason: str, fields: list[str], keywords: list[str] = ()):
    label, base_keywords = ref.SENSITIVITY_TAGS[code]
    tag = tags.setdefault(code, {"code": code, "label": label, "strength": strength,
                                 "reasons": [], "source_fields": [], "news_keywords": list(base_keywords)})
    if strength == "высокая":
        tag["strength"] = "высокая"
    if reason not in tag["reasons"]:
        tag["reasons"].append(reason)
    for f in fields:
        if f not in tag["source_fields"]:
            tag["source_fields"].append(f)
    for k in keywords:
        if k not in tag["news_keywords"]:
            tag["news_keywords"].append(k)


def build_sensitivities(row, scale, industry, legal, trade, bank, group, cps) -> list[dict]:
    tags: dict = {}
    has = {k: p["has"] for k, p in bank["products"].items()}

    if has["loans"] or has["leasing"]:
        add_tag(tags, "key_rate", "высокая",
                f"есть заёмное финансирование: {', '.join(bank['debt_instruments']).lower()}",
                ["CL_PRODUCT_LOA_MARK", "CL_PRODUCT_LEA_MARK"])
    if has["deposits"]:
        add_tag(tags, "deposit_yield", "средняя", "есть депозиты в банке", ["CL_PRODUCT_DEP_MARK"])

    if trade["is_international"]:
        currencies = [c["name"] for c in trade["likely_currencies"]]
        add_tag(tags, "fx", "высокая" if trade["countries_count"] >= 2 else "средняя",
                f"ведёт ВЭД со странами: {', '.join(trade['countries'])}",
                ["FTS_FLAG", "FTS_COUNTRY_NAME"], [f"курс: {c}" for c in currencies])
    if trade["trades_with_china"]:
        add_tag(tags, "cny", "высокая", "в списке стран ВЭД есть Китай или Гонконг", ["FTS_COUNTRY_NAME"])
    if trade["unfriendly_countries"]:
        add_tag(tags, "sanctions", "высокая",
                f"торговля с недружественными странами: {', '.join(trade['unfriendly_countries'])}",
                ["FTS_COUNTRY_NAME"])
    if trade["transit_hub_countries"]:
        add_tag(tags, "sanctions", "средняя",
                f"торговля через страны-хабы для импорта и платежей: {', '.join(trade['transit_hub_countries'])}",
                ["FTS_COUNTRY_NAME"])
    if legal["has_foreign_capital"]:
        add_tag(tags, "foreign_owner", "средняя", f"форма собственности: {legal['ownership']}", ["OKFS_NAME"])
        add_tag(tags, "sanctions", "средняя", "иностранный капитал в собственности", ["OKFS_NAME"])

    code = industry["okved_code"]
    for tag in ref.match_prefix(code, ref.SECTOR_TAG_RULES):
        add_tag(tags, tag, "высокая", f"основной вид деятельности: {industry['okved_name']}", ["MAINOKVED_CODE"],
                industry["demand_drivers"])
    if industry["commodity_exposure"]:
        add_tag(tags, "commodity", "высокая",
                f"отрасль зависит от цен на: {', '.join(industry['commodity_exposure'])}",
                ["MAINOKVED_CODE"], [f"цены: {c}" for c in industry["commodity_exposure"]])
    if industry["regulation"]:
        add_tag(tags, "regulation", "средняя", "; ".join(industry["regulation"]), ["MAINOKVED_CODE"],
                industry["regulation"])
    if has["guarantees"]:
        add_tag(tags, "gov_contracts", "средняя", "есть банковские гарантии: вероятно участие в тендерах",
                ["CL_PRODUCT_GAR_MARK"])

    if scale["is_large_employer"]:
        add_tag(tags, "labor_market", "высокая", "более 250 сотрудников", ["AMOUNT_WORKERS_FNS_GRADATION"])
    elif row["AMOUNT_WORKERS_FNS_GRADATION"] == "101-250" and has["payroll"]:
        add_tag(tags, "labor_market", "средняя", "101–250 сотрудников и зарплатный проект",
                ["AMOUNT_WORKERS_FNS_GRADATION", "CL_PRODUCT_ZPP_MARK"])

    if cps["has_payment_data"]:
        if cps["top5_suppliers_share"] >= 0.8 and cps["suppliers_count"] >= 5:
            add_tag(tags, "supplier_concentration", "высокая",
                    f"топ-5 поставщиков дают {fmt_share(cps['top5_suppliers_share'])} исходящих платежей",
                    ["df_Uid_payment_hack.csv"])
        if cps["top5_customers_share"] >= 0.8 and cps["customers_count"] >= 5:
            add_tag(tags, "customer_concentration", "высокая",
                    f"топ-5 покупателей дают {fmt_share(cps['top5_customers_share'])} входящих платежей",
                    ["df_Uid_payment_hack.csv"])
        # Чувствительности, унаследованные от отраслей ключевых поставщиков.
        for s in cps["suppliers_sectors"]:
            if s["share"] < 0.2 or s["sector"].startswith("нет данных"):
                continue
            sector = s["sector"]
            reason = f"{s['share_label']} закупок приходится на поставщиков из сектора «{sector}»"
            if sector == "Строительство":
                add_tag(tags, "construction", "средняя", reason, ["df_Uid_payment_hack.csv"])
            elif sector == "Транспорт и логистика":
                add_tag(tags, "logistics", "средняя", reason, ["df_Uid_payment_hack.csv"])
            elif sector in ("Энергетика",):
                add_tag(tags, "energy_tariffs", "средняя", reason, ["df_Uid_payment_hack.csv"])
            elif sector in ("Добыча полезных ископаемых", "Металлургия и металлоизделия",
                            "Химия, нефтепереработка, фармацевтика и пластмассы"):
                add_tag(tags, "commodity", "средняя", reason, ["df_Uid_payment_hack.csv"])
            elif sector == "IT, связь и медиа":
                add_tag(tags, "it_budgets", "средняя", reason, ["df_Uid_payment_hack.csv"])

    if group["group_size"] > 100:
        add_tag(tags, "holding", "средняя", group["group_size_label"], ["GROUP_ID"])

    order = {"высокая": 0, "средняя": 1}
    return sorted(tags.values(), key=lambda t: (order[t["strength"]], t["code"]))


def build_data_quality(row, trade, cps, people) -> dict:
    missing = [c for c in ("ECO_MARK_LABEL", "FTS_COUNTRY_NAME") if pd.isna(row[c])]
    notes = []
    if not cps["has_payment_data"]:
        notes.append("нет данных о платежах: контрагенты неизвестны")
    if trade["unknown_countries"]:
        notes.append(f"страны без справочных данных: {', '.join(trade['unknown_countries'])}")
    if not people["linked_individuals_count"]:
        notes.append("нет данных о связанных физлицах")
    return {
        "missing_fields": missing,
        "ignored_fields": {
            "AMOUNT_WORKERS_SPARK_GRADATION": "пусто почти у всех компаний",
            "ORG_TYPE": "у всех компаний одно значение (коммерческая организация)",
            "NEGATIVEMARK": "смысл не подтверждён",
        },
        "notes": notes,
    }


def build_summary(p: dict) -> str:
    s, ind, geo, bank, trade, cps = (p["scale"], p["industry"], p["geography"],
                                      p["bank_relationship"], p["foreign_trade"], p["counterparties"])
    parts = [
        f"{ind['sector']}: {lower_first(ind['okved_name'])}.",
        f"Выручка {s['revenue_label']}, {s['employees_label']}.",
        f"{geo['region']} ({geo['federal_district']}).",
        f"{upper_first(bank['client_status'])}.",
        f"Продукты банка: {', '.join(lower_first(x) for x in bank['active_products'])}." if bank["active_products"]
        else "Продуктов Альфа-Банка нет.",
    ]
    if trade["is_international"]:
        parts.append(f"ВЭД: {', '.join(trade['countries'])}.")
    else:
        parts.append("Внешнеторговой деятельности не видно.")
    if cps["has_payment_data"]:
        parts.append(
            f"Поставщиков {cps['suppliers_count']}, покупателей {cps['customers_count']}; "
            f"на топ-5 поставщиков приходится {fmt_share(cps['top5_suppliers_share'])} закупок."
        )
    if p["group"]["is_holding_member"]:
        parts.append(f"{upper_first(p['group']['group_size_label'])}.")
    high = [lower_first(t["label"]) for t in p["sensitivities"] if t["strength"] == "высокая"]
    if high:
        parts.append(f"Главные чувствительности: {', '.join(high)}.")
    return " ".join(parts)


# --- Сборка ---------------------------------------------------------------------------

def group_statistics(all_companies: pd.DataFrame) -> dict:
    stats = {}
    for gid, g in all_companies.groupby("GROUP_ID"):
        if len(g) < 2:
            continue
        sectors = g["SECTOR"].value_counts().head(3)
        stats[gid] = {
            "size": len(g),
            "revenue": g["REVENUE_COMPANY"].sum(),
            "sectors": [f"{s} ({n})" for s, n in sectors.items()],
            "members": g.index.tolist(),
        }
    return stats


def build_profiles() -> list[dict]:
    companies, all_companies, links, payments = load()
    all_companies["SECTOR"] = all_companies["MAINOKVED_CODE"].astype(str).map(lambda c: ref.okved_section(c)[0])
    working_ids = set(companies["SYNTHETIC_U"])
    group_stats = group_statistics(all_companies)

    outgoing = payments[payments["PAYER"].isin(working_ids)].rename(columns={"PAYER": "U", "BENEFICIARY": "CP"})
    incoming = payments[payments["BENEFICIARY"].isin(working_ids)].rename(columns={"BENEFICIARY": "U", "PAYER": "CP"})
    links = links[links["SYNTHETIC_F"].isin(links.loc[links["SYNTHETIC_U"].isin(working_ids), "SYNTHETIC_F"])]

    companies = companies.sort_values("REVENUE_COMPANY", ascending=False).reset_index(drop=True)
    profiles = []
    for rank, row in companies.iterrows():
        uid = row["SYNTHETIC_U"]
        scale = build_scale(row, rank + 1, len(companies))
        industry = build_industry(row)
        legal = build_legal(row)
        geography = build_geography(row)
        trade = build_foreign_trade(row)
        cps = build_counterparties(uid, row["GROUP_ID"], outgoing, incoming, all_companies, working_ids)
        bank = build_bank(row, scale, industry, trade, cps)
        group = build_group(row, all_companies, group_stats, working_ids)
        people = build_people(uid, links)
        profile = {
            "company_id": uid,
            "scale": scale,
            "industry": industry,
            "legal": legal,
            "geography": geography,
            "foreign_trade": trade,
            "bank_relationship": bank,
            "group": group,
            "counterparties": cps,
            "related_individuals": people,
        }
        profile["sensitivities"] = build_sensitivities(row, scale, industry, legal, trade, bank, group, cps)
        profile["news_keywords"] = list(dict.fromkeys(k for t in profile["sensitivities"] for k in t["news_keywords"]))
        profile["data_quality"] = build_data_quality(row, trade, cps, people)
        profile["summary"] = build_summary(profile)
        profile["raw"] = {k: clean(v) for k, v in row.items()}
        profiles.append({"company_id": uid, "summary": profile.pop("summary"), **profile})
    return profiles


def flatten(p: dict) -> dict:
    s, ind, legal, geo, trade, bank, group, cps = (
        p["scale"], p["industry"], p["legal"], p["geography"], p["foreign_trade"],
        p["bank_relationship"], p["group"], p["counterparties"],
    )
    return {
        "company_id": p["company_id"],
        "is_target_audience": s["is_target_audience"],
        "summary": p["summary"],
        "revenue_label": s["revenue_label"],
        "revenue_band": s["revenue_band"],
        "employees": s["employees_label"],
        "staff_anomaly": bool(s["staff_anomaly"]),
        "sector": ind["sector"],
        "business_model": ind["business_model"],
        "okved": f"{ind['okved_code']} {ind['okved_name']}",
        "capital_intensive": ind["is_capital_intensive"],
        "commodities": ", ".join(ind["commodity_exposure"]),
        "demand_drivers": ", ".join(ind["demand_drivers"]),
        "regulation": "; ".join(ind["regulation"]),
        "ownership": legal["ownership"],
        "foreign_capital": legal["has_foreign_capital"],
        "state_capital": legal["has_state_capital"],
        "age": legal["age_label"],
        "development_stage": legal["development_stage"],
        "region": geo["region"],
        "federal_district": geo["federal_district"],
        "international": trade["is_international"],
        "countries": ", ".join(trade["countries"]),
        "country_groups": ", ".join(trade["country_groups"]),
        "currencies": ", ".join(c["code"] for c in trade["likely_currencies"]),
        "unfriendly_countries": ", ".join(trade["unfriendly_countries"]),
        "client_status": bank["client_status"],
        "active_client": bank["is_active_client"],
        **{f"has_{k}": v["has"] for k, v in bank["products"].items()},
        "products_count": bank["products_count"],
        "uses_debt_financing": bank["uses_debt_financing"],
        "cross_sell": "; ".join(o["product"] for o in bank["cross_sell_opportunities"]),
        "group_size": group["group_size"],
        "holding_member": group["is_holding_member"],
        "suppliers_count": cps["suppliers_count"],
        "customers_count": cps["customers_count"],
        "outgoing_volume": cps["suppliers_volume_label"],
        "incoming_volume": cps["customers_volume_label"],
        "top5_suppliers_share": cps["top5_suppliers_share"],
        "top5_customers_share": cps["top5_customers_share"],
        "supplier_concentration": cps["suppliers_concentration"],
        "customer_concentration": cps["customers_concentration"],
        "top_supplier_sectors": ", ".join(f"{x['sector']} {x['share_label']}" for x in cps["suppliers_sectors"][:3]),
        "top_customer_sectors": ", ".join(f"{x['sector']} {x['share_label']}" for x in cps["customers_sectors"][:3]),
        "sensitivities_high": ", ".join(t["label"] for t in p["sensitivities"] if t["strength"] == "высокая"),
        "sensitivities_medium": ", ".join(t["label"] for t in p["sensitivities"] if t["strength"] == "средняя"),
    }


def brief(p: dict) -> dict:
    """Короткая версия профиля для промпта LLM: факты и чувствительности без справочного шума."""
    s, ind, trade, bank, group, cps = (p["scale"], p["industry"], p["foreign_trade"],
                                        p["bank_relationship"], p["group"], p["counterparties"])

    def top(cards: list[dict]) -> list[str]:
        return [f"{c['company_id']} — {c['share_label']}, {c['sector']}" for c in cards[:3]]

    return {
        "company_id": p["company_id"],
        "summary": p["summary"],
        "facts": {
            "выручка": s["revenue_label"],
            "целевая_аудитория_кейса": s["is_target_audience"],
            "сотрудники": s["employees_label"],
            "отрасль": f"{ind['okved_code']} {ind['okved_name']}",
            "сектор": ind["sector"],
            "бизнес_модель": ind["business_model"],
            "зависит_от_цен_на": ind["commodity_exposure"],
            "драйверы_спроса": ind["demand_drivers"],
            "регулирование": ind["regulation"],
            "регион": f"{p['geography']['region']}, {p['geography']['federal_district']}",
            "собственность": p["legal"]["ownership"],
            "возраст": p["legal"]["age_label"],
            "статус_в_банке": bank["client_status"],
            "продукты_банка": bank["active_products"],
            "заёмное_финансирование": bank["debt_instruments"],
            "страны_вэд": trade["countries"],
            "вероятные_валюты": [c["code"] for c in trade["likely_currencies"]],
            "группа": group["group_size_label"],
            "поставщики": f"{cps['suppliers_count']}, концентрация {cps['suppliers_concentration']} "
                          f"(топ-5 = {fmt_share(cps['top5_suppliers_share'])})",
            "покупатели": f"{cps['customers_count']}, концентрация {cps['customers_concentration']} "
                          f"(топ-5 = {fmt_share(cps['top5_customers_share'])})",
            "крупнейшие_поставщики": top(cps["top_suppliers"]),
            "крупнейшие_покупатели": top(cps["top_customers"]),
            "отрасли_поставщиков": [f"{x['sector']} {x['share_label']}" for x in cps["suppliers_sectors"]],
            "отрасли_покупателей": [f"{x['sector']} {x['share_label']}" for x in cps["customers_sectors"]],
        },
        "sensitivities": [
            {"тема": t["label"], "сила": t["strength"], "почему": t["reasons"]} for t in p["sensitivities"]
        ],
        "cross_sell": [f"{o['product']}: {o['reason']}" for o in bank["cross_sell_opportunities"]],
        "news_keywords": p["news_keywords"],
        "caveats": [x for x in (s["staff_anomaly"], trade["note"] if trade["is_international"] else None) if x],
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=clean) + "\n")


def write(profiles: list[dict], suffix: str) -> None:
    write_jsonl(FILTERED_DIR / f"company_profiles{suffix}.jsonl", profiles)
    write_jsonl(FILTERED_DIR / f"company_briefs{suffix}.jsonl", [brief(p) for p in profiles])
    pd.DataFrame([flatten(p) for p in profiles]).to_csv(
        FILTERED_DIR / f"company_profiles{suffix}.csv", sep=SEP, index=False
    )
    print(f"company_profiles{suffix}: {len(profiles)} профилей")


def main() -> None:
    profiles = build_profiles()
    write(profiles, "")
    write([p for p in profiles if p["scale"]["is_target_audience"]], "_target")


if __name__ == "__main__":
    main()
