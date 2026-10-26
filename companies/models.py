"""Данные кейса о компаниях-клиентах: компании, их платежи и связи с физлицами.

Загружаются командой `python manage.py import_data` из applications/csv
и applications/filtered_data (профили собирает scripts/build_profiles.py).
"""
from django.db import models


class Company(models.Model):
    """Юрлицо из df_Uid_data_hack.csv. Загружаются все компании: мелкие нужны как контрагенты."""

    company_id = models.CharField("ID компании (SYNTHETIC_U)", max_length=32, primary_key=True)

    # Принадлежность к наборам.
    in_working_set = models.BooleanField("рабочий набор (выручка ≥ 10 млрд ₽)", default=False, db_index=True)
    is_target = models.BooleanField("целевая аудитория (выручка ≥ 15 млрд ₽)", default=False, db_index=True)

    # Масштаб.
    revenue = models.BigIntegerField("годовая выручка, ₽", db_index=True)
    employees_fns = models.CharField("сотрудники по ФНС (диапазон)", max_length=32, blank=True)
    employees_spark = models.CharField("сотрудники по СПАРК (диапазон)", max_length=32, blank=True)

    # Отрасль.
    okved_code = models.CharField("код ОКВЭД", max_length=16, db_index=True)
    okved_name = models.CharField("вид деятельности по ОКВЭД", max_length=512)
    sector = models.CharField("сектор (из ОКВЭД)", max_length=128, blank=True, db_index=True)

    # Юридические характеристики.
    org_type = models.CharField("тип организации", max_length=32, blank=True)
    ownership = models.CharField("форма собственности (ОКФС)", max_length=128, blank=True)
    age_years = models.CharField("возраст компании, лет (диапазон)", max_length=16, blank=True)
    development_stage = models.CharField("стадия развития (метка банка)", max_length=64, blank=True)
    negative_mark = models.BooleanField("NEGATIVEMARK (смысл не подтверждён)", default=False)

    # География.
    city = models.CharField("город регистрации", max_length=128, blank=True)
    region = models.CharField("регион регистрации", max_length=128, blank=True, db_index=True)

    # ВЭД.
    has_foreign_trade = models.BooleanField("есть в базе ФТС", default=False)
    trade_countries = models.CharField("страны ВЭД (как в ФТС)", max_length=512, blank=True)

    # Отношения с банком.
    client_type = models.CharField("статус (CLIENT / PROSPECT / CLOSE_CLIENT)", max_length=16, db_index=True)
    has_rko = models.BooleanField("РКО", default=False)
    has_guarantees = models.BooleanField("банковские гарантии", default=False)
    has_loans = models.BooleanField("кредиты", default=False)
    has_leasing = models.BooleanField("лизинг", default=False)
    has_deposits = models.BooleanField("депозиты", default=False)
    has_payroll = models.BooleanField("зарплатный проект", default=False)
    has_factoring = models.BooleanField("факторинг", default=False)

    group_id = models.CharField("ID группы компаний", max_length=64, blank=True, db_index=True)

    # Профиль для LLM есть только у компаний рабочего набора.
    summary = models.TextField("краткое описание для LLM", blank=True)
    brief = models.JSONField("короткий профиль для промпта", null=True, blank=True)
    profile = models.JSONField("полный профиль", null=True, blank=True)

    class Meta:
        verbose_name = "компания"
        verbose_name_plural = "компании"
        ordering = ["-revenue"]

    def __str__(self) -> str:
        return f"{self.company_id} — {self.okved_name[:60]}"


class CompanySensitivity(models.Model):
    """Тег чувствительности компании (ставка, юань, санкции…) — основа для подбора новостей."""

    STRENGTH_CHOICES = [("высокая", "высокая"), ("средняя", "средняя")]

    company = models.ForeignKey(Company, on_delete=models.CASCADE, related_name="sensitivities")
    code = models.CharField("код темы", max_length=64, db_index=True)
    label = models.CharField("тема", max_length=128)
    strength = models.CharField("сила", max_length=16, choices=STRENGTH_CHOICES)
    reasons = models.JSONField("почему (объяснения)", default=list)
    source_fields = models.JSONField("из каких полей выведено", default=list)
    news_keywords = models.JSONField("ключевые слова для поиска новостей", default=list)

    class Meta:
        verbose_name = "чувствительность компании"
        verbose_name_plural = "чувствительности компаний"
        constraints = [models.UniqueConstraint(fields=["company", "code"], name="uniq_company_sensitivity")]

    def __str__(self) -> str:
        return f"{self.company_id}: {self.label} ({self.strength})"


class Payment(models.Model):
    """Агрегированные платежи между компаниями (df_Uid_payment_hack.csv).

    ID хранятся строкой, а не внешним ключом: около трети участников нет в файле компаний.
    Загружены как есть, включая ~27 тыс. платежей самой себе (payer == beneficiary):
    при анализе контрагентов их нужно исключать — .exclude(payer_id=F("beneficiary_id")).
    """

    payer_id = models.CharField("плательщик", max_length=32, db_index=True)
    beneficiary_id = models.CharField("получатель", max_length=32, db_index=True)
    amount = models.BigIntegerField("объём платежей, ₽")
    count = models.IntegerField("количество платежей")

    class Meta:
        verbose_name = "платёж"
        verbose_name_plural = "платежи"


class PersonLink(models.Model):
    """Связь обезличенного физлица с компанией (df_Fid_Uid_link_hack.csv)."""

    person_id = models.CharField("ID физлица (SYNTHETIC_F)", max_length=32, db_index=True)
    company_id = models.CharField("ID компании", max_length=32, db_index=True)
    city = models.CharField("город проживания", max_length=128, blank=True)
    region = models.CharField("регион проживания", max_length=128, blank=True)

    class Meta:
        verbose_name = "связь с физлицом"
        verbose_name_plural = "связи с физлицами"
