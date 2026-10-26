from django.contrib import admin

from .models import Company, CompanySensitivity, Payment, PersonLink


class SensitivityInline(admin.TabularInline):
    model = CompanySensitivity
    extra = 0
    fields = ("label", "strength", "reasons")
    readonly_fields = fields


@admin.register(Company)
class CompanyAdmin(admin.ModelAdmin):
    list_display = ("company_id", "okved_name", "sector", "revenue", "region", "client_type", "is_target")
    list_filter = ("is_target", "in_working_set", "client_type", "sector")
    search_fields = ("company_id", "okved_name", "okved_code", "region")
    inlines = [SensitivityInline]


@admin.register(CompanySensitivity)
class CompanySensitivityAdmin(admin.ModelAdmin):
    list_display = ("company", "label", "strength")
    list_filter = ("code", "strength")


@admin.register(Payment)
class PaymentAdmin(admin.ModelAdmin):
    list_display = ("payer_id", "beneficiary_id", "amount", "count")
    search_fields = ("payer_id", "beneficiary_id")


@admin.register(PersonLink)
class PersonLinkAdmin(admin.ModelAdmin):
    list_display = ("person_id", "company_id", "city", "region")
    search_fields = ("person_id", "company_id")
