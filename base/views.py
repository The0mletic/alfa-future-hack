import json
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.db.models import Count

from .models import News, Company, Category
from .mock_data import USER, MARKETS, FEED_UPDATED  # Маркетс и юзера пока оставим статикой


def _get_company():
    # Для хакатона берем первую компанию из БД (в реальности — из сессии/токена)
    return Company.objects.first() or {
        'name': 'ООО «Восток Логистик»', 'initials': 'ВЛ', 'inn': '5406 718 293',
        'client_since': '2014', 'industry': 'Грузоперевозки', 'region': 'Новосибирская обл.',
        'revenue': '18,4 млрд ₽', 'foreign_trade': 'Китай, Казахстан', 'products': ['Кредиты']
    }


def _common_context():
    return {'user': USER, 'company': _get_company()}


def index(request):
    # Получаем новости, сортируем по убыванию релевантности
    news_qs = News.objects.all().select_related('category')

    # Формируем категории с подсчетом новостей
    categories = Category.objects.annotate(count=Count('news')).filter(count__gt=0)

    context = {
        **_common_context(),
        'news': news_qs,
        'urgent_news': news_qs.filter(urgent=True),
        'categories': categories,
        'markets': MARKETS,
        'feed_updated': FEED_UPDATED,
    }
    return render(request, 'base/index.html', context)


def news_detail(request, news_id):
    item = get_object_or_404(News, id=news_id)
    # Похожие новости (просто другие из той же категории)
    related = News.objects.filter(category=item.category).exclude(id=news_id)[:3]

    context = {
        **_common_context(),
        'item': item,
        'related': related,
    }
    return render(request, 'base/news_detail.html', context)


@require_POST
def chat_api(request, news_id):
    """
    Эндпоинт для мини-чата с LLM.
    Ожидает JSON: {"message": "текст вопроса"}
    Возвращает JSON: {"answer": "ответ модели"}
    """
    try:
        data = json.loads(request.body)
        message = data.get('message', '')
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    news_item = get_object_or_404(News, id=news_id)

    # ==========================================
    # ЗДЕСЬ ВАША ЛОГИКА LLM / МУЛЬТИАГЕНТНОЙ СИСТЕМЫ
    # ==========================================
    # Пока для теста отдаем заглушку, которая зависит от вопроса
    # TODO: убрать заглушку

    if 'финанс' in message.lower() or 'влияние' in message.lower():
        answer = f"Согласно нашему анализу, это событие повлияет на {news_item.company.name if hasattr(news_item, 'company') else 'вашу компанию'} через риск: {news_item.risks[0]['title'] if news_item.risks else 'не выявлен'}."
    elif 'делать' in message.lower():
        answer = "Рекомендую ознакомиться с блоком «Возможности» и обсудить шаги с персональным менеджером Альфа-Банка."
    elif 'основан' in message.lower():
        answer = f"Вывод основан на фактах из источника «{news_item.source}» и аналитике банка: {news_item.basis}."
    else:
        answer = f"Я изучил новость «{news_item.title}». Спросите меня о рисках, возможностях или конкретных действиях."

    return JsonResponse({'answer': answer})