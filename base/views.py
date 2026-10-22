from django.http import Http404
from django.shortcuts import render

from .mock_data import CATEGORIES, COMPANY, FEED_UPDATED, MARKETS, NEWS, USER


def _common_context():
    return {'user': USER, 'company': COMPANY}


def index(request):
    categories = [
        {'slug': slug, 'label': label, 'count': sum(n['category'] == slug for n in NEWS)}
        for slug, label in CATEGORIES
    ]
    context = {
        **_common_context(),
        'news': NEWS,
        'urgent_news': [n for n in NEWS if n['urgent']],
        'categories': [c for c in categories if c['count']],
        'markets': MARKETS,
        'feed_updated': FEED_UPDATED,
    }
    return render(request, 'base/index.html', context)


def news_detail(request, news_id):
    item = next((n for n in NEWS if n['id'] == news_id), None)
    if item is None:
        raise Http404('Новость не найдена')
    context = {
        **_common_context(),
        'item': item,
        'related': [n for n in NEWS if n['id'] != news_id][:3],
    }
    return render(request, 'base/news_detail.html', context)
