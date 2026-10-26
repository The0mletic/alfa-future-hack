from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('news/<int:news_id>/', views.news_detail, name='news_detail'),
    path('api/news/<int:news_id>/chat/', views.chat_api, name='chat_api'), # <-- Новый API
]