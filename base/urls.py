from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('news/<int:news_id>/', views.news_detail, name='news_detail'),
    path('news/<int:news_id>/analyze/', views.news_analyze, name='news_analyze'),
    path('api/feed/status/', views.feed_status, name='feed_status'),
    path('api/news/<int:news_id>/status/', views.news_status, name='news_status'),
    path('api/news/<int:news_id>/chat/', views.chat_api, name='chat_api'),
]
