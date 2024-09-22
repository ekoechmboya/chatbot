from django.urls import path, include
from . import views


urlpatterns = [             
    path('register', views.register, name="register"),
    path('login/', views.login_user, name="login"),
    path('', views.login_user, name="login"),
    path('logout/', views.logout_view, name="logout"),
    path('chatbot/chat/', views.chat_home, name='chat'),

]
