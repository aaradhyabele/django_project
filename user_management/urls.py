from django.urls import path
from . import views

urlpatterns = [
    path('', views.user_list, name='user_list'),
    path('create/', views.user_create, name='user_create'),
    path('assign-role/<int:user_id>/', views.assign_role, name='assign_role'),
]
