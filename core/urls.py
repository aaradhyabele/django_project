from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('predict/', views.fraud_prediction, name='predict'),
    path('about/', views.about, name='about'),
    path('report/', views.report, name='report'),
    path('fraud_analysis/', views.fraud_analysis, name='fraud_analysis'),
]

