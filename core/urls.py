from django.urls import path
from . import views
from django.urls import path
from . import views
from django.contrib.auth import views as auth_views


urlpatterns = [
    path('login/', auth_views.LoginView.as_view(template_name='core/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='home'), name='logout'),
    path('', views.home, name='home'),
    path('predict/', views.fraud_prediction, name='predict'),
    path('about/', views.about, name='about'),
    path('report/', views.report, name='report'),
    path('fraud_analysis/', views.fraud_analysis, name='fraud_analysis'),
    path('export_fraud_csv/', views.export_fraud_csv, name='export_fraud_csv'),
    path('train-global-model/', views.train_global_model, name='train_global_model'),

    # Registration
    path('register/', views.register, name='register'),
]


