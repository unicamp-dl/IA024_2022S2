from django.urls import path

from . import views

urlpatterns = [
    path('receive_losses', views.receive_loss, name="receive_loss"),
    path('sinc_losses', views.sinc_loss, name="sinc_loss"),
    path('receive_weights', views.receive_weight, name="receive_weights"),
    path('sinc_weights', views.sinc_weight, name="sinc_weights"),
    path('reset_cache', views.reset_cache, name="reset_cache"),
]
