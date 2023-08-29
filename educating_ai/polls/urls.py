from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path('capture/', views.capture_image, name='capture_image'),
    path('cap_helper/', views.cap_helper, name='cap_helper'),
    path('cap/', views.capture, name='cap'),
]