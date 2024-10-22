from django.urls import path
from .views import get_memes

urlpatterns = [
    path('get_memes/', get_memes, name='get_memes'),  # URL for generating memes
]
