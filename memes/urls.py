from django.urls import path
from .views import MemeGenerator, home

urlpatterns = [
    path('', home, name='home'),  # Root URL for home (http://127.0.0.1:8000/memes/)
    path('generate_memes/', MemeGenerator.as_view(), name='generate_memes'),  # URL for generating memes
]
