from django.urls import path
from .views import get_memes
from django.conf.urls.static import static
from django.conf import settings


urlpatterns = [
    path('get_memes/', get_memes, name='get_memes'),
    # path('get_memes_with_text_overlay/', get_memes_with_text_overlay, name='get_memes_with_text_overlay'),
    # path('overlay_text_on_image/', overlay_text_on_image, name='overlay_text_on_image'),

        # For fetching memes based on text
    # path('predict-image-emotions/', predict_image_emotions, name='predict_image_emotions'),  # For predicting image emotions
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
