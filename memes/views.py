
from django.shortcuts import get_list_or_404
from django.http import JsonResponse
from PIL import Image
import numpy as np
import re
import nltk
import torch
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, AutoModelForSequenceClassification
from .models import MemeImage
import tensorflow as tf
from django.http import JsonResponse, HttpResponse
from PIL import Image, ImageDraw, ImageFont
import os
from django.conf import settings
from django.core.files.storage import default_storage


emotion_tokenizer = BertTokenizer.from_pretrained('/Users/keyur/meme_generator/memes/emotional_model')
emotion_model_text = AutoModelForSequenceClassification.from_pretrained('/Users/keyur/meme_generator/memes/emotional_model')
emotion_model_text.eval()

emotion_model_image = tf.keras.models.load_model("/Users/keyur/meme_generator/memes/emotional_model/emotion_detection_model.h5")

text_emotion_labels = {0: 'neutral', 1: 'sad', 2: 'fearful', 3: 'angry', 4: 'happy'}
image_emotion_labels = {0: "angry", 1: "disgusted", 2: "fearful", 3: "happy", 4: "neutral", 5: "sad", 6: "surprised"}

jokes_df = pd.read_excel("/Users/keyur/meme_generator/memes/assets/dataset copy.xlsx")
jokes_df['text_corrected'] = jokes_df['Response'].fillna('')
joke_corpus = jokes_df["Response"].tolist()

nltk.download("punkt")
nltk.download("stopwords")
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    tokens = nltk.word_tokenize(text)
    return " ".join([word for word in tokens if word not in stopwords.words("english")])

def preprocess_image(image_path):
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img = img.resize((48, 48)) 
    img_array = np.array(img) / 255.0  # Normalize
    img_array = img_array.reshape(1, 48, 48, 1) 
    return img_array

def predict_text_emotion(text):
    encoding = emotion_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    if "token_type_ids" in encoding:
        del encoding["token_type_ids"]
    outputs = emotion_model_text(**encoding)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return text_emotion_labels[prediction]

def predict_image_emotion(image_path):
    preprocessed_image = preprocess_image(image_path)
    predictions = emotion_model_image.predict(preprocessed_image)
    predicted_index = int(np.argmax(predictions[0]))
    return image_emotion_labels.get(predicted_index, "unknown")

def get_top_similar_memes(query, top_n=10):
    query_tokens = preprocess_text(query).split()
    matched_entries = [key for key in joke_corpus if all(token in preprocess_text(key) for token in query_tokens)]
    if matched_entries:
        return matched_entries[:top_n]
    matched_entries_any = [key for key in joke_corpus if any(token in preprocess_text(key) for token in query_tokens)]
    return matched_entries_any[:top_n]
        

def overlay_text_on_image(image_path, text, output_filename):
    """
    Overlays text onto an image and saves the modified image.
    """
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    font_path = os.path.join(settings.BASE_DIR, "static/fonts/Arial.ttf")  
    font = ImageFont.truetype(font_path, size=12) 

    text_bbox = font.getbbox(text)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    image_width, image_height = image.size
    x = (image_width - text_width) / 2  
    y = image_height - text_height - 10  

    draw.rectangle(
        [(x - 10, y - 10), (x + text_width + 10, y + text_height + 10)],
        fill="black"
    )
    draw.text((x, y), text, fill="white", font=font)

    output_path = os.path.join(settings.MEDIA_ROOT, "generated_memes", output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)

    return os.path.join(settings.MEDIA_URL, "generated_memes", output_filename)


# def get_memes(request):
#     if request.method == 'GET':
#         user_question = request.GET.get('query', '').strip()
#         if not user_question:
#             return JsonResponse({'error': 'Query parameter missing'}, status=400)

#         text_emotion = predict_text_emotion(user_question)

#         top_memes = get_top_similar_memes(user_question)

#         images = MemeImage.objects.all()

#         image_results = []
#         for image in images:
#             predicted_emotion = predict_image_emotion(image.image_file.path)
#             image_results.append({
#                 "image": image.image_file.url,
#                 "predicted_emotion": predicted_emotion,  
#                 "path": image.image_file.path 
#             })

#         memes_with_images = []
#         for index, meme_text in enumerate(top_memes):
#             meme_emotion = predict_text_emotion(meme_text)

#             matching_image_data = next(
#                 (img for img in image_results if img['predicted_emotion'] == meme_emotion),
#                 None
#             )

#             if matching_image_data:
#                 image_path = matching_image_data['path']
#                 output_filename = f"meme_{index}.jpg"
#                 modified_image_url = overlay_text_on_image(image_path, meme_text, output_filename)

#                 memes_with_images.append({
#                     "text": meme_text,
#                     "text_emotion": meme_emotion,
#                     "image": modified_image_url,
#                     "image_emotion": matching_image_data['predicted_emotion'],  
#                 })

#         response = {
#             "query": user_question,
#             "text_emotion": text_emotion,
#             "top_memes": memes_with_images,
#             "all_images": image_results
#         }

#         return JsonResponse(response, status=200)


def get_memes(request):
    if request.method == 'GET':
        user_question = request.GET.get('query', '').strip()
        if not user_question:
            return JsonResponse({'error': 'Query parameter missing'}, status=400)

        # Predict the emotion of the user's query
        text_emotion = predict_text_emotion(user_question)

        # Get top memes based on the query
        top_memes = get_top_similar_memes(user_question)

        # Fetch all meme image templates
        images = MemeImage.objects.all()

        # Predict emotions for all meme templates
        image_results = []
        for image in images:
            predicted_emotion = predict_image_emotion(image.image_file.path)
            image_results.append({
                "image": image.image_file.url,
                "predicted_emotion": predicted_emotion,
                "path": image.image_file.path
            })

        # Generate memes with all matching templates
        memes_with_images = []
        for index, meme_text in enumerate(top_memes):
            meme_emotion = predict_text_emotion(meme_text)

            # Get all matching image templates for the current meme emotion
            matching_images = [
                img for img in image_results if img['predicted_emotion'] == meme_emotion
            ]

            # Create memes using all matching images
            for img_index, matching_image_data in enumerate(matching_images):
                image_path = matching_image_data['path']
                output_filename = f"meme_{index}_{img_index}.jpg"
                modified_image_url = overlay_text_on_image(image_path, meme_text, output_filename)

                memes_with_images.append({
                    "text": meme_text,
                    "text_emotion": meme_emotion,
                    "image": modified_image_url,
                    "image_emotion": matching_image_data['predicted_emotion'],
                })

        # Prepare response data
        response = {
            "query": user_question,
            "text_emotion": text_emotion,
            "top_memes": memes_with_images,
            "all_images": image_results
        }

        return JsonResponse(response, status=200)
