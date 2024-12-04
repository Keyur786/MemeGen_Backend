
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
        

# def get_memes(request):
#     if request.method == 'GET':
#         user_question = request.GET.get('query', '').strip()
#         if not user_question:
#             return JsonResponse({'error': 'Query parameter missing'}, status=400)

#         # Predict text emotion for the user query
#         text_emotion = predict_text_emotion(user_question)

#         # Get top matching memes
#         top_memes = get_top_similar_memes(user_question)

#         # Process images and predict emotions
#         images = get_list_or_404(MemeImage)
#         image_results = []
#         for meme in images:
#             image_emotion = predict_image_emotion(meme.image_file.path)
#             image_results.append({
#                 "image": meme.image_file.url,
#                 "emotion": image_emotion
#             })

#         # Match images to top_memes based on their emotions
#         memes_with_images = []
#         for meme_text in top_memes:
#             meme_emotion = predict_text_emotion(meme_text)
#             matching_image = next(
#                 (img['image'] for img in image_results if img['emotion'] == meme_emotion),
#                 None
#             )
#             memes_with_images.append({
#                 "text": meme_text,
#                 "emotion": meme_emotion,
#                 "image": matching_image
#             })

#         # Construct response
#         response = {
#             "query": user_question,
#             "text_emotion": text_emotion,
#             "top_memes": memes_with_images,
#             "all_images": image_results
#         }

#         return JsonResponse(response, status=200)

# def overlay_text_on_image(image_path, text):
#     """Overlay text on an image and save it."""
#     # Resolve the full absolute path
#     file_path = os.path.abspath(image_path)

#     # Check if the file exists
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"Image file not found: {file_path}")

#     try:
#         # Open the image from the path
#         img = Image.open(file_path)
#     except Exception as e:
#         raise RuntimeError(f"Error opening image: {e}")

#     # Create a drawing context
#     draw = ImageDraw.Draw(img)

#     # Font settings
#     font_path = os.path.join(os.getcwd(), "arial.ttf")
#     font = ImageFont.truetype(font_path, size=40) if os.path.exists(font_path) else ImageFont.load_default()

#     # Get text bounding box to calculate positioning
#     text_bbox = draw.textbbox((0, 0), text, font=font)
#     text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
#     text_position = (img.width // 2 - text_width // 2, img.height - text_height - 10)

#     # Draw the text on the image
#     draw.text(text_position, text, font=font, fill="white")

#     # Save the new image with overlayed text
#     overlayed_image_path = os.path.join(os.getcwd(), "overlayed_" + os.path.basename(file_path))
#     img.save(overlayed_image_path)

#     return overlayed_image_path


# def get_memes_with_text_overlay(request):
#     if request.method == 'GET':
#         user_query = request.GET.get('query', '').strip()
#         if not user_query:
#             return JsonResponse({'error': 'Query parameter missing'}, status=400)

#         # Predict emotion of the user's query
#         text_emotion = predict_text_emotion(user_query)

#         # Get the top matching memes (text + image pairs)
#         top_memes = get_top_similar_memes(user_query)

#         overlayed_memes = []

#         for meme in top_memes:
#             # Validate the format of the meme object
#             if not isinstance(meme, dict) or 'text' not in meme or 'image' not in meme:
#                 print(f"Skipping invalid meme: {meme}")
#                 continue

#             text = meme['text']
#             original_image_path = os.path.join(settings.MEDIA_ROOT, meme['image'].lstrip('/'))

#             try:
#                 # Overlay text on the image
#                 overlayed_image_path = overlay_text_on_image(original_image_path, text)

#                 # Construct overlayed meme details
#                 overlayed_memes.append({
#                     "text": text,
#                     "emotion": meme['emotion'],
#                     "image": overlayed_image_path.replace(settings.MEDIA_ROOT, settings.MEDIA_URL),
#                 })

#             except Exception as e:
#                 print(f"Error processing image: {original_image_path} with text '{text}'. Error: {e}")

#         # Response data
#         response_data = {
#             "query": user_query,
#             "text_emotion": text_emotion,
#             "top_memes": overlayed_memes,
#         }

#         return JsonResponse(response_data, status=200)





#generate images but tetx can not fill in the image 
# def get_memes(request):
#     if request.method == 'GET':
#         user_question = request.GET.get('query', '').strip()
#         if not user_question:
#             return JsonResponse({'error': 'Query parameter missing'}, status=400)

#         # Predict text emotion for the user query
#         text_emotion = predict_text_emotion(user_question)

#         # Get top matching memes
#         top_memes = get_top_similar_memes(user_question)

#         # Process images and predict emotions
#         images = get_list_or_404(MemeImage)
#         image_results = []
#         for meme in images:
#             image_emotion = predict_image_emotion(meme.image_file.path)
#             image_results.append({
#                 "image": meme.image_file.path,  # Use path for further processing
#                 "url": meme.image_file.url,    # Use URL for frontend access
#                 "emotion": image_emotion
#             })

#         # Match images to top_memes based on their emotions
#         memes_with_images = []
#         for meme_text in top_memes:
#             meme_emotion = predict_text_emotion(meme_text)
#             matching_image = next(
#                 (img for img in image_results if img['emotion'] == meme_emotion),
#                 None
#             )
#             if matching_image:
#                 # Overlay text on the matching image
#                 overlaid_image_path = overlay_text_on_image(
#                     matching_image['image'], meme_text
#                 )
#                 memes_with_images.append({
#                     "text": meme_text,
#                     "emotion": meme_emotion,
#                     "image_url": matching_image['url'],  # Original image URL
#                     "overlaid_image_url": overlaid_image_path  # Path to overlaid image
#                 })

#         # Construct response
#         response = {
#             "query": user_question,
#             "text_emotion": text_emotion,
#             "top_memes": memes_with_images,
#         }

#         return JsonResponse(response, status=200)


# def overlay_text_on_image(image_path, text):
#     """
#     Overlays text onto an image and saves the modified image.
#     """
#     # Open the image
#     image = Image.open(image_path)
#     draw = ImageDraw.Draw(image)
#     font_path = os.path.join(settings.BASE_DIR, "static/fonts/Arial.ttf")  # Update with your font path
#     font = ImageFont.truetype(font_path, size=24)  # Adjust size as needed

#     # Get text dimensions using getbbox
#     text_bbox = font.getbbox(text)
#     text_width = text_bbox[2] - text_bbox[0]
#     text_height = text_bbox[3] - text_bbox[1]

#     # Image dimensions
#     image_width, image_height = image.size
#     x = (image_width - text_width) / 2  # Center the text horizontally
#     y = image_height - text_height - 10  # Place text near the bottom with padding

#     # Draw text with background
#     draw.rectangle(
#         [(x - 10, y - 10), (x + text_width + 10, y + text_height + 10)],
#         fill="black"
#     )
#     draw.text((x, y), text, fill="white", font=font)

#     # Save the modified image to a temporary location
#     output_path = os.path.join(settings.MEDIA_ROOT, "generated_memes", os.path.basename(image_path))
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     image.save(output_path)

#     # Return the URL to the modified image
#     return default_storage.url(output_path)


## working final 
# def overlay_text_on_image(image_path, text, output_filename):
#     """
#     Overlays text onto an image and saves the modified image.
#     """
#     # Open the image
#     image = Image.open(image_path)
#     draw = ImageDraw.Draw(image)
#     font_path = os.path.join(settings.BASE_DIR, "static/fonts/Arial.ttf")  # Update with your font path
#     font = ImageFont.truetype(font_path, size=24)  # Adjust size as needed

#     # Get text dimensions
#     text_bbox = font.getbbox(text)
#     text_width = text_bbox[2] - text_bbox[0]
#     text_height = text_bbox[3] - text_bbox[1]

#     # Image dimensions
#     image_width, image_height = image.size
#     x = (image_width - text_width) / 2  # Center the text horizontally
#     y = image_height - text_height - 10  # Place text near the bottom with padding

#     # Draw text with background
#     draw.rectangle(
#         [(x - 10, y - 10), (x + text_width + 10, y + text_height + 10)],
#         fill="black"
#     )
#     draw.text((x, y), text, fill="white", font=font)

#     # Save the modified image to a temporary location
#     output_path = os.path.join(settings.MEDIA_ROOT, "generated_memes", output_filename)
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     image.save(output_path)

#     # Return the URL to the modified image
#     return default_storage.url(output_path) 

# def get_memes(request):
#     if request.method == 'GET':
#         user_question = request.GET.get('query', '').strip()
#         if not user_question:
#             return JsonResponse({'error': 'Query parameter missing'}, status=400)

#         # Predict text emotion for the user query
#         text_emotion = predict_text_emotion(user_question)

#         # Get top matching memes
#         top_memes = get_top_similar_memes(user_question)

#         # Fetch all images from the database
#         images = MemeImage.objects.all()

#         # Prepare image results with predicted emotions
#         image_results = []
#         for image in images:
#             predicted_emotion = predict_image_emotion(image.image_file.path)
#             image_results.append({
#                 "image": image.image_file.url,
#                 "predicted_emotion": predicted_emotion,  # Predicted emotion
#                 "path": image.image_file.path  # Needed for processing
#             })

#         # Match images to top_memes based on predicted emotions and overlay text
#         memes_with_images = []
#         for index, meme_text in enumerate(top_memes):
#             meme_emotion = predict_text_emotion(meme_text)

#             # Find a matching image based on predicted emotion
#             matching_image_data = next(
#                 (img for img in image_results if img['predicted_emotion'] == meme_emotion),
#                 None
#             )

#             # Overlay text on the image if a match is found
#             if matching_image_data:
#                 image_path = matching_image_data['path']
#                 output_filename = f"meme_{index}.jpg"
#                 modified_image_url = overlay_text_on_image(image_path, meme_text, output_filename)

#                 # Add the meme with image to the results
#                 memes_with_images.append({
#                     "text": meme_text,
#                     "text_emotion": meme_emotion,
#                     "image": modified_image_url,
#                     "image_emotion": matching_image_data['predicted_emotion'],  # Predicted emotion of the image
#                 })

#         # Construct response
#         response = {
#             "query": user_question,
#             "text_emotion": text_emotion,
#             "top_memes": memes_with_images,
#             "all_images": image_results
#         }

#         return JsonResponse(response, status=200)





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


def get_memes(request):
    if request.method == 'GET':
        user_question = request.GET.get('query', '').strip()
        if not user_question:
            return JsonResponse({'error': 'Query parameter missing'}, status=400)

        text_emotion = predict_text_emotion(user_question)

        top_memes = get_top_similar_memes(user_question)

        images = MemeImage.objects.all()

        image_results = []
        for image in images:
            predicted_emotion = predict_image_emotion(image.image_file.path)
            image_results.append({
                "image": image.image_file.url,
                "predicted_emotion": predicted_emotion,  
                "path": image.image_file.path 
            })

        memes_with_images = []
        for index, meme_text in enumerate(top_memes):
            meme_emotion = predict_text_emotion(meme_text)

            matching_image_data = next(
                (img for img in image_results if img['predicted_emotion'] == meme_emotion),
                None
            )

            if matching_image_data:
                image_path = matching_image_data['path']
                output_filename = f"meme_{index}.jpg"
                modified_image_url = overlay_text_on_image(image_path, meme_text, output_filename)

                memes_with_images.append({
                    "text": meme_text,
                    "text_emotion": meme_emotion,
                    "image": modified_image_url,
                    "image_emotion": matching_image_data['predicted_emotion'],  
                })

        response = {
            "query": user_question,
            "text_emotion": text_emotion,
            "top_memes": memes_with_images,
            "all_images": image_results
        }

        return JsonResponse(response, status=200)
