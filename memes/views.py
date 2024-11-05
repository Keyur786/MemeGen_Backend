# import re
# import nltk
# import pandas as pd
# from django.http import JsonResponse
# from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import TfidfVectorizer

# nltk.download("punkt")
# nltk.download("stopwords")

# jokes_df = pd.read_excel("/Users/keyur/meme_generator/memes/assets/dataset copy.xlsx")

# jokes_df['text_corrected'] = jokes_df['Response'].fillna('')

# joke_corpus = jokes_df["Response"].tolist()

# # Function to preprocess text
# def preprocess(text):
#     if not isinstance(text, str):
#         text = str(text)
#     text = re.sub(r'[^a-zA-Z]', ' ', text)
#     text = text.lower()
#     tokens = nltk.word_tokenize(text)
#     tokens = [word for word in tokens if word not in stopwords.words("english")]
#     return " ".join(tokens)

# processed_corpus = [preprocess(joke) for joke in joke_corpus if joke.strip()]

# tfidf_vectorizer = TfidfVectorizer()
# tfidf_matrix = tfidf_vectorizer.fit_transform(processed_corpus)

# # Function to get top similar memes
# def get_top_similar_memes(query, top_n=10):
#     query_tokens = preprocess(query).split()

#     matched_entries = [
#         key for key in joke_corpus if all(token in preprocess(key) for token in query_tokens)
#     ]

#     if matched_entries:
#         return matched_entries[:top_n]

#     matched_entries_any = [
#         key for key in joke_corpus if any(token in preprocess(key) for token in query_tokens)
#     ]

#     return matched_entries_any[:top_n]

# # Django view to handle requests
# def get_memes(request):
#     if request.method == 'GET':
#         user_question = request.GET.get('query', '')
#         if user_question:
#             top_jokes = get_top_similar_memes(user_question)
#             return JsonResponse({'question': user_question, 'suggestions': top_jokes}, status=200)
#         else:
#             return JsonResponse({'error': 'Query parameter missing'}, status=400)



import re
import nltk
import torch
import pandas as pd
from django.http import JsonResponse
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, AutoModelForSequenceClassification

nltk.download("punkt")
nltk.download("stopwords")

# Load dataset
jokes_df = pd.read_excel("/Users/keyur/meme_generator/memes/assets/dataset copy.xlsx")
jokes_df['text_corrected'] = jokes_df['Response'].fillna('')
joke_corpus = jokes_df["Response"].tolist()

# Load emotion detection model and tokenizer
emotion_tokenizer = BertTokenizer.from_pretrained('/Users/keyur/meme_generator/memes/emotional_model')
emotion_model = AutoModelForSequenceClassification.from_pretrained('/Users/keyur/meme_generator/memes/emotional_model')
emotion_model.eval()  # Set the model to evaluation mode

# Define inverse label map (assuming you have this saved or can recreate from training)
inverse_label_map = {0: 'neutral', 1: 'sadness', 2: 'fear', 3: 'anger', 4: 'joy'}  # Example label map

# Preprocessing function
def preprocess(text):
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    return " ".join(tokens)

# Generate TF-IDF matrix for jokes
processed_corpus = [preprocess(joke) for joke in joke_corpus if joke.strip()]
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(processed_corpus)

# Function to get top similar memes
def get_top_similar_memes(query, top_n=10):
    query_tokens = preprocess(query).split()
    matched_entries = [key for key in joke_corpus if all(token in preprocess(key) for token in query_tokens)]
    if matched_entries:
        return matched_entries[:top_n]
    matched_entries_any = [key for key in joke_corpus if any(token in preprocess(key) for token in query_tokens)]
    return matched_entries_any[:top_n]

# Emotion prediction function
def predict_emotion(text):
    encoding = emotion_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    if "token_type_ids" in encoding:
        del encoding["token_type_ids"]
    outputs = emotion_model(**encoding)
    prediction = torch.argmax(outputs.logits, dim=1)
    return inverse_label_map[prediction.item()]

# Django view to handle requests
def get_memes(request):
    if request.method == 'GET':
        user_question = request.GET.get('query', '')
        if user_question:
            top_jokes = get_top_similar_memes(user_question)
            memes_with_emotion = []

            # Predict emotion for each meme suggestion
            for joke in top_jokes:
                emotion = predict_emotion(joke)
                memes_with_emotion.append({"meme_text": joke, "emotion": emotion})

            return JsonResponse({'question': user_question, 'suggestions': memes_with_emotion}, status=200)
        else:
            return JsonResponse({'error': 'Query parameter missing'}, status=400)
