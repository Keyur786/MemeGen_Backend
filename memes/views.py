import re
import nltk
import pandas as pd
from django.http import JsonResponse
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")

# Load the dataset (ensure the path is correct or load it globally)
jokes_df = pd.read_excel("/Users/keyur/meme_generator/memes/assets/dataset copy.xlsx")

# Ensure NaN values in 'text_corrected' column are handled
jokes_df['text_corrected'] = jokes_df['Response'].fillna('')

# Create a list for text_corrected values
joke_corpus = jokes_df["Response"].tolist()

# Function to preprocess text
def preprocess(text):
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    return " ".join(tokens)

# Preprocess the joke corpus
processed_corpus = [preprocess(joke) for joke in joke_corpus if joke.strip()]

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(processed_corpus)

# Function to get top similar jokes
def get_top_similar_memes(query, top_n=10):
    query_tokens = preprocess(query).split()

    matched_entries = [
        key for key in joke_corpus if all(token in preprocess(key) for token in query_tokens)
    ]

    if matched_entries:
        return matched_entries[:top_n]

    matched_entries_any = [
        key for key in joke_corpus if any(token in preprocess(key) for token in query_tokens)
    ]

    return matched_entries_any[:top_n]

# Django view to handle requests
def get_memes(request):
    if request.method == 'GET':
        user_question = request.GET.get('query', '')
        if user_question:
            top_jokes = get_top_similar_memes(user_question)
            return JsonResponse({'question': user_question, 'suggestions': top_jokes}, status=200)
        else:
            return JsonResponse({'error': 'Query parameter missing'}, status=400)
