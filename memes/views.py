# api/views.py
from django.views.decorators.csrf import csrf_exempt
import re
import nltk
import pandas as pd
from django.http import JsonResponse
from django.views import View
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from django.shortcuts import render

def home(request):
    return render(request, 'home.html')

# Load the dataset globally
jokes_df = pd.read_excel("/Users/keyur/meme_generator/memes/assets/dataset copy.xlsx")  # Update with your file path
joke_corpus = dict(zip(jokes_df["Input"], jokes_df["Response"]))

# Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")

# Define a function to preprocess text
def preprocess(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    return " ".join(tokens)

# Preprocess the joke corpus
processed_corpus = [preprocess(question) for question in joke_corpus.keys()]

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(processed_corpus)

class MemeGenerator(View):
    @method_decorator(csrf_exempt)
    def post(self, request):
        user_input = request.POST.get("input")
        query = preprocess(user_input)
        query_vec = tfidf_vectorizer.transform([query])
        cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_n_indices = cosine_similarities.argsort()[-10:][::-1]

        top_similar_jokes = [
            {"question": list(joke_corpus.keys())[i], "response": joke_corpus[list(joke_corpus.keys())[i]]}
            for i in top_n_indices
        ]

        return JsonResponse(top_similar_jokes, safe=False)

