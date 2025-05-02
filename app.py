import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Preprocessing function
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = [stemmer.stem(word) for word in text.split() if word not in stop_words]
    return ' '.join(words)

# Load model and TF-IDF (after training, save these as files!)
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# Streamlit UI
st.title("🎬 Movie Review Sentiment Analyzer")
review = st.text_area("Enter your movie review:")

if st.button("Predict"):
    cleaned = preprocess(review)
    vectorized = tfidf.transform([cleaned]).toarray()
    prediction = model.predict(vectorized)[0]
    st.write("### Sentiment: ", "😊 Positive" if prediction == 1 else "😠 Negative")
