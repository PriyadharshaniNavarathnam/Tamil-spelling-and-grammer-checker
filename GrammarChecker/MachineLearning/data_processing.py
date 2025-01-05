import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

def load_data(filepath):
    data = pd.read_excel(filepath)
    return data['Original Sentence'].values, data['Error Type'].values, data['Corrected Sentence'].values

def preprocess_data(sentences):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)
    return X, vectorizer

def encode_labels(labels):
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    return y, encoder
