import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def train_model():
    # Load dataset
    data = pd.read_csv("spam.csv")

    # Convert labels
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})

    X = data['message']
    y = data['label']

    # Vectorization
    vectorizer = CountVectorizer()
    X_vec = vectorizer.fit_transform(X)

    # Train model
    model = MultinomialNB()
    model.fit(X_vec, y)

    return model, vectorizer
