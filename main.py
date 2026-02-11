import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

# Import local modules
from src.features import extract_text_features
from src.model import MultinomialNB

# Configuration
DATA_PATH = 'data/dataset.csv'
TEST_SIZE = 0.2
RANDOM_STATE = 42

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    return df['Text'].tolist(), df['language'].tolist()

def main():
    # 1. Load Data
    try:
        texts, labels = load_data(DATA_PATH)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"Loaded {len(texts)} texts.")

    # 2. Split Data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 3. Feature Extraction (TF-IDF + Open N-Grams)
    print("Vectorizing text (this may take a moment)...")
    
    # We pass our custom 'extract_text_features' function as the analyzer
    vectorizer = TfidfVectorizer(analyzer=extract_text_features)
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"Vocabulary size: {X_train_tfidf.shape[1]} features.")

    # 4. Train Model
    print("Training Custom Naive Bayes...")
    model = MultinomialNB(alpha=1.0)
    model.fit(X_train_tfidf, y_train)

    # 5. Evaluate
    print("Predicting on test set...")
    y_pred = model.predict(X_test_tfidf)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {acc*100:.2f}%")
    
    # Detailed report
    print("\nClassification Report:")
    # We explicitly exclude the invalid parameter here
    print(classification_report(y_test, y_pred))

    # 6. Live Demo
    print("-" * 30)
    print("Live Demo")
    sample_texts = [
        "This is a test sentence in English.",
        "Esto es una frase de prueba en Español.",
        "Ceci est une phrase de test en Français.",
    ]
    
    sample_vecs = vectorizer.transform(sample_texts)
    predictions = model.predict(sample_vecs)
    
    for text, pred in zip(sample_texts, predictions):
        print(f"Text: '{text}' -> Predicted: {pred}")

if __name__ == "__main__":
    main()