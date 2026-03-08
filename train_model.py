"""Train a sentiment/emotion classification model from emotion_sentimen_dataset.csv.

Usage:
    python train_model.py

Output:
    - model.joblib: trained sklearn Pipeline (TfidfVectorizer + LogisticRegression)
"""

import os

import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing import clean_text


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """A simple sklearn transformer that cleans text."""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # sklearn passes numpy arrays; we want pandas-like apply to keep simple
        return pd.Series(X).astype(str).apply(clean_text)


def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    # Drop any unnamed index column if it exists
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df = df.dropna(subset=["text", "Emotion"]).reset_index(drop=True)
    return df


def train(csv_path: str, out_path: str):
    df = load_data(csv_path)

    X = df["text"].astype(str)
    y = df["Emotion"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        [
            ("cleaner", TextPreprocessor()),
            ("tfidf", TfidfVectorizer(max_features=50_000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("Classification report (test set):")
    print(classification_report(y_test, y_pred, digits=4))

    joblib.dump(pipeline, out_path)
    print(f"Saved trained model to: {out_path}")


if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), "emotion_sentimen_dataset.csv")
    out_path = os.path.join(os.path.dirname(__file__), "model.joblib")
    train(csv_path, out_path)
