"""FastAPI backend for the sentiment/emotion model."""

import os
from typing import Dict, List, Optional

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sentiment_utils import map_polarity, polarity_distribution


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    emotion: str
    polarity: str
    emotion_probs: Dict[str, float]
    polarity_probs: Dict[str, float]


def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
    return joblib.load(model_path)


app = FastAPI(title="Sentiment/Emotion API", version="1.0")
model = load_model()


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text cannot be empty")

    emotion = model.predict([text])[0]
    polarity = map_polarity(emotion)

    emotion_probs: Dict[str, float] = {}
    polarity_probs: Dict[str, float] = {}
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([text])[0]
        labels = list(model.classes_)
        emotion_probs = {label: float(p) for label, p in zip(labels, proba)}
        polarity_probs = polarity_distribution(proba, labels)

    return PredictResponse(
        emotion=emotion,
        polarity=polarity,
        emotion_probs=emotion_probs,
        polarity_probs=polarity_probs,
    )
