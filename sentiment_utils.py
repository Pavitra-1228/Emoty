"""Utility helpers for mapping emotions to sentiment polarity."""

from typing import Dict, Iterable, List, Optional, Tuple


# Mapping from dataset emotion labels to a simplified polarity.
# Choose the mappings that make sense for your application.
EMOTION_TO_POLARITY: Dict[str, str] = {
    "anger": "negative",
    "hate": "negative",
    "sadness": "negative",
    "worry": "negative",
    "boredom": "negative",
    "empty": "negative",

    "neutral": "neutral",

    "love": "positive",
    "happiness": "positive",
    "relief": "positive",
    "fun": "positive",
    "enthusiasm": "positive",
    "surprise": "positive",
}


def map_polarity(emotion_label: str) -> str:
    """Map a single emotion label to polarity.

    If the emotion label isn't in the mapping, returns "neutral".
    """

    return EMOTION_TO_POLARITY.get(emotion_label, "neutral")


def polarity_distribution(probs: Iterable[float], labels: List[str]) -> Dict[str, float]:
    """Convert emotion probability distribution into polarity distribution.

    For each polarity, sum the probabilities of all emotions that map to it.
    """

    polarity_sums: Dict[str, float] = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    for prob, label in zip(probs, labels):
        polarity = map_polarity(label)
        polarity_sums[polarity] += float(prob)
    return polarity_sums
