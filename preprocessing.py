"""Text preprocessing utilities for sentiment/emotion analysis."""

import re


URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMOJI_RE = re.compile(
    "["  # Start character class
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U00002702-\U000027B0"  # dingbats
    "\U000024C2-\U0001F251"  # enclosed characters
    "]+",
    flags=re.UNICODE,
)
PUNCT_RE = re.compile(r"[^\w\s]")


def clean_text(text: str) -> str:
    """Clean text for model input.

    - Lowercases text
    - Removes URLs
    - Removes emojis
    - Removes punctuation (keeps alphanumerics and whitespace)
    - Collapses multiple spaces
    """

    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = URL_RE.sub("", text)
    text = EMOJI_RE.sub("", text)
    text = PUNCT_RE.sub("", text)
    text = " ".join(text.split())
    return text
