"""A Streamlit web app that predicts emotion + polarity from user text."""

import os
from datetime import datetime

import joblib
import streamlit as st

from sentiment_utils import map_polarity, polarity_distribution


EXAMPLE_TEXTS = {
    "Happy / positive": "I feel amazing today, everything is going so well!",
    "Sad / negative": "I am feeling really down, nothing seems to go right.",
    "Angry / negative": "I can't stand this anymore, I'm so frustrated.",
    "Neutral / informational": "I will go to the store later and buy some milk.",
    "Surprise / positive": "Wow! I didn't expect that at all, what a nice surprise!",
}

EMOTION_COLORS = {
    "anger": "#d62728",
    "hate": "#8c564b",
    "sadness": "#9467bd",
    "worry": "#e377c2",
    "boredom": "#7f7f7f",
    "empty": "#7f7f7f",
    "neutral": "#2ca02c",
    "love": "#d62728",
    "happiness": "#2ca02c",
    "relief": "#17becf",
    "fun": "#1f77b4",
    "enthusiasm": "#ff7f0e",
    "surprise": "#bcbd22",
}

POLARITY_COLORS = {
    "positive": "#2ca02c",
    "neutral": "#7f7f7f",
    "negative": "#d62728",
}


@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
    return joblib.load(model_path)


def init_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []


def add_history(text: str, emotion: str, polarity: str, probs: dict):
    st.session_state.history.insert(
        0,
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "text": text,
            "emotion": emotion,
            "polarity": polarity,
            "probabilities": probs,
        },
    )


def _badge(text: str, color: str) -> str:
    return (
        f"<span style='background:{color};color:#fff;padding:6px 12px;border-radius:999px;"
        "font-weight:600;font-size:0.9rem'>{text}</span>"
    )


def _card(title: str, body: str, icon: str = "") -> str:
    return (
        f"<div style='padding:18px;border-radius:16px;background:#ffffff;"
        "box-shadow:0 12px 30px rgba(0,0,0,0.08);margin-bottom:14px;'>"
        f"<div style='font-size:1.1rem;margin-bottom:10px;font-weight:700;'>{icon} {title}</div>"
        f"<div style='color:#444;font-size:0.95rem;line-height:1.5;'>{body}</div>"
        "</div>"
    )


def main():
    st.set_page_config(
        page_title="Sentiment / Emotion Analyzer",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .reportview-container .main {
            background: linear-gradient(135deg, #f7fbff 0%, #e5f2ff 100%);
        }
        .stButton>button {
            background-color: #1f77b4;
            color: white;
        }
        .stTextArea>div>div>textarea {
            font-family: inherit;
        }
        .stTextArea>div>div {
            border-radius: 14px;
        }
        .stTextInput>div>div>input {
            border-radius: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    header_col1, header_col2 = st.columns([2, 1])
    with header_col1:
        st.title("😊 Sentiment + Emotion Analyzer")
        st.write(
            "Enter text in the box below and click **Analyze** to see emotion and polarity predictions."
        )

    with header_col2:
        st.markdown(_card("Quick start", "Paste some text, click analyze, and see the emotion result instantly!", "⚡"), unsafe_allow_html=True)
        st.markdown(
            _card(
                "Pro tip",
                "Use longer sentences for stronger signal, and emojis can help the model detect feelings more clearly.",
                "💡",
            ),
            unsafe_allow_html=True,
        )

    st.sidebar.title("📌 Controls")
    st.sidebar.markdown(
        "Use the example dropdown to auto-fill sample sentences, or clear the history to start fresh."
    )

    if st.sidebar.button("Clear history"):
        st.session_state.history = []

    init_session_state()

    example_key = st.sidebar.selectbox(
        "Example text",
        ["(none)"] + list(EXAMPLE_TEXTS.keys()),
        index=0,
        help="Pick an example to auto-fill the text input.",
    )

    default_text = ""
    if example_key != "(none)":
        default_text = EXAMPLE_TEXTS[example_key]

    user_text = st.text_area("Text to analyze", value=default_text, height=160)

    model = load_model()

    analyze = st.button("Analyze")
    if analyze:
        if not user_text.strip():
            st.warning("Please enter text to analyze.")
        else:
            pred = model.predict([user_text])[0]
            polarity = map_polarity(pred)

            probs = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba([user_text])[0]
                labels = list(model.classes_)
                probs = {
                    "emotion_probs": dict(zip(labels, map(float, proba))),
                    "polarity_probs": polarity_distribution(proba, labels),
                }

            add_history(user_text, pred, polarity, probs)

            emotion_color = EMOTION_COLORS.get(pred, "#444")
            polarity_color = POLARITY_COLORS.get(polarity, "#444")

            emotion_badge = _badge(pred, emotion_color)
            polarity_badge = _badge(polarity, polarity_color)

            result_col, info_col = st.columns([2, 1])
            with result_col:
                st.markdown("## Result")
                st.markdown(
                    f"<div style='padding:18px;border-radius:16px;background:#ffffff;box-shadow:0 12px 30px rgba(0,0,0,0.08);'>"
                    f"<p style='margin:0;font-size:1.1rem;'>Emotion: {emotion_badge}</p>"
                    f"<p style='margin:0;font-size:1.1rem;'>Polarity: {polarity_badge}</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                if probs is not None:
                    st.markdown("### Emotion probabilities")
                    for label, p in sorted(probs["emotion_probs"].items(), key=lambda x: -x[1]):
                        st.write(f"**{label}** — {p:.2%}")
                        st.progress(p)

                    st.markdown("### Polarity probabilities")
                    for label, p in sorted(probs["polarity_probs"].items(), key=lambda x: -x[1]):
                        st.write(f"**{label}** — {p:.2%}")
                        st.progress(p)

            with info_col:
                st.markdown(_card("What's happening?", "The model predicts an emotion label, then maps it to a simplified polarity (positive/neutral/negative)."))
                st.markdown(
                    _card(
                        "API ready",
                        "Run `uvicorn api:app --reload` to expose a /predict endpoint for integration.",
                        "🔌",
                    )
                )

    if st.session_state.history:
        st.markdown("---")
        with st.expander("📜 Prediction history", expanded=True):
            for entry in st.session_state.history:
                st.markdown(
                    f"**{entry['timestamp']}** — {entry['polarity'].capitalize()} ({entry['emotion']})"
                )
                st.write(entry["text"])
                st.markdown("---")


if __name__ == "__main__":
    main()
