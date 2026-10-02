"""Small reusable pieces of the Streamlit interface."""
from pathlib import Path

import streamlit as st

DISCLAIMER = (
    "This application is developed for educational and research purposes only "
    "and is not intended for medical diagnosis or clinical decision-making."
)

# Names shown to the user (the model itself uses the short folder names)
DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary Tumor",
}


def pretty(class_name):
    return DISPLAY_NAMES.get(class_name, class_name)


def load_css():
    css_path = Path(__file__).parent / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


def show_disclaimer():
    st.warning(DISCLAIMER)


def show_prediction(result):
    """Predicted class, confidence and the probability of every class."""
    col1, col2 = st.columns(2)
    col1.metric("Predicted class", pretty(result["predicted_class"]))
    col2.metric("Confidence", f"{result['confidence'] * 100:.1f}%")
    st.caption(
        "Confidence is the model's own score, not the chance of being correct. "
        "The model can be confidently wrong."
    )

    st.markdown("**Class probabilities**")
    for name, p in sorted(result["probabilities"].items(), key=lambda kv: -kv[1]):
        label_col, bar_col = st.columns([2, 3])
        label_col.write(f"{pretty(name)}: {p * 100:.1f}%")
        bar_col.progress(min(1.0, max(0.0, float(p))))