"""Streamlit app: Brain Tumor Detection from MRI Images (educational project).

Run from the project folder with:   streamlit run app/app.py
"""
import json
import sys
from io import BytesIO
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))   # so "from src import ..." works
sys.path.insert(0, str(APP_DIR))        # so "import components" works

import pandas as pd
import streamlit as st
from PIL import Image

import components
from src import config
from src.explainability.gradcam import make_gradcam_heatmap, overlay_heatmap
from src.predict import load_best_model, load_class_names, predict_image

st.set_page_config(page_title="Brain Tumor Detection from MRI Images",
                   page_icon="🧠", layout="wide")
components.load_css()


@st.cache_resource
def get_model():
    """Load the model once and reuse it (loading is slow)."""
    return load_best_model(), load_class_names()


# ---------------------------------------------------------------- pages
def home_page():
    st.title("Brain Tumor Detection from MRI Images")
    st.write(
        "This project classifies a brain MRI image into one of four classes: "
        "**Glioma**, **Meningioma**, **Pituitary Tumor** or **No Tumor**. "
        "It uses deep learning (transfer learning with a CNN) and Grad-CAM to "
        "visualise which image regions influenced the model's decision."
    )
    st.write("Use the menu on the left: upload an image under **MRI Prediction**, "
             "then look at **Explainability** for the Grad-CAM view.")
    components.show_disclaimer()


def prediction_page():
    st.header("MRI Prediction")
    uploaded = st.file_uploader("Upload an MRI image (JPG or PNG)", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Upload an image to begin.")
        return

    file_bytes = uploaded.getvalue()
    file_id = f"{uploaded.name}-{len(file_bytes)}"

    # Check that the file really is an image before using it
    try:
        Image.open(BytesIO(file_bytes)).verify()
    except Exception:
        st.error("This file could not be read as an image. Please upload a valid JPG or PNG file.")
        return

    left, right = st.columns(2)
    left.image(file_bytes, caption="Uploaded image")

    with right:
        if st.button("Run prediction", type="primary"):
            with st.spinner("Running the model..."):
                model, class_names = get_model()
                result = predict_image(model, file_bytes, class_names)
                heatmap, _, _ = make_gradcam_heatmap(model, result["image"])
            st.session_state["file_id"] = file_id
            st.session_state["result"] = result
            st.session_state["heatmap"] = heatmap

        # Show results only if they belong to the image that is uploaded now
        if st.session_state.get("file_id") == file_id:
            components.show_prediction(st.session_state["result"])
            st.success("Open the Explainability page to see the Grad-CAM view.")
    components.show_disclaimer()


def explainability_page():
    st.header("Explainability (Grad-CAM)")
    if "result" not in st.session_state:
        st.info("Run a prediction on the MRI Prediction page first.")
        return

    result = st.session_state["result"]
    heatmap_rgb, overlay = overlay_heatmap(result["image"], st.session_state["heatmap"])
    col1, col2, col3 = st.columns(3)
    col1.image(result["image"].astype("uint8"), caption="Original MRI")
    col2.image(heatmap_rgb, caption="Grad-CAM heatmap")
    col3.image(overlay, caption="Overlay")

    st.write(f"Predicted class: **{components.pretty(result['predicted_class'])}**")
    st.info(
        "Grad-CAM is a model-explanation visualisation. Warm colours mark image regions "
        "that influenced the model's prediction most. It does NOT show where a tumor "
        "really is, and a highlighted area is not a medical finding."
    )
    components.show_disclaimer()


def model_information_page():
    st.header("Model Information")
    metadata_path = config.MODELS_DIR / "model_metadata.json"
    if not metadata_path.exists():
        st.error("model_metadata.json not found. Run notebook 05 first.")
        return
    meta = json.loads(metadata_path.read_text())

    col1, col2, col3 = st.columns(3)
    col1.metric("Selected model", meta["selected_model"])
    col2.metric("Input size", f"{meta['input_size'][0]} x {meta['input_size'][1]}")
    col3.metric("Number of classes", meta["num_classes"])

    st.subheader("Performance on the official test set")
    col1, col2 = st.columns(2)
    col1.metric("Accuracy", f"{meta['test_accuracy'] * 100:.1f}%")
    col2.metric("Macro F1-score", f"{meta['test_f1_macro']:.3f}")
    st.caption(f"The model was chosen by: {meta['selection_rule']}.")

    comparison_path = config.METRICS_DIR / "model_comparison.csv"
    if comparison_path.exists():
        st.subheader("All models compared")
        st.dataframe(pd.read_csv(comparison_path), hide_index=True)

    st.subheader("Technologies used")
    st.write("Python, TensorFlow/Keras (ResNet50, EfficientNetB0, custom CNN), "
             "scikit-learn, pandas, NumPy, Matplotlib, Pillow, Streamlit.")

    with st.expander("Known limitations (please read)"):
        st.markdown(
            "- Image size and format differ between classes in this dataset, so a model "
            "can partly learn them as shortcuts instead of real medical features.\n"
            "- In the official split, some kinds of scans (for example differently sized "
            "glioma images) appear only in the test set.\n"
            "- Meningioma is the hardest class for every model tested.\n"
            "- Models were trained on CPU for only a few epochs.\n"
            "- The model can be confidently wrong."
        )
    components.show_disclaimer()


def disclaimer_page():
    st.header("Disclaimer")
    components.show_disclaimer()
    st.write("This is a student project. Do not use it to make decisions about "
             "anyone's health. If you have a medical concern, speak to a qualified doctor.")


# ---------------------------------------------------------------- navigation
PAGES = {
    "Home": home_page,
    "MRI Prediction": prediction_page,
    "Explainability": explainability_page,
    "Model Information": model_information_page,
    "Disclaimer": disclaimer_page,
}
choice = st.sidebar.radio("Navigate", list(PAGES))
st.sidebar.caption("Educational project - not a medical device.")
PAGES[choice]()