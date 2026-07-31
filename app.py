import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input,
    decode_predictions,
)

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Image Recognition using MobileNetV2",
    page_icon="🖼️",
    layout="centered",
)

st.title("🖼️ Image Recognition using MobileNetV2")
st.write(
    "Upload an image and the AI model will identify the object and display the Top 5 predictions."
)

# -----------------------------
# Load Model (Cached)
# -----------------------------
@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet")

model = load_model()

# -----------------------------
# Image Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    image = image.resize((224, 224))

    image_array = np.array(image)

    image_array = np.expand_dims(image_array, axis=0)

    image_array = preprocess_input(image_array)

    with st.spinner("Recognizing Image..."):
        predictions = model.predict(image_array, verbose=0)

    results = decode_predictions(predictions, top=5)[0]

    st.success("Prediction Complete!")

    st.subheader("Top 5 Predictions")

    for i, (_, label, probability) in enumerate(results, start=1):
        st.write(
            f"**{i}. {label.replace('_', ' ').title()}** — {probability * 100:.2f}%"
        )

    st.subheader("Confidence Scores")

    confidence = [prob * 100 for _, _, prob in results]
    labels = [label.replace("_", " ").title() for _, label, _ in results]

    chart_data = {
        "Prediction": labels,
        "Confidence (%)": confidence,
    }

    st.bar_chart(
        data=chart_data,
        x="Prediction",
        y="Confidence (%)",
        horizontal=True,
    )

else:
    st.info("📤 Upload an image to begin.")