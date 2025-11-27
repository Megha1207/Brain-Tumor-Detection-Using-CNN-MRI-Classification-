import streamlit as st
import requests
from PIL import Image
import numpy as np
import io

st.set_page_config(page_title="Brain Tumor Detector", layout="centered")

FASTAPI_URL = "http://localhost:8000/predict"

st.title("Brain Tumor Detection")
st.write("Upload an MRI scan to classify tumor presence and view Grad-CAM explanation.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded MRI", use_column_width=True)

    # Convert image to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()

    if st.button("Predict"):
        with st.spinner("Processing..."):
            response = requests.post(
                FASTAPI_URL,
                files={"file": ("image.png", img_bytes, "image/png")}
            )

        if response.status_code == 200:
            result = response.json()
            st.subheader(f"Prediction: {result['prediction']}")
            st.write(f"Confidence: {result['confidence']:.3f}")

            st.subheader("Grad-CAM Heatmap")
            heatmap_bytes = bytes(result["heatmap"])
            heatmap_img = Image.open(io.BytesIO(heatmap_bytes))
            st.image(heatmap_img, caption="Grad-CAM", use_column_width=True)
        else:
            st.error("Error connecting to API")
