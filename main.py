import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import base64

# -----------------------------
# Function to set background image
# -----------------------------
def add_bg_from_local(image_file):
    if os.path.exists(image_file):
        with open(image_file, "rb") as img:
            encoded = base64.b64encode(img.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpeg;base64,{encoded}");
                background-size: cover;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("Background image not found!")

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Brain Tumor Detection",
    layout="centered",
    page_icon="🧠"
)

# -----------------------------
# Add background
# -----------------------------
add_bg_from_local(r"E:\Projects\Computer VIsion\Brain_Tumor_Detection\background.jpg")

st.title("🧠 Brain Tumor Detection using YOLO")

# -----------------------------
# Load YOLO model
# -----------------------------
MODEL_PATH = r"E:\Projects\Computer VIsion\Brain_Tumor_Detection\brain_tumor_detection.pt"
if os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)
else:
    st.error("Model not found! Make sure the path is correct.")

# -----------------------------
# File uploader
# -----------------------------
uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(img)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    # -----------------------------
    # Run YOLO inference
    # -----------------------------
    results = model(img_array)[0]

    # -----------------------------
    # Draw bounding boxes + mask
    # -----------------------------
    overlay = img_array.copy()  # Copy to apply transparent mask

    for box, score, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
        x1, y1, x2, y2 = map(int, box)
        label = "Positive" if int(cls) == 0 else "Negative"

        # Random color for each detection
        color = tuple(np.random.randint(0, 256, size=3).tolist())

        # Draw rectangle border
        cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 2)

        # Put label
        cv2.putText(
            img_array,
            f"{label} {score:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        # Fill mask inside bounding box with transparency
        overlay[y1:y2, x1:x2] = (overlay[y1:y2, x1:x2] * 0.4 + np.array(color) * 0.6).astype(np.uint8)

    # Merge overlay
    alpha = 0.6  # Transparency factor
    final_img = cv2.addWeighted(overlay, alpha, img_array, 1 - alpha, 0)

    # -----------------------------
    # Show result
    # -----------------------------
    st.image(final_img, caption="Prediction with Mask", use_column_width=True)
