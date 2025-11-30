import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

# Set page config
st.set_page_config(
    page_title="Brain Tumor Detection",
    layout="centered",
    page_icon="🧠"
)

# Simple background
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main-title {
        color: white;
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🧠 Brain Tumor Detection</div>', unsafe_allow_html=True)

# Info section
with st.container():
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.write("Upload a brain MRI image to detect potential tumors using AI.")
    st.markdown('</div>', unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        # Try to find model file
        model_paths = [
            "brain_tumor_detection.pt",
            "model.pt", 
            "./brain_tumor_detection.pt"
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                from ultralytics import YOLO
                model = YOLO(path)
                st.success(f"✅ Model loaded from {path}")
                return model
        
        st.warning("⚠️ Model file not found. Running in demo mode.")
        return "demo"
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return "error"

model = load_model()

# File upload
st.markdown("---")
uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Process image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Original Image", use_container_width=True)
    
    with col2:
        if model == "demo":
            # Demo detection
            demo_img = img_array.copy()
            h, w = demo_img.shape[:2]
            cv2.rectangle(demo_img, (w//4, h//4), (3*w//4, 3*h//4), (0, 255, 0), 3)
            cv2.putText(demo_img, "Demo Detection", (w//4, h//4-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            st.image(demo_img, caption="Demo Detection", use_container_width=True)
            st.info("Add 'brain_tumor_detection.pt' to your repository for real detection.")
            
        elif model != "error":
            # Real detection
            with st.spinner("Analyzing image..."):
                results = model(img_array)[0]
            
            result_img = img_array.copy()
            detections = 0
            
            if hasattr(results, 'boxes') and results.boxes is not None:
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    label = "Tumor" if class_id == 0 else "Abnormality"
                    color = (0, 0, 255) if label == "Tumor" else (255, 165, 0)
                    
                    # Draw detection
                    cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(result_img, f"{label} {confidence:.2f}", 
                               (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    detections += 1
            
            st.image(result_img, caption="Detection Results", use_container_width=True)
            
            if detections > 0:
                st.success(f"Found {detections} detection(s)")
            else:
                st.info("No abnormalities detected")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: white;'>
        <p>Brain Tumor Detection App • Built with Streamlit & YOLO</p>
    </div>
    """,
    unsafe_allow_html=True
)
