import streamlit as st
import sys
import os

# Add debug info and set environment variables early
st.write(f"Python version: {sys.version}")

# Set environment variables for compatibility
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

try:
    import cv2
    st.success("✅ OpenCV imported successfully")
except ImportError as e:
    st.error(f"❌ OpenCV import failed: {e}")

try:
    import numpy as np
    st.success("✅ NumPy imported successfully")
except ImportError as e:
    st.error(f"❌ NumPy import failed: {e}")

try:
    from PIL import Image
    st.success("✅ PIL imported successfully")
except ImportError as e:
    st.error(f"❌ PIL import failed: {e}")

try:
    import base64
    st.success("✅ base64 imported successfully")
except ImportError as e:
    st.error(f"❌ base64 import failed: {e}")

try:
    from ultralytics import YOLO
    st.success("✅ Ultralytics YOLO imported successfully")
except ImportError as e:
    st.error(f"❌ Ultralytics import failed: {e}")
    st.stop()

# -----------------------------
# Function to set background image (updated for cloud deployment)
# -----------------------------
def add_bg_from_local(image_file):
    try:
        # For Streamlit Cloud, use relative paths
        if os.path.exists(image_file):
            with open(image_file, "rb") as img:
                encoded = base64.b64encode(img.read()).decode()
            st.markdown(
                f"""
                <style>
                .stApp {{
                    background-image: url("data:image/jpeg;base64,{encoded}");
                    background-size: cover;
                    background-attachment: fixed;
                }}
                </style>
                """,
                unsafe_allow_html=True
            )
        else:
            # Use a simple color background if image not found
            st.markdown(
                """
                <style>
                .stApp {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                }
                </style>
                """,
                unsafe_allow_html=True
            )
    except Exception as e:
        st.warning(f"Could not load background image: {e}")
        # Fallback to color background
        st.markdown(
            """
            <style>
            .stApp {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }
            </style>
            """,
            unsafe_allow_html=True
        )

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Brain Tumor Detection",
    layout="centered",
    page_icon="🧠"
)

# -----------------------------
# Add background - use relative path for cloud
# -----------------------------
add_bg_from_local("background.jpg")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: white;
        text-align: center;
        text-shadow: 2px 2px 4px #000000;
    }
    .sub-header {
        font-size: 1.5rem;
        color: white;
        text-align: center;
        text-shadow: 1px 1px 2px #000000;
    }
    .info-box {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🧠 Brain Tumor Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Using YOLO Deep Learning Model</div>', unsafe_allow_html=True)

# -----------------------------
# Load YOLO model with error handling
# -----------------------------
@st.cache_resource
def load_model():
    try:
        # Try multiple possible model paths for cloud deployment
        possible_paths = [
            "brain_tumor_detection.pt",
            "model/brain_tumor_detection.pt", 
            "models/brain_tumor_detection.pt",
            "./brain_tumor_detection.pt"
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                st.success(f"✅ Model found at: {path}")
                break
        
        if model_path is None:
            st.error("❌ Model file not found. Please ensure 'brain_tumor_detection.pt' is in your repository.")
            return None
        
        # Load the model
        model = YOLO(model_path)
        st.success("✅ Model loaded successfully!")
        return model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Load the model
model = load_model()

# -----------------------------
# Information section
# -----------------------------
with st.container():
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.subheader("ℹ️ About this App")
    st.write("""
    This application uses a YOLO (You Only Look Once) deep learning model to detect brain tumors in MRI images.
    Upload an MRI scan to get instant detection results with bounding boxes and confidence scores.
    """)
    
    st.subheader("📝 Instructions:")
    st.write("""
    1. Upload a brain MRI image (JPG, JPEG, or PNG format)
    2. The model will automatically analyze the image
    3. View the detection results with highlighted regions
    4. Positive detections indicate potential tumor regions
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# File uploader
# -----------------------------
st.markdown("---")
st.subheader("📤 Upload MRI Image")

uploaded_file = st.file_uploader(
    "Choose an MRI image file", 
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG"
)

if uploaded_file is not None and model is not None:
    try:
        # Open and convert image
        img = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(img)

        # Display original image
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="📷 Original MRI Image", use_container_width=True)

        # -----------------------------
        # Run YOLO inference
        # -----------------------------
        with st.spinner("🔍 Analyzing image for tumor detection..."):
            results = model(img_array)[0]

        # Create copies for processing
        detected_img = img_array.copy()
        overlay = img_array.copy()

        # Check if any detections were made
        if len(results.boxes) > 0:
            st.success(f"✅ Detected {len(results.boxes)} region(s) of interest")

            for box, score, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                label = "Tumor" if int(cls) == 0 else "Abnormal"
                confidence = float(score)

                # Color coding: Red for tumor, Orange for other abnormalities
                color = (255, 0, 0) if label == "Tumor" else (255, 165, 0)

                # Draw rectangle border
                cv2.rectangle(detected_img, (x1, y1), (x2, y2), color, 3)

                # Put label with confidence
                label_text = f"{label} {confidence:.2f}"
                cv2.putText(
                    detected_img,
                    label_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

                # Create transparent overlay
                overlay[y1:y2, x1:x2] = cv2.addWeighted(
                    overlay[y1:y2, x1:x2], 0.4,
                    np.full_like(overlay[y1:y2, x1:x2], color), 0.6, 0
                )

            # Merge overlay with original
            alpha = 0.6
            final_img = cv2.addWeighted(overlay, alpha, detected_img, 1 - alpha, 0)

            # Display result
            with col2:
                st.image(final_img, caption="🎯 Detection Results", use_container_width=True)

            # Results summary
            st.markdown("---")
            st.subheader("📊 Detection Summary")
            
            tumor_count = sum(1 for cls in results.boxes.cls if int(cls) == 0)
            other_count = len(results.boxes.cls) - tumor_count
            
            col3, col4 = st.columns(2)
            with col3:
                st.metric("Tumor Regions Detected", tumor_count)
            with col4:
                st.metric("Other Findings", other_count)
                
        else:
            st.info("ℹ️ No abnormalities detected in the MRI image.")
            with col2:
                st.image(img, caption="✅ No Detections - Clear MRI", use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
        st.info("Please try uploading a different image or check the file format.")

elif uploaded_file is not None and model is None:
    st.error("❌ Cannot process image - model failed to load. Please check the model file.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: white;'>
        <p>Built with Streamlit • YOLO Deep Learning • Medical Imaging</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Debug information (collapsible)
with st.expander("🔧 Debug Information"):
    st.write(f"Python executable: {sys.executable}")
    st.write(f"Current working directory: {os.getcwd()}")
    st.write(f"Files in directory: {os.listdir('.')}")
    if 'model' in locals() and model is not None:
        st.write("Model: Loaded successfully")
    else:
        st.write("Model: Not loaded")
