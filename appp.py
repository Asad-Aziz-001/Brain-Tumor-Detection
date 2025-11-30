import streamlit as st
import os
import sys

# Set environment variables early
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# Debug info
st.write(f"Python version: {sys.version}")

# Import with comprehensive error handling
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

# Check if ultralytics can be imported
try:
    from ultralytics import YOLO
    st.success("✅ Ultralytics imported successfully")
except ImportError as e:
    st.error(f"❌ Ultralytics import failed: {e}")
    st.stop()

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
        margin-bottom: 1rem;
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

# Load model with error handling
@st.cache_resource
def load_model():
    try:
        # Try different model paths
        model_paths = [
            "brain_tumor_detection.pt",
            "model.pt",
            "./brain_tumor_detection.pt",
            "model/brain_tumor_detection.pt"
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                model = YOLO(path)
                st.success(f"✅ Model loaded from {path}")
                return model
        
        # If no model found, return None for demo mode
        st.warning("⚠️ Model file not found. Please add 'brain_tumor_detection.pt' to your repository.")
        return None
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# Load the model
model = load_model()

# App info
with st.container():
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.write("**Upload a brain MRI image to detect potential tumors using YOLO AI.**")
    st.write("Supported formats: JPG, JPEG, PNG")
    st.markdown('</div>', unsafe_allow_html=True)

# File uploader
st.markdown("---")
uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Process the image
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="📷 Original Image", use_container_width=True)
        
        with col2:
            if model is not None:
                # Real detection
                with st.spinner("🔍 Analyzing image for tumors..."):
                    results = model(img_array)[0]
                
                # Create result image
                result_img = img_array.copy()
                detections = 0
                
                if hasattr(results, 'boxes') and results.boxes is not None:
                    for box in results.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        
                        # Determine label and color
                        label = "Tumor" if class_id == 0 else "Abnormality"
                        color = (0, 0, 255) if label == "Tumor" else (255, 165, 0)
                        
                        # Draw bounding box
                        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 3)
                        
                        # Draw label
                        label_text = f"{label} {confidence:.2f}"
                        cv2.putText(result_img, label_text, 
                                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
                        detections += 1
                
                st.image(result_img, caption="🎯 Detection Results", use_container_width=True)
                
                # Show results summary
                if detections > 0:
                    st.success(f"✅ Found {detections} detection(s)")
                else:
                    st.info("ℹ️ No abnormalities detected")
                    
            else:
                # Demo mode
                demo_img = img_array.copy()
                h, w = demo_img.shape[:2]
                
                # Draw sample detection
                cv2.rectangle(demo_img, (w//4, h//4), (3*w//4, 3*h//4), (0, 255, 0), 3)
                cv2.putText(demo_img, "Sample Detection", (w//4, h//4-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                st.image(demo_img, caption="🔬 Sample Detection", use_container_width=True)
                st.info("💡 This is a demo. Add your model file 'brain_tumor_detection.pt' for real detection.")
                
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: white;'>
        <p>Built with Streamlit & YOLO • Brain Tumor Detection</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Debug information
with st.expander("🔧 Debug Information"):
    st.write(f"Current directory: {os.getcwd()}")
    st.write(f"Files in directory: {[f for f in os.listdir('.') if not f.startswith('.')]}")
    st.write(f"Model loaded: {model is not None}")
