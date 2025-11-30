import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import sys

# -----------------------------
# Environment & Imports
# -----------------------------
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

try:
    from ultralytics import YOLO
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Brain Tumor Detection",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Full Modern CSS Styling
# -----------------------------
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }

    /* Headers */
    .main-header {
        font-size: 3.8rem;
        background: linear-gradient(135deg, #ffffff, #e0e7ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: 900;
        margin-bottom: 0.5rem;
        text-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #e0e7ff;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 300;
    }

    /* Info Cards */
    .info-box {
        background: rgba(255, 255, 255, 0.97);
        backdrop-filter: blur(12px);
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        margin: 20px 0;
        color: #333;
    }

    /* Modern Upload Area */
    .upload-area {
        background: rgba(255, 255, 255, 0.97);
        backdrop-filter: blur(12px);
        border-radius: 24px;
        padding: 50px 20px;
        text-align: center;
        border: 2px dashed #764ba2;
        transition: all 0.4s ease;
        margin: 30px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    }
    .upload-area:hover {
        border-color: #667eea;
        background: rgba(255, 255, 255, 1);
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.15);
    }
    .upload-icon {
        font-size: 5rem;
        margin-bottom: 15px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .upload-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #333;
        margin-bottom: 10px;
    }
    .upload-hint {
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }
    .supported-text {
        font-size: 0.95rem;
        color: #777;
        font-style: italic;
    }

    /* Result Cards */
    .result-box {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
    }

    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border-radius: 30px !important;
        padding: 12px 30px !important;
        font-weight: 600 !important;
        border: none !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.markdown('<div class="main-header">🧠 Brain Tumor Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered MRI Analysis using YOLOv8</div>', unsafe_allow_html=True)

# -----------------------------
# Info Section
# -----------------------------
with st.container():
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("About This App")
        st.write("""
        - Real-time brain tumor detection from MRI scans  
        - Powered by **YOLOv8** – state-of-the-art object detection  
        - Confidence scores & visual bounding boxes  
        - Designed for research and educational use
        """)
    with c2:
        st.subheader("How to Use")
        st.write("""
        1. Upload a brain MRI image (JPG/PNG)  
        2. AI analyzes it in seconds  
        3. View results with highlighted tumor regions  
        """)
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Load YOLO Model
# -----------------------------
@st.cache_resource
def load_model():
    paths = ["brain_tumor_detection.pt", "./brain_tumor_detection.pt", "models/brain_tumor_detection.pt"]
    for p in paths:
        if os.path.exists(p):
            return YOLO(p)
    return None

model = load_model()

# -----------------------------
# MODERN UPLOAD SECTION
# -----------------------------
st.markdown("---")
st.markdown("<h2 style='text-align:center; color:white;'>Upload Your Brain MRI Scan</h2>", unsafe_allow_html=True)

st.markdown("""
<div class="upload-area">
    <div class="upload-title">Drop your MRI image here</div>
    <div class="upload-hint">or click to select from your device</div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file:
    st.markdown(f"""
    <div style="text-align:center; padding: 20px; background: linear-gradient(135deg, #00C851, #007E33); 
                border-radius: 16px; color: white; font-weight: 600; margin: 20px 0; font-size:1.2rem;">
        Successfully uploaded: <strong>{uploaded_file.name}</strong> ({uploaded_file.size / (1024*1024):.1f} MB)
    </div>
    """, unsafe_allow_html=True)

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI – Ready for Analysis", width='stretch')

# -----------------------------
# AI Processing & Results
# -----------------------------
if uploaded_file and model:
    try:
        with st.spinner("AI is analyzing your MRI scan..."):
            img_array = np.array(image)
            results = model(img_array)[0]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Original Image")
            st.image(image, width='stretch')

        with col2:
            # -----------------------------
            # Draw bounding boxes + mask
            # -----------------------------
            overlay = img_array.copy()  # Copy to apply transparent mask

            tumor_count = 0
            other_count = 0

            for box, score, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                label = "Tumor" if int(cls) == 0 else "Abnormal"
                confidence = float(score)

                # Color coding based on detection type
                if label == "Tumor":
                    color = (0, 0, 255)  # Red for tumors (BGR format)
                    tumor_count += 1
                else:
                    color = (0, 165, 255)  # Orange for other findings (BGR format)
                    other_count += 1

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
            if len(results.boxes) > 0:
                st.markdown("#### Detection Results with Mask")
                st.image(final_img, caption="AI Detection with Transparent Mask Overlay", width='stretch')

                st.markdown("### Analysis Summary")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Tumors Detected", tumor_count)
                    st.markdown('</div>', unsafe_allow_html=True)
                with c2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Other Findings", other_count)
                    st.markdown('</div>', unsafe_allow_html=True)
                with c3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Total Detections", len(results.boxes))
                    st.markdown('</div>', unsafe_allow_html=True)

                with st.expander("Detailed Report"):
                    st.write(f"- **Tumor Regions:** {tumor_count}")
                    st.write(f"- **Other Findings:** {other_count}")
                    st.write(f"- **Total Detections:** {len(results.boxes)}")
                    st.write("- **Red boxes** = Tumor regions (high priority)")
                    st.write("- **Orange boxes** = Other abnormalities")
                    st.write("- **Transparent overlay** = Highlighted detection areas")
                    st.info("Always consult a doctor for official diagnosis.")

            else:
                st.success("No tumors or abnormalities detected!")
                st.image(image, caption="Clean MRI Scan", width='stretch')

    except Exception as e:
        st.error(f"Error during processing: {e}")

elif uploaded_file and not model:
    st.error("Model `brain_tumor_detection.pt` not found. Please add it to your project root.")

# -----------------------------
# Sidebar Information
# -----------------------------
with st.sidebar:
    st.markdown("### 🏥 System Status")
    
    # Model status
    if model is not None:
        st.success("🟢 **AI Model:** Active & Ready")
    else:
        st.error("🔴 **AI Model:** Not Loaded")
    
    st.markdown("---")
    st.markdown("### 🔧 Technology Stack")
    st.write("""
    **Powered By:**
    - 🤖 **YOLOv8** - Advanced object detection
    - 🌐 **Streamlit** - Modern web interface
    - 🖼️ **OpenCV** - Image processing
    - 🧠 **PyTorch** - Deep learning framework
    """)
    
    st.markdown("---")
    st.markdown("### 📏 Specifications")
    st.write("""
    **Supported Formats:**
    • JPEG, JPG, PNG
    
    **Maximum File Size:**
    • 200MB per image
    
    **Processing Time:**
    • 2-10 seconds typical
    
    **Detection Accuracy:**
    • High precision mode
    """)
    
    st.markdown("---")
    st.markdown("### ℹ️ Medical Disclaimer")
    st.info("""
    **Important Notice:**
    This tool is for research and educational purposes only. Always consult qualified healthcare professionals for medical diagnosis and treatment decisions.
    """)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:rgba(255,255,255,0.9); padding:20px;'>
    <h3>Brain Tumor Detection System</h3>
    <p>Powered by YOLOv8 • Built with Streamlit • Research Use Only</p>
    <small>Always consult a qualified medical professional for diagnosis.</small>
</div>
""", unsafe_allow_html=True)
