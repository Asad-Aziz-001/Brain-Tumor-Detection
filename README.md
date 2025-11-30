# 🧠 Brain Tumor Detection using YOLO and Streamlit

A web app to detect brain tumors from MRI images using YOLOv8 and visualize predictions with bounding boxes and semi-transparent masks. Built with Python, OpenCV, and Streamlit.

# Features

Upload MRI images (JPG, PNG) for analysis.

Detect brain tumors with YOLOv8.

Display bounding boxes with transparent masks highlighting detected regions.

Show prediction confidence and label ("Positive" or "Negative").

Interactive web interface powered by Streamlit.

# Installation

## Clone the repository:

git clone https://github.com/your-username/brain-tumor-detection.git
cd brain-tumor-detection


Create a virtual environment (optional but recommended):

python -m venv venv
# Activate environment:
# Linux/Mac
source venv/bin/activate
# Windows
venv\Scripts\activate


# Install dependencies:

pip install -r requirements.txt


Download your YOLO model (brain_tumor_detection.pt) and place it in the project folder.

Usage

Run the Streamlit app:

streamlit run main.py


Open the provided URL (usually http://localhost:8501) in your browser and upload an MRI image to see predictions.

# Project Structure
brain-tumor-detection/
 │
├── main.py                    # Streamlit app
├── brain_tumor_detection.pt   # YOLO trained model
├── background.jpg             # Background image for the app
├── requirements.txt           # Python dependencies
 └── README.md

# License

This project is licensed under the MIT License.
