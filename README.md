# 🧠 Brain Tumor Detection using YOLOv8

<div align="center">

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=OpenCV&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-00FF00?style=for-the-badge&logo=ultralytics&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

*A powerful web application for detecting brain tumors in MRI scans using state-of-the-art YOLOv8 deep learning model*

[![Demo](https://img.shields.io/badge/🚀-Live_Demo-blue?style=for-the-badge)](https://brain-tumor-detection-342dgceyrprgskelub8cyj.streamlit.app/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

</div>

## 🌟 Overview

Brain Tumor Detection is an intelligent web application that leverages the power of YOLOv8 (You Only Look Once) to automatically detect and localize brain tumors in MRI images. This tool provides medical professionals and researchers with an accessible interface for rapid preliminary analysis of brain MRI scans.

### 🎯 Key Features

- **🔍 Accurate Detection**: Powered by YOLOv8 for precise tumor localization
- **🎨 Visual Annotations**: Bounding boxes with semi-transparent masks
- **📊 Confidence Scoring**: Real-time prediction confidence levels
- **🌐 Web Interface**: User-friendly Streamlit-based web app
- **⚡ Real-time Processing**: Instant analysis with visual feedback
- **📱 Responsive Design**: Works seamlessly on different devices

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/brain-tumor-detection.git
   cd brain-tumor-detection
   ```

2. **Create virtual environment** (Recommended)
   ```bash
   python -m venv venv
   
   # Linux/Mac
   source venv/bin/activate
   
   # Windows
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the YOLO model**
   - Place your trained `brain_tumor_detection.pt` file in the project root directory

### Running the Application

```bash
streamlit run main.py
```

Open your browser and navigate to `http://localhost:8501` to access the application.

## 📁 Project Structure

```
brain-tumor-detection/
├── main.py                      # Main Streamlit application
├── brain_tumor_detection.pt     # YOLOv8 trained model (not included)
├── requirements.txt             # Python dependencies
├── runtime.txt                  # Python version specification
├── packages.txt                 # System dependencies
├── .streamlit/
│   └── config.toml             # Streamlit configuration
└── README.md                   # Project documentation
```

## 🛠️ Technical Details

### Model Architecture

This application uses **YOLOv8** (You Only Look Once version 8), which provides:
- **Real-time object detection** capabilities
- **High accuracy** in medical image analysis
- **Efficient processing** with optimized neural network architecture

### Detection Process

1. **Image Preprocessing**: MRI images are normalized and prepared for inference
2. **YOLOv8 Inference**: Model processes the image and identifies potential tumor regions
3. **Post-processing**: Non-maximum suppression filters overlapping detections
4. **Visualization**: Bounding boxes and masks are overlaid on the original image

### Supported Formats

- **Image Formats**: JPG, JPEG, PNG
- **MRI Types**: T1-weighted, T2-weighted, FLAIR sequences

## 💻 Usage Guide

### Step-by-Step Process

1. **Launch the Application**
   - Run `streamlit run main.py`
   - The web interface will open automatically

2. **Upload MRI Image**
   - Click "Upload MRI Image" button
   - Select your brain MRI scan from your device

3. **View Results**
   - Original image displays on the left
   - Annotated results show on the right with:
     - Bounding boxes around detected regions
     - Color-coded masks highlighting tumor areas
     - Confidence scores for each detection
     - Classification labels ("Tumor" or "Normal")

### Interpretation of Results

- **🔴 Red Boxes**: High-confidence tumor detections
- **🟠 Orange Boxes**: Other abnormalities or lower-confidence findings
- **Confidence Score**: Percentage indicating detection certainty (0-100%)
- **Mask Overlay**: Semi-transparent colored regions highlighting affected areas

## 🏗️ Deployment

### Local Deployment

Follow the installation steps above for local deployment.

### Cloud Deployment (Streamlit Cloud)

1. Fork this repository
2. Visit [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub account
4. Deploy the app by selecting your forked repository
5. Ensure all required files are present in the repository

## 📊 Model Performance

The YOLOv8 model used in this application has been trained on diverse MRI datasets and demonstrates:

- **High Precision**: Accurate tumor localization
- **Fast Inference**: Real-time processing capabilities
- **Robust Performance**: Consistent across different MRI machines and protocols

## 🔧 Configuration

### Customizing Detection Parameters

Modify the following in `main.py` for different use cases:

```python
# Confidence threshold (0-1)
conf_threshold = 0.25

# IoU threshold for non-maximum suppression
iou_threshold = 0.45

# Detection classes
class_names = {0: "Tumor", 1: "Normal"}
```

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

1. **Report Issues**: Found a bug? Create an issue with detailed information
2. **Feature Requests**: Suggest new features or improvements
3. **Code Contributions**: Submit pull requests for bug fixes or enhancements
4. **Documentation**: Help improve documentation and examples

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a pull request

## ⚠️ Important Disclaimer

**Medical Disclaimer**: This application is intended for research and educational purposes only. It should not be used as a substitute for professional medical diagnosis, advice, or treatment. Always consult qualified healthcare professionals for medical concerns.

- This tool provides **preliminary analysis** only
- **Not FDA-approved** for clinical use
- Results should be **verified by medical professionals**
- Use at your own risk

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ultralytics** for the YOLOv8 framework
- **Streamlit** for the amazing web app framework
- **OpenCV** for computer vision capabilities
- **PyTorch** for deep learning infrastructure
- The medical imaging research community for continuous advancements

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/your-username/brain-tumor-detection/issues) page
2. Create a new issue with detailed description
3. Provide relevant error logs and system information

---

<div align="center">

**Made with ❤️ for the medical research community**

*Contributions welcome! Help us make medical AI more accessible.*

[⭐ Star this repo] • [🐛 Report Issues] • [💡 Suggest Features]

</div>
