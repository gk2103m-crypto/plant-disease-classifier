---
title: Plant Disease Classifier
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
license: mit
---

# 🌿 Plant Disease Classifier (AI-Based Diagnostic Engine)

An enterprise-grade, full-stack AI pathology classification and remediation system. This engine leverages a PyTorch deep learning model to diagnose plant diseases from specimen imagery and utilizes the Google Gemini API to provide automated, professional treatment strategies.

## 📊 Performance

- **Test Accuracy:** 97.72%
- **Training Accuracy:** 96.96%
- **Validation Accuracy:** 97.54%
- **Disease Classes:** 38
- **Dataset:** 54,305 images

## 🏗️ Architecture

- **Vision Model:** MobileNetV2 (ImageNet pre-trained) with Custom Classifier Head (1280 → 512 → 38)
- **Remediation Engine:** Google Gemini AI with local fallback generation mechanism.
- **Backend API:** FastAPI (Asynchronous processing)
- **Frontend UI:** Vanilla JS, HTML5, Tailwind CSS (Glassmorphism Deep-Tech Theme)
- **Database:** SQLite3 for persistent diagnostic logging

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

````

### Local Testing (Full-Stack Web App)

Start the enterprise FastAPI server:

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Open your browser and navigate to `http://localhost:8000` to access the interactive web dashboard.

## 🧠 Model Details

### Transfer Learning Approach

- Froze early feature extraction layers
- Fine-tuned last 2 feature blocks
- Custom classification head for 38 diseases
- Data augmentation: rotation, flip, affine, color jitter

### Supported Diseases (38 Categories)

- **Apple:** Cedar Apple Rust, Black Rot, Scab
- **Potato:** Early Blight, Late Blight
- **Tomato:** Septoria Leaf Spot, Target Spot, Powdery Mildew
- **Strawberry:** Leaf Scorch
- _And 34 more diseases..._

### Training Details

- **Optimizer:** Adam (lr=0.001)
- **Loss Function:** CrossEntropyLoss
- **Batch Size:** 32
- **Epochs:** 10
- **Early Stopping:** Patience 5
- **Training Time:** 2.5 hours (GPU)
- **Dataset Split:** 64% train, 15% validation, 20% test

## 💻 Usage (API / Scripting)

### Python Script Inference

```python
from inference import PlantDiseaseClassifier

clf = PlantDiseaseClassifier('best_plant_model.pth', 'class_names.json')
disease, confidence = clf.predict('path/to/image.jpg')

print(f"Disease: {disease}")
print(f"Confidence: {confidence:.2f}%")
```

## 🛠️ Technical Stack

- **Language:** Python 3.10+
- **Deep Learning:** PyTorch 2.12, TorchVision 0.27
- **LLM Integration:** Google Generative AI (Gemini)
- **Backend Framework:** FastAPI, Uvicorn, Pydantic
- **Frontend UI:** HTML5, Tailwind CSS, JavaScript
- **Image Processing:** Pillow 12.2
- **Deployment:** Docker

## 📁 Project Structure

```text
plant-disease-classifier/
├── main.py                # FastAPI backend & Frontend HTML template
├── inference.py           # Prediction class
├── model.py               # Model architecture
├── train.py               # Training script
├── test_model.py          # Testing script
├── best_plant_model.pth   # Pre-trained weights (11.5 MB)
├── class_names.json       # 38 disease labels
├── requirements.txt       # Python dependencies
├── Dockerfile             # Containerization instructions
└── README.md              # Documentation
```

## 📈 Results and Insights

- **Training Metrics:** Started with 0% accuracy, trained to 96.96%. Validation accuracy peaked at 97.54%. Test accuracy: 97.72%.
- **Error Analysis:** Most errors occur between visually similar diseases. Class imbalance handled by data augmentation.
- **Known Limitations:** Domain Shift (Model trained on controlled PlantVillage dataset). Real-world images with different angles or lighting may show lower confidence.

## ☁️ Deployment

**Cloud (Hugging Face Spaces):** This project is fully dockerized and deployed on Hugging Face Spaces. The application runs on a standalone Docker container exposing port 7860.

---

_Built with PyTorch & FastAPI | Integrated with Google Gemini | Deployed via Docker_

```



````
