# Plant Disease Classifier

Deep learning model for automated plant leaf disease detection using MobileNetV2 transfer learning.

## Performance

- Test Accuracy: 97.72%
- Training Accuracy: 96.96%
- Validation Accuracy: 97.54%
- Disease Classes: 38
- Dataset: 54,305 images

## Architecture

- Base Model: MobileNetV2 (ImageNet pre-trained)
- Custom Classifier Head: 1280 → 512 → 38
- Framework: PyTorch
- Input Size: 224x224 pixels
- Optimization: Adam (lr=0.001)

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Local Testing

```bash
python -c "from inference import PlantDiseaseClassifier; clf = PlantDiseaseClassifier('best_plant_model.pth', 'class_names.json'); disease, conf = clf.predict('image.jpg'); print(f'{disease}: {conf:.2f}%')"
```

### Interactive UI (Gradio)

```bash
python app.py
```

Open browser to http://localhost:7860

## Model Details

### Transfer Learning Approach

- Froze early feature extraction layers
- Fine-tuned last 2 feature blocks
- Custom classification head for 38 diseases
- Data augmentation: rotation, flip, affine, color jitter

### Supported Diseases

38 plant disease categories including:

- Apple (Cedar Apple Rust, Black Rot, Scab)
- Potato (Early Blight, Late Blight)
- Tomato (Septoria Leaf Spot, Target Spot, Powdery Mildew)
- Strawberry (Leaf Scorch)
- And 34 more diseases

### Training Details

- Optimizer: Adam
- Loss Function: CrossEntropyLoss
- Batch Size: 32
- Epochs: 10
- Early Stopping: Patience 5
- Training Time: 2.5 hours (GPU)
- Dataset Split: 64% train, 15% validation, 20% test

## Usage

### Python Script

```python
from inference import PlantDiseaseClassifier

clf = PlantDiseaseClassifier('best_plant_model.pth', 'class_names.json')
disease, confidence = clf.predict('path/to/image.jpg')

print(f"Disease: {disease}")
print(f"Confidence: {confidence:.2f}%")
```

### Batch Processing

```python
import os
from inference import PlantDiseaseClassifier

clf = PlantDiseaseClassifier('best_plant_model.pth', 'class_names.json')

for image in os.listdir('images_folder'):
    disease, conf = clf.predict(f'images_folder/{image}')
    print(f"{image}: {disease} ({conf:.2f}%)")
```

## Model Performance Analysis

### Accuracy by Image Source

- PlantVillage dataset images: 97.72% accuracy
- High confidence predictions (>90%): Reliable
- Medium confidence (80-90%): Generally accurate
- Low confidence (<80%): Requires verification

### Known Limitations

Domain Shift: Model trained on controlled PlantVillage dataset. Real-world images with different angles, lighting, or image quality may show lower confidence. Predictions remain accurate but confidence scores vary.

Confidence Variation: Similar-looking diseases may produce comparable confidence scores. In such cases, verify predictions visually.

## Technical Stack

- Language: Python 3.13
- Deep Learning: PyTorch 2.12
- Computer Vision: TorchVision 0.27
- UI Framework: Gradio 6.14
- Image Processing: Pillow 12.2
- Numerical Computing: NumPy 2.4

## Project Structure

plant-disease-classifier/
├── app.py # Gradio web UI
├── config.py # Configuration constants
├── data_loader.py # Dataset loading utilities
├── inference.py # Prediction class
├── model.py # Model architecture
├── train.py # Training script
├── test_model.py # Testing script
├── best_plant_model.pth # Pre-trained weights (11.5 MB)
├── class_names.json # 38 disease labels
├── requirements.txt # Python dependencies
└── README.md # Documentation

## Results and Insights

### Training Metrics

- Started with 0% accuracy, trained to 96.96%
- Validation accuracy peaked at 97.54%
- Test accuracy: 97.72%
- Minimal overfitting observed
- Learning curve stable after epoch 7

### Error Analysis

- Most errors occur between visually similar diseases
- Class imbalance handled by data augmentation
- Model shows consistent performance across disease categories

## Future Improvements

- Fine-tune on real-world agricultural images
- Implement confidence threshold API
- Add batch prediction endpoint
- Mobile app integration
- Multi-image analysis
- Ensemble methods for improved robustness

## Deployment

### Local

```bash
python app.py
```

### Cloud (Vercel)

Repository is configured for Vercel deployment. Connect GitHub repository to Vercel for automatic deployment.

## License

Dataset: CC-BY-NC-SA-4.0 (PlantVillage Dataset)

## Author

Developed for agricultural AI and computer vision portfolio.

## Contact & Support

For questions or issues, please open an issue in the repository.

---

**Built with PyTorch | Trained on PlantVillage Dataset | Deployed with Gradio**
