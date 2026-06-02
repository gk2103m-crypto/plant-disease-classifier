import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from torchvision import transforms
from PIL import Image
import json
from config import *

class PlantDiseaseClassifier:
    def __init__(self, model_path=MODEL_PATH, class_names_path=CLASS_NAMES_PATH):
        # Initialize the base model
        self.model = mobilenet_v2(weights='DEFAULT')
        
        # Recreate the exact custom classifier head used during training
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, NUM_CLASSES)
        )
        
        # Load the pre-trained weights safely to the configured device (CPU/GPU)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model = self.model.to(DEVICE)
        self.model.eval()
        
        # Load disease class names mapping
        with open(class_names_path, 'r') as f:
            self.class_names = json.load(f)
        
        # Define the exact image transformations used during training/validation
        self.transform = transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
    
    def predict(self, image_path):
        """
        Executes inference on a single image and returns the predicted disease and confidence.
        """
        # Load and preprocess the image
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(DEVICE)
        
        # Run inference without tracking gradients
        with torch.no_grad():
            outputs = self.model(img_tensor)
            # Convert raw logits to probabilities
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, dim=0)
            
        disease_name = self.class_names[predicted_idx.item()]
        confidence_percentage = confidence.item() * 100
        
        return disease_name, confidence_percentage