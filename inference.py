import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from torchvision import transforms
from PIL import Image
import json
from config import *

class PlantDiseaseClassifier:
    def __init__(self, model_path=MODEL_PATH, class_names_path=CLASS_NAMES_PATH):
        self.model = mobilenet_v2(weights='DEFAULT')
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, NUM_CLASSES)
        )
        
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model = self.model.to(DEVICE)
        self.model.eval()
        
        with open(class_names_path, 'r') as f:
            self.class_names = json.load(f)
        
        self.transform = transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
    
    def predict(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = self.model(img_tensor)
            probs = torch.softmax(output, dim=1)
            conf, idx = torch.max(probs, 1)
        
        disease = self.class_names[idx.item()]
        confidence = conf.item() * 100
        
        return disease, confidence