import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from config import *

def create_model():
    model = mobilenet_v2(weights='DEFAULT')
    
    for param in model.features.parameters():
        param.requires_grad = False
    
    for param in model.features[-2:].parameters():
        param.requires_grad = True
    
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(1280, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, NUM_CLASSES)
    )
    
    model = model.to(DEVICE)
    return model

def load_model(model_path):
    model = create_model()
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
    
    model.eval()
    return model