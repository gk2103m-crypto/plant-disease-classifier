import torch
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# LOCAL PATHS (not Colab!)
MODEL_PATH = "best_plant_model.pth"
CLASS_NAMES_PATH = "class_names.json"

NUM_CLASSES = 38
INPUT_SIZE = 224

BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
PATIENCE = 5

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]