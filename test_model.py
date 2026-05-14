from inference import PlantDiseaseClassifier
import os

clf = PlantDiseaseClassifier('best_plant_model.pth', 'class_names.json')
print("✓ Model loaded")

disease, conf = clf.predict('test.jpg')
print(f"\nDisease: {disease}")
print(f"Confidence: {conf:.2f}%")

if conf > 80:
    print("CONFIDENCE OK - MODEL WORKS!")
else:
    print("Low confidence - needs review")