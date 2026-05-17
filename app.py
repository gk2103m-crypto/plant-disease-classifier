import gradio as gr
import tempfile
import os
from inference import PlantDiseaseClassifier

clf = PlantDiseaseClassifier()

def predict(image):
    try:
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            image.save(tmp.name)
            temp_path = tmp.name
        
        disease, conf = clf.predict(temp_path)
        os.remove(temp_path)
        
        return f"Disease: {disease}\n\nConfidence: {conf:.2f}%"
    
    except Exception as e:
        return f"Error: {str(e)}"

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Plant Disease Classifier",
    description="Detect plant diseases with 97.72% accuracy"
)

if __name__ == "__main__":
   interface.launch(share=False, server_name="0.0.0.0", server_port=8000)