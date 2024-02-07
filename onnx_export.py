import os
from ultralytics import YOLO

# Define the folder containing the model files
folder_path = "Models"

# Loop through files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".pt"):
        # Construct the full path to the model file
        model_path = os.path.join(folder_path, filename)
        
        # Load the model
        model = YOLO(model_path)
        
        # Export the model to ONNX format
        model.export(format="onnx", dynamic=True, simplify=True)
