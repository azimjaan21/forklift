
from ultralytics import YOLO
import os

# Load the YOLO model
model = YOLO('ppe.pt')

# Define input and output paths
image_path = r'C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\SafeFactory Object Detection\forklift\samples\#'
output_folder = r'C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\SafeFactory Object Detection\forklift\predictions'
os.makedirs(output_folder, exist_ok=True)

# Run model prediction
results = model.predict(
    source=image_path,
    conf=0.25,
    save=True,       # Set save=True to save the predicted image
    save_txt=True,   # Save labels to text file
    project=output_folder,  # Set output directory for saved images and labels
    device="cuda"
)
