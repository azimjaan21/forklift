from ultralytics import YOLO

def validate():
    # Load the trained model
    model = YOLO(r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\SafeFactory Object Detection\forklift\runs\detect\train\weights\best.pt")
    
    # Run validation on the validation set
    results = model.val(
        data=r'C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\SafeFactory Object Detection\fall\forklift_data.yaml',  # Path to dataset YAML
        imgsz=640,  # Image size
        batch=16,  # Batch size
        conf=0.5,  # Confidence threshold
        iou=0.5,  # IoU threshold
        save_json=True,  
    )
    
    print("Validation Results:", results)

if __name__ == "__main__":
    validate()
