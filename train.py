from ultralytics import YOLO

def train_model():
    
    model = YOLO('yolov8m.pt') #pretrained m version

    model.train(
        data=r'C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\SafeFactory Object Detection\forklift\forklift_data.yaml', 
        epochs=100,                  
        batch=16,                   
        imgsz=640,          
        optimizer='AdamW',            
        augment=True,                             
        device='cuda',                               
        verbose=True, 
    )

if __name__ == '__main__':
    train_model()
