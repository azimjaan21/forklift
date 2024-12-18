# Forklift Detection with YOLOv8 🚚

## Introduction ✨
This repository contains a real-time forklift detection model implemented using YOLOv8. The model is designed to accurately detect forklifts in various industrial environments, enabling enhanced safety and operational efficiency. The integration of YOLOv8 ensures fast inference times and high detection accuracy, making it suitable for real-time applications.

---

## Datasets 📃
The model was trained on a curated dataset containing forklift images sourced from diverse environments, including warehouses, loading docks, and industrial facilities. The dataset includes:

- **Training Set**: Annotated forklift images with bounding boxes.
- **Validation Set**: Images for evaluating the model during training.
- **Test Set**: Independent images for final evaluation.

Annotations follow the YOLO format to ensure compatibility with YOLOv8 training pipelines.

---

## Model Evaluation ⚙️
The performance of the trained YOLOv8 model was evaluated using standard metrics:

- **Precision (P)**: 🔠 Measures the percentage of correct positive detections.
- **Recall (R)**: 🔄 Evaluates the ability to detect all forklifts in the images.
- **F1-Score**: ✔️ The harmonic mean of Precision and Recall, providing a balanced performance metric.

Detailed results:

| Metric     | Value  |
|------------|--------|
| Precision  | 0.95   |
| Recall     | 0.93   |
| F1-Score   | 0.94   |

---

## Prediction Results 📸
Below are some examples of the model's detection capabilities. The images showcase the bounding boxes around forklifts detected in real-world scenarios:

![🔍 Detection Example 1](path/to/example1.jpg)
![🔍 Detection Example 2](path/to/example2.jpg)
![🔍 Detection Example 3](path/to/example3.jpg)

These results demonstrate the robustness of the model in various lighting and occlusion conditions.

---

## Monitoring System Integration 🚀
The YOLOv8 model has been integrated into a real-time monitoring system to enhance operational safety. Key features include:

1. **Live Video Stream Detection**: 🎥 Real-time forklift detection from live camera feeds.
2. **Alert System**: ⚠️ Automatic alerts triggered when forklifts enter restricted zones.
3. **Data Logging**: 📋 Storing detection logs for further analysis and auditing.

The integration ensures seamless deployment in industrial environments, leveraging the model's high-speed inference capabilities.

---

## How to Use 📚

### Prerequisites 🛠️
- Python 3.8+
- [Ultralytics YOLOv8](https://github.com/ultralytics/yolov8)
- Required libraries: `torch`, `opencv-python`, `numpy`

### Installation ⚙️
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/forklift-detection.git
   cd forklift-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Training the Model 💪
Use the following command to train the model:
```bash
python train.py --data forklift.yaml --epochs 50 --img 640
```

### Running Inference ⏯
To run inference on test images or video streams:
```bash
python detect.py --weights best.pt --source path/to/images_or_video
```

---

## Contributions 🙌
Contributions are welcome! Please submit issues or pull requests for improvements or bug fixes.

---

## License 🔒
This project is licensed under the MIT License. See the LICENSE file for more details.


