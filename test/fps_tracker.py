import cv2
import time
import os
from ultralytics import YOLO

# Load the YOLOv8 model
try:
    model = YOLO("ppe.pt")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Path to the sample video
video_path = r'C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\SafeFactory Object Detection\forklift\samples\############'

# Create output directory if it doesn't exist
output_dir = r'C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\SafeFactory Object Detection\forklift\test\fps_test'
os.makedirs(output_dir, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit(1)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
output_file = os.path.join(output_dir, 'output_video(captured).mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

# Define class names and colors for each class
class_names = {
    0: 'helmet',
    1: 'vest',
    2: 'head'
}

class_colors = {
    0: (255, 0, 0),    # Blue for helmets
    1: (0, 100, 0),    # Dark green for vests
    2: (255, 255, 0)   # Cyan for heads
}

# Start the time and frame counter
prev_time = time.time()
fps_count = 0
fps_total = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Calculating FPS
    current_time = time.time()
    fps_count += 1
    fps_total += 1 / (current_time - prev_time)
    prev_time = current_time

    # Display average FPS every 10 frames
    if fps_count % 10 == 0:
        avg_fps = fps_total / fps_count
        cv2.putText(frame, f'Avg FPS: {avg_fps:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

    # Draw results on the frame
    for result in results:
        boxes = result.boxes.xyxy  # Get box coordinates in xyxy format
        confidences = result.boxes.conf  # Get confidence scores
        class_ids = result.boxes.cls  # Get class labels

        # Loop through each box to draw
        for box, conf, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)  # Convert to integer coordinates
            class_name = class_names.get(int(class_id), 'Unknown')
            color = class_colors.get(int(class_id), (255, 255, 255))  # Default to white if class is unknown

            label = f'{class_name} Conf: {conf:.2f}'
            
            # Draw rectangle and label on the frame with reduced thickness
            line_thickness = 2  # Thinner bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_thickness)

            # Calculate text size for background rectangle
            font_scale = 0.5  # Smaller font size for clarity
            font_thickness = 1
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]

            # Draw filled rectangle behind the text
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + text_size[0], y1), color, cv2.FILLED)

            # Put text on the frame
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

    # Show the frame with detections
    cv2.imshow("YOLOv8 Output", frame)

    # Write the processed frame to the output video
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
