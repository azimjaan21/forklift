from ultralytics import YOLO
import cv2

# Load the trained YOLO model
model = YOLO(r'C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\SafeFactory Object Detection\forklift\best.pt')

# Open the webcam (use 0 for the default webcam)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Run the model prediction on the frame
    results = model.predict(
        source=frame,
        conf=0.5,  # Base confidence threshold
        save=False,  # Do not save frames
        device="cuda"  # Use GPU if available
    )

    # Flag to determine if a fall is detected
    fall_detected = False

    # Process predictions and draw bounding boxes
    for result in results:
        for box in result.boxes:
            # Extract bounding box details
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0]  # Confidence score
            cls = int(box.cls[0])  # Class index
            label = result.names[cls]  # Class name

            # Draw bounding boxes for detections
            if label == "fall" and conf > 0.89:
                # Draw bounding box for 'fall' with high confidence
                color = (0, 0, 255)  # Red for 'fall'
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(
                    frame,
                    f"{conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )
                fall_detected = True
            elif conf > 0.5:
                # Draw bounding box for other labels with confidence > 0.5
                color = (255, 0, 0)  # Blue for other labels
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(
                    frame,
                    f"{conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

    # Overlay the status message
    if fall_detected:
        cv2.putText(
            frame,
            "Fall Detected!",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 255),  # Red color for "Fall Detected!"
            3
        )
    else:
        cv2.putText(
            frame,
            "Normal",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),  # Green color for "Normal"
            3
        )

    # Display the frame with detections and message
    cv2.imshow("Webcam Test - Real-Time Detection", frame)

    # Press 'q' to exit the real-time webcam feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
