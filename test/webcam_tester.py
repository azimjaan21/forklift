import cv2

for i in range(-1, 3):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera index {i} is working.")
        cap.release()
        break
    else:
        print(f"Camera index {i} could not be opened.")