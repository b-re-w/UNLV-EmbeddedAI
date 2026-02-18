import cv2
import math
from ultralytics import YOLO

# --- CONFIGURATION ---
# 1. The Object you want to detect (must be in COCO dataset)
TARGET_CLASS = 'cell phone'

# 2. The Real-world width of that object (in cm/inches)
# Example: A standard iPhone is roughly 7.5 cm wide
KNOWN_WIDTH = 7.5  # cm

# 3. The Focal Length of your camera (Calibrate this!)
# Formula: F = (Pixel_Width * Known_Distance) / Known_Width
# If you don't know it, guess 600-800 for a standard 720p webcam as a starting point.
FOCAL_LENGTH = 700

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Start Webcam
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret: break

    # Run YOLO detection
    results = model(frame, verbose=False)

    # Iterate through detections
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get Class ID and Name
            cls = int(box.cls[0])
            label = model.names[cls]

            # Only process if it matches our Target Object
            if label == TARGET_CLASS:
                # Get Bounding Box Coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Calculate the width of the box in pixels
                pixel_width = x2 - x1

                # --- DISTANCE ESTIMATION FORMULA ---
                if pixel_width > 0:
                    distance = (KNOWN_WIDTH * FOCAL_LENGTH) / pixel_width

                    # Draw Bounding Box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw Distance Label
                    text = f"{label}: {distance:.2f} cm"
                    cv2.putText(frame, text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('YOLO Distance Estimation', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()