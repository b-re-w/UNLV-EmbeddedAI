import cv2
import numpy as np
from ultralytics import YOLO

# --- CONFIGURATION ---
# HSV Color Range for Yellow (Adjust these if your yellow is different!)
# Hue is 0-179 in OpenCV. Yellow is usually around 20-35.
LOWER_YELLOW = np.array([20, 100, 100])
UPPER_YELLOW = np.array([35, 255, 255])

# Confidence to accept a YOLO detection
CONF_THRESHOLD = 0.5

# 1. Load the Model
# We use 'yolov8n.pt' for speed
model = YOLO('yolov8n.pt')

# 2. Setup Webcam
cap = cv2.VideoCapture(1)


# Helper function to check if an image region is "Yellow"
def is_yellow(roi):
    # Convert the small cutout (Region of Interest) to HSV
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Create a mask (1 for yellow pixels, 0 for others)
    mask = cv2.inRange(hsv_roi, LOWER_YELLOW, UPPER_YELLOW)

    # Count yellow pixels
    yellow_count = cv2.countNonZero(mask)
    total_pixels = roi.shape[0] * roi.shape[1]

    # If more than 30% of the pixels are yellow, we confirm it
    if total_pixels > 0 and (yellow_count / total_pixels) > 0.3:
        return True
    return False


print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break

    # 3. Run YOLO Tracking
    # persist=True enables the internal tracker (ByteTrack) to keep IDs consistent
    # classes=[32] filters ONLY 'sports ball' (Class ID 32 in COCO)
    results = model.track(frame, persist=True, verbose=False, classes=[32])

    # 4. Process Results
    for result in results:
        boxes = result.boxes

        # Iterating through every ball detected
        for box in boxes:
            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Get Track ID (if available yet)
            track_id = int(box.id[0]) if box.id is not None else 0

            # Extract the ball image (ROI)
            # Ensure we don't crop outside image bounds
            h, w, _ = frame.shape
            roi = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

            if roi.size > 0:
                # 5. Color Check
                if is_yellow(roi):
                    # It IS a yellow ball: Draw Green Box + ID
                    label = f"Yellow Ball #{track_id}"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    # It is a ball, but NOT yellow: Draw Red Box (Optional)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

    cv2.imshow("Yellow Ball Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()