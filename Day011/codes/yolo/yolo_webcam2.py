import cv2
from ultralytics import YOLO

# 1. Load the Model
model = YOLO('yolov8n.pt')

# 2. Open the Webcam (0 is usually the default camera)
cap = cv2.VideoCapture(1)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    # 3. Read a frame from the webcam
    success, frame = cap.read()
    if not success:
        break

        # 4. Run YOLO Inference on the frame
    # stream=True is recommended for video sources to manage memory better
    results = model(frame, stream=True)

    # 5. Visualize Results
    # The generator yields one result per frame
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs

        for box in boxes:
            # Bounding Box Coordinates (x1, y1, x2, y2)
            x1, y1, x2, y2 = box.xyxy[0]

            # Confidence Score (0.0 to 1.0)
            conf = box.conf[0]

            # Class ID (0, 1, 2...)
            cls = int(box.cls[0])

            # Class Name (e.g., 'person', 'cup')
            name = model.names[cls]

            if name == 'person' and conf > 0.5:
                print(f"Person detected at [{x1}, {y1}] with {conf:.2f} confidence") 

        # 6. Break Loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Cleanup
cap.release()
cv2.destroyAllWindows()
