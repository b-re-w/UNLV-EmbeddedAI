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
        # Plot the boxes/labels on the frame
        annotated_frame = result.plot()

        # Display the frame
        cv2.imshow("YOLO Real-Time Inference", annotated_frame)

        # 6. Break Loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Cleanup
cap.release()
cv2.destroyAllWindows()
