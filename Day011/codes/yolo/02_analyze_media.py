import cv2
from ultralytics import YOLO

# ================= CONFIGURATION =================
# Choose your file (Use the filename you generated in Part 2)
# Examples: 'capture_12345.jpg' OR 'video_12345.mp4' OR just 0 for live webcam
SOURCE = 1

# Choose your TASK: 'detect', 'classify', or 'segment'
TASK = 'segment'
# =================================================

# 1. Select the correct Model based on the Task
if TASK == 'detect':
    print("Loading Detection Model (yolov8n.pt)...")
    model = YOLO('yolov8n.pt')  # Standard object detection

elif TASK == 'classify':
    print("Loading Classification Model (yolov8n-cls.pt)...")
    model = YOLO('yolov8n-cls.pt')  # Image Classification (ImageNet)

elif TASK == 'segment':
    print("Loading Segmentation Model (yolov8n-seg.pt)...")
    model = YOLO('yolov8n-seg.pt')  # Instance Segmentation

# 2. Open the Source (Image or Video)
# YOLO handles opening files internally, but standard CV2 loop gives us control
if isinstance(SOURCE, int):
    cap = cv2.VideoCapture(SOURCE)  # Webcam
else:
    # Check if it's an image or video based on extension
    if SOURCE.endswith(('.jpg', '.png', '.jpeg')):
        # Run single image inference
        results = model(SOURCE)
        res_plotted = results[0].plot()
        cv2.imshow(f"YOLO {TASK.title()}", res_plotted)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        exit()
    else:
        cap = cv2.VideoCapture(SOURCE)  # Video File

# 3. Video/Webcam Inference Loop
while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # Run Inference
    # verbose=False keeps the terminal clean
    results = model(frame, verbose=False)

    # Visualize
    annotated_frame = results[0].plot()

    # If Classifying, YOLO plots the "Top-5" classes.
    # If Detecting/Segmenting, it plots boxes/masks.

    cv2.imshow(f"YOLOv8 {TASK.title()}", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
