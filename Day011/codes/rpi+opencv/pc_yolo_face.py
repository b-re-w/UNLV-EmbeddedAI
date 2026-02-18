import cv2
from ultralytics import YOLO

# --- CONFIGURATION ---
PI_IP = "192.168.1.15"  # <--- CHANGE THIS to your Pi's IP
PORT = 8554

# Load Model
# Use 'yolov8n.pt' for standard Person detection.
# Use 'yolov8n-face.pt' if you have downloaded specific face weights.
print("Loading YOLO model...")
model = YOLO('yolov8n.pt')

# Connect to the Pi's TCP Stream
stream_url = f"tcp://{PI_IP}:{PORT}"
print(f"Connecting to {stream_url}...")
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("Error: Could not connect to stream.")
    print("1. Ensure 'pi_stream.py' is running on the Pi.")
    print("2. Check the IP address.")
    exit()

print("Connected! Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Stream ended.")
        break

    # Run YOLO Inference
    # verbose=False keeps the terminal clean
    # classes=[0] filters results to ONLY Class 0 ('person') in standard COCO
    results = model(frame, verbose=False, classes=[0])

    # Visualize Results
    # .plot() handles the drawing of boxes and labels automatically
    annotated_frame = results[0].plot()

    cv2.imshow("Remote YOLO Face/Person Detection", annotated_frame)

    # Latency Management:
    # If the network buffers, the video might lag.
    # This loop discards old frames to keep the display "live".
    # Uncomment if you notice delay:
    # for _ in range(2): cap.grab()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()