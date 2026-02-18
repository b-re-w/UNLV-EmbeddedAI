import cv2
import time
from djitellopy import Tello
from ultralytics import YOLO

# --- CONFIGURATION ---
# The specific model you requested
MODEL_NAME = "yolov8x-tuned-hand-gestures.pt"

# Tello takeoff is ~80cm. We add 20cm to reach ~100cm.
TARGET_HEIGHT_ADDITION = 20
MOVE_DIST = 10  # Requested 10cm (Note: Tello min reliable dist is usually 20cm)
CONF_THRESHOLD = 0.6  # High confidence to avoid accidental moves
COOLDOWN = 2.0  # Seconds between moves

# --- GESTURE MAPPING ---
# You MUST verify these class names match your specific 'tuned' model!
# Run 'print(model.names)' to see the actual list if these don't work.
GESTURE_MAP = {
    "ThumbUp": "UP",
    "ThumbDown": "DOWN",
    "Open_Palm": "LAND",  # Common name in gesture datasets
    "Victory": "LEFT",  # Peace sign
    "Fist": "RIGHT"
}

# --- SETUP ---
print(f"Loading {MODEL_NAME}...")
try:
    # Load the custom tuned model
    model = YOLO(MODEL_NAME)
    print("Model loaded successfully.")
    print("Classes known by model:", model.names)
except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure the .pt file is in the same folder as this script.")
    exit()

print("Connecting to Tello...")
tello = Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")

tello.streamon()
frame_read = tello.get_frame_read()
time.sleep(2)  # Warmup

# --- FLIGHT SEQUENCE ---
print("Taking Off...")
tello.takeoff()

print(f"Adjusting height to ~100cm (Up {TARGET_HEIGHT_ADDITION}cm)...")
tello.move_up(TARGET_HEIGHT_ADDITION)

print("Gesture Control Active! (Press 'q' to emergency land)")
last_action_time = 0

try:
    while True:
        # 1. Get Frame
        frame = frame_read.frame
        # Resize is CRITICAL for 'yolov8x' on CPU, otherwise it will be too slow
        frame = cv2.resize(frame, (640, 480))

        # 2. Run Inference
        results = model(frame, verbose=False, conf=CONF_THRESHOLD)

        detected_action = None

        # 3. Parse Gestures
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get class name
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]

                # Check mapping
                if cls_name in GESTURE_MAP:
                    detected_action = GESTURE_MAP[cls_name]

                    # Visual Feedback
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"CMD: {detected_action}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    break
            if detected_action: break

        # 4. Execute Command
        if detected_action and (time.time() - last_action_time > COOLDOWN):
            print(f">>> EXECUTING: {detected_action}")

            if detected_action == "UP":
                tello.move_up(MOVE_DIST)
            elif detected_action == "DOWN":
                tello.move_down(MOVE_DIST)
            elif detected_action == "LEFT":
                tello.move_left(MOVE_DIST)
            elif detected_action == "RIGHT":
                tello.move_right(MOVE_DIST)
            elif detected_action == "LAND":
                print("Landing...")
                tello.land()
                break  # Exit loop

            last_action_time = time.time()

        # Display
        cv2.imshow("Tello Gesture Pilot", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            tello.land()
            break

except Exception as e:
    print(f"Error: {e}")
    tello.land()

finally:
    tello.streamoff()
    cv2.destroyAllWindows()