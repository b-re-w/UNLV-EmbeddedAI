import cv2
import cv2.aruco as aruco
import numpy as np
import time
from djitellopy import Tello

# =====================
# --- CONFIGURATION ---
# =====================

SAFE_TEST_MODE = False   # Set to False to allow flight commands

TARGET_HEIGHT_ADDITION = 40  # cm (Takeoff + 40cm = ~120cm)
MOVE_DIST = 20              # Tello minimum reliable distance (cm)
COOLDOWN = 3              # seconds between commands

# Marker ID -> Action Map
ACTION_MAP = {
    0: "UP",
    5: "DOWN",
    12: "LEFT",
    42: "RIGHT",
    100: "LAND"
}

# =================
# --- CONNECT ---
# =================

print("Connecting to Tello...")
tello = Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")

tello.streamon()
time.sleep(2)

# =========================
# --- ARUCO SETUP (NEW API)
# =========================

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

# =====================
# --- FLIGHT SETUP ---
# =====================

if not SAFE_TEST_MODE:
    print("Taking Off...")
    tello.takeoff()

    print(f"Adjusting altitude to ~120cm (Moving UP {TARGET_HEIGHT_ADDITION}cm)...")
    tello.move_up(TARGET_HEIGHT_ADDITION)
else:
    print("SAFE TEST MODE ENABLED — NO FLIGHT COMMANDS WILL BE SENT")

print("Scanning for ArUco Markers...")
last_action_time = 0

# ==================
# --- MAIN LOOP ---
# ==================

try:
    frame_read = tello.get_frame_read()

    while True:
        frame = frame_read.frame

        if frame is None:
            continue

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect markers (NEW API)
        corners, ids, rejected = detector.detectMarkers(gray)

        current_time = time.time()

        # Process detected markers
        if ids is not None and (current_time - last_action_time > COOLDOWN):
            detected_ids = ids.flatten()

            for marker_id in detected_ids:
                if marker_id in ACTION_MAP:
                    action = ACTION_MAP[marker_id]
                    print(f">>> Marker {marker_id} Detected! Action: {action}")

                    if not SAFE_TEST_MODE:
                        if action == "UP":
                            tello.move_up(MOVE_DIST)
                        elif action == "DOWN":
                            tello.move_down(MOVE_DIST)
                        elif action == "LEFT":
                            tello.move_left(MOVE_DIST)
                        elif action == "RIGHT":
                            tello.move_right(MOVE_DIST)
                        elif action == "LAND":
                            print("Landing command received...")
                            tello.land()
                            raise StopIteration

                    last_action_time = current_time
                    break

        # Draw detected markers
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)

        # Display window
        cv2.imshow("Tello ArUco Controller", frame)

        # Manual landing / exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Manual Exit / Landing")
            if not SAFE_TEST_MODE:
                tello.land()
            break

except StopIteration:
    print("Mission Complete")

except Exception as e:
    print(f"Error: {e}")
    if not SAFE_TEST_MODE:
        tello.land()

finally:
    tello.streamoff()
    cv2.destroyAllWindows()
