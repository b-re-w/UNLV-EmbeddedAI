import cv2
import numpy as np
import math
import os

# -----------------------------
# Utility: rotation matrix -> Euler angles (roll, pitch, yaw)
# Convention used: intrinsic rotations about X (roll), Y (pitch), Z (yaw)
# Output in degrees.
# -----------------------------
def rmat_to_euler_xyz(R: np.ndarray):
    """
    Convert rotation matrix to Euler angles (roll, pitch, yaw) in degrees.
    Uses XYZ intrinsic (equivalently ZYX extrinsic) convention.
    Handles gimbal lock.
    """
    # Ensure numeric stability
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        roll  = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = math.atan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock
        roll  = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = 0.0

    return np.degrees([roll, pitch, yaw])


def load_calibration(npz_path: str):
    """
    Expects an .npz file containing:
      - camera_matrix (3x3)
      - dist_coeffs (1xN) or (N,)
    """
    if not os.path.exists(npz_path):
        return None, None
    data = np.load(npz_path)
    K = data.get("camera_matrix", None)
    D = data.get("dist_coeffs", None)
    if K is None or D is None:
        return None, None
    D = D.reshape(-1, 1) if D.ndim == 1 else D
    return K, D


# -----------------------------
# Configuration
# -----------------------------
CAM_INDEX = 0                 # change if needed
MARKER_LENGTH_M = 0.05        # marker side length in meters (e.g., 0.05 = 5 cm)
CALIB_FILE = "camera_calib.npz"  # optional; put your calibration here

# ArUco dictionary
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
DETECTOR_PARAMS = cv2.aruco.DetectorParameters()
DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, DETECTOR_PARAMS)

# Load calibration if available
K, D = load_calibration(CALIB_FILE)

# -----------------------------
# Video capture
# -----------------------------
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Could not open camera index {CAM_INDEX}")

# If no calibration, we will approximate intrinsics from the first frame
approx_intrinsics_ready = False

print("Press 'q' to quit.")
while True:
    ok, frame = cap.read()
    if not ok:
        break

    h, w = frame.shape[:2]

    # Approximate camera matrix if no calibration provided
    if (K is None or D is None) and not approx_intrinsics_ready:
        # Rough guess: focal length ~ max(w,h), principal point at center
        f = max(w, h)
        K = np.array([[f, 0, w / 2],
                      [0, f, h / 2],
                      [0, 0, 1]], dtype=np.float64)
        D = np.zeros((5, 1), dtype=np.float64)
        approx_intrinsics_ready = True
        print("WARNING: Using approximate intrinsics. For accurate pose/orientation, calibrate your camera.")

    # Detect markers
    corners, ids, rejected = DETECTOR.detectMarkers(frame)

    if ids is not None and len(ids) > 0:
        # Draw detected markers
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Estimate pose for each marker
        # OpenCV expects corners as list of (1,4,2) arrays
        rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(
            corners, MARKER_LENGTH_M, K, D
        )

        for i, marker_id in enumerate(ids.flatten()):
            rvec = rvecs[i].reshape(3, 1)
            tvec = tvecs[i].reshape(3, 1)

            # Draw axes on marker
            cv2.drawFrameAxes(frame, K, D, rvec, tvec, MARKER_LENGTH_M * 0.5)

            # Convert rvec -> rotation matrix
            R, _ = cv2.Rodrigues(rvec)

            # Euler angles
            roll, pitch, yaw = rmat_to_euler_xyz(R)

            # Position in meters (camera frame)
            x, y, z = tvec.flatten()

            # Overlay text
            text1 = f"ID:{marker_id}  t[m]=({x:+.3f},{y:+.3f},{z:+.3f})"
            text2 = f"Euler deg (R,P,Y)=({roll:+.1f},{pitch:+.1f},{yaw:+.1f})"

            # Place text near first corner
            c0 = corners[i][0][0]  # first corner (x,y)
            px, py = int(c0[0]), int(c0[1])
            cv2.putText(frame, text1, (px, py - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(frame, text2, (px, py - 5),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    cv2.imshow("ArUco DICT_5x5_100 Pose + Euler", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
