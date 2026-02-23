import socket
import json
import numpy as np
import collections
from river import compose
from river import preprocessing
from river import tree

# --- CONFIGURATION ---
HOST_IP = '0.0.0.0'
PORT = 65432
WINDOW_SIZE = 20  # Number of samples to aggregate for one "observation"

# --- THE MODEL PIPELINE ---
# 1. StandardScaler: Scales features (variance can vary wildly)
# 2. HoeffdingTreeClassifier: The standard for streaming classification
model = compose.Pipeline(
    preprocessing.StandardScaler(),
    tree.HoeffdingTreeClassifier()
)

# Class Labels mapping
LABELS = {0: "STILL/IDLE", 1: "MOTION/VIBRATION"}


def extract_features(window):
    """
    Raw accelerometer data is noisy. We don't classify single points.
    We classify the STATS of a window (e.g., standard deviation represents vibration).
    """
    data = np.array(window)  # Shape: (20, 3)

    # Calculate Standard Deviation for X, Y, Z (captures intensity of movement)
    # and Mean (captures orientation/tilt)
    features = {
        "std_x": np.std(data[:, 0]),
        "std_y": np.std(data[:, 1]),
        "std_z": np.std(data[:, 2]),
        "mean_x": np.mean(data[:, 0]),
        "mean_y": np.mean(data[:, 1]),
        "mean_z": np.mean(data[:, 2]),
    }
    return features


def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST_IP, PORT))
    server.listen(1)

    print(f"Listening on {PORT}...")
    conn, addr = server.accept()

    # Sliding Window Buffer
    window = collections.deque(maxlen=WINDOW_SIZE)
    buffer_str = ""

    # Training State
    # We will automate training for the demo:
    # Seconds 0-10: Learn "IDLE" (Class 0)
    # Seconds 10-20: Learn "MOTION" (Class 1)
    # Seconds 20+: Predict
    start_time = None

    with conn:
        print(f"Connected by {addr}")

        while True:
            data = conn.recv(1024)
            if not data: break

            buffer_str += data.decode('utf-8')

            while "\n" in buffer_str:
                line, buffer_str = buffer_str.split("\n", 1)
                if not line: continue

                try:
                    # 1. Parse Data
                    packet = json.loads(line)
                    raw_vec = [packet['ax'], packet['ay'], packet['az']]
                    window.append(raw_vec)

                    # Only process if window is full
                    if len(window) < WINDOW_SIZE:
                        continue

                    # 2. Extract Features
                    x = extract_features(window)

                    # 3. Time-Based Logic (Auto-Training)
                    if start_time is None: start_time = packet['timestamp']
                    elapsed = packet['timestamp'] - start_time

                    y = None  # The label
                    mode = ""

                    if elapsed < 15:
                        # PHASE 1: TEACH "STILL"
                        y = 0
                        mode = "TRAINING (KEEP SENSOR STILL)"
                        model.learn_one(x, y)  # <--- ONLINE LEARNING HAPPENS HERE

                    elif elapsed < 30:
                        # PHASE 2: TEACH "MOTION"
                        y = 1
                        mode = "TRAINING (SHAKE THE SENSOR!)"
                        model.learn_one(x, y)  # <--- ONLINE LEARNING HAPPENS HERE

                    else:
                        # PHASE 3: PREDICTION
                        mode = "PREDICTING"
                        # We predict first
                        y_pred = model.predict_one(x)

                        # (Optional) Unsupervised Learning / Reinforcement
                        # In true unsupervised online learning, we might update the model
                        # based on cluster centroids, but here we just predict.

                        label_name = LABELS.get(y_pred, "Unknown")
                        print(f"Mode: {mode} | Detected: {label_name}", end='\r')
                        continue

                    # Print Training Status
                    print(f"Time: {elapsed:.1f}s | {mode} | Learning Class: {LABELS[y]}", end='\r')

                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    print(e)


if __name__ == "__main__":
    start_server()