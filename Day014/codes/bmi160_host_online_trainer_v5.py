import socket
import json
import numpy as np
from river import tree, compose, preprocessing
import collections
import threading
import time
import sys

# --- CONFIGURATION ---
HOST_IP = '0.0.0.0'
HOST_PORT = 65432
WINDOW_SIZE = 50  # 1 second of data @ 50Hz

# --- GLOBAL STATE ---
system_state = "IDLE"
current_label = None
is_running = True
is_model_trained = False

# --- ML PIPELINE (FIXED) ---
model = compose.Pipeline(
    preprocessing.StandardScaler(),
    tree.HoeffdingTreeClassifier(
        grace_period=100,  # Wait 100 samples before trying to split
        delta=0.01  # REPLACED 'split_confidence' with 'delta'
    )
)

data_buffer = collections.deque(maxlen=WINDOW_SIZE)


def extract_features(window):
    """
    Combines V3 stability with V4 physics awareness.
    """
    arr = np.array(window)
    ax, ay, az = arr[:, 0], arr[:, 1], arr[:, 2]

    # 1. Physics: Magnitude (Total Energy)
    mag = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)

    features = {}

    # --- Feature Set ---
    # Standard Deviation of Magnitude is the #1 feature for Vertical Motion
    features['mag_std'] = np.std(mag)
    features['mag_mean'] = np.mean(mag)

    # Range of motion on individual axes
    features['ax_ptp'] = np.ptp(ax)
    features['ay_ptp'] = np.ptp(ay)
    features['az_ptp'] = np.ptp(az)

    # Standard Deviation of axes (Good for 'Shake')
    features['ax_std'] = np.std(ax)
    features['ay_std'] = np.std(ay)
    features['az_std'] = np.std(az)

    return features


def network_worker():
    global is_running, system_state, current_label, is_model_trained

    print(f" >> [Background] Server listening on {HOST_PORT}...")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST_IP, HOST_PORT))
        s.listen()

        conn, addr = s.accept()
        print(f" >> [Background] RPi Connected: {addr}")

        with conn:
            raw_buffer = ""
            while is_running:
                try:
                    chunk = conn.recv(4096).decode('utf-8')
                    if not chunk: break
                    raw_buffer += chunk

                    while "\n" in raw_buffer:
                        line, raw_buffer = raw_buffer.split("\n", 1)
                        if not line: continue

                        try:
                            data = json.loads(line)
                            data_buffer.append([data['ax'], data['ay'], data['az']])

                            if len(data_buffer) == WINDOW_SIZE:
                                feat_vector = extract_features(data_buffer)

                                # --- TRAINING ---
                                if system_state == "TRAINING" and current_label is not None:
                                    model.learn_one(feat_vector, current_label)
                                    is_model_trained = True

                                # --- INFERENCE ---
                                elif system_state == "INFERENCE":
                                    if is_model_trained:
                                        pred = model.predict_one(feat_vector)
                                        probs = model.predict_proba_one(feat_vector)
                                        conf = probs.get(pred, 0.0) if probs else 0.0

                                        sys.stdout.write(f"\r >> [Inference] Prediction: {pred} ({conf:.2f})      ")
                                        sys.stdout.flush()
                                    else:
                                        sys.stdout.write("\r >> [Inference] Model not trained yet!      ")
                                        sys.stdout.flush()
                        except json.JSONDecodeError:
                            pass

                except Exception as e:
                    print(f"\n >> [Error] {e}")
                    break
    print("\n >> [Background] Closed.")


def main():
    global is_running, system_state, current_label

    t = threading.Thread(target=network_worker, daemon=True)
    t.start()
    time.sleep(1)

    print("\n" + "=" * 50)
    print(" MOTION CLASSIFIER V5 (FIXED)")
    print(" 1. Enter Label to train (e.g. 'static', 'vertical').")
    print(" 2. Enter 'stop' to test/predict.")
    print("=" * 50 + "\n")

    while is_running:
        if system_state != "INFERENCE":
            try:
                command = input("\nEnter Command (Label or 'stop'): ").strip()
            except EOFError:
                break

            if command.lower() == 'stop':
                print(" >> Switching to INFERENCE MODE... (Type 'q' to quit)")
                system_state = "INFERENCE"

                # Inference Loop
                while True:
                    user_input = input()
                    if user_input.lower() == 'q':
                        print(" >> Quitting...")
                        is_running = False
                        sys.exit(0)

            elif command:
                current_label = command
                system_state = "TRAINING"
                print(f" >> TRAINING '{current_label}'... (10s)")

                for i in range(10, 0, -1):
                    sys.stdout.write(f"\r >> Time: {i}s ")
                    sys.stdout.flush()
                    time.sleep(1)

                print("\n >> DONE. Enter next command.")
                system_state = "IDLE"


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        is_running = False