import socket
import json
import numpy as np
from sklearn.ensemble import IsolationForest
import warnings

# --- CONFIGURATION ---
HOST_IP = '0.0.0.0'  # Listen on all network interfaces
HOST_PORT = 65432  # Must match the port in rpi_sender.py
TRAIN_SIZE = 500  # Number of samples to collect for initial training
CONTAMINATION = 0.05  # Sensitivity (0.01 = low sensitivity, 0.1 = high)

# Suppress sklearn warnings about feature names
warnings.filterwarnings("ignore")


class MLAnomalyDetector:
    def __init__(self):
        self.buffer = []
        self.model = None
        self.is_trained = False

        # Initialize Isolation Forest
        # n_estimators: Number of trees in the forest
        # contamination: Expected proportion of outliers in the data
        self.clf = IsolationForest(n_estimators=100,
                                   contamination=CONTAMINATION,
                                   random_state=42)

    def process_stream(self, data_point):
        """
        Input: Dictionary {'ax': 1.2, 'ay': 0.1, 'az': 0.9}
        Output: Status String
        """
        # Convert dict to list [ax, ay, az]
        features = [data_point['ax'], data_point['ay'], data_point['az']]

        # PHASE 1: Data Collection
        if not self.is_trained:
            self.buffer.append(features)
            count = len(self.buffer)
            progress = (count / TRAIN_SIZE) * 100

            if count >= TRAIN_SIZE:
                print(f"\n[System] Buffer full ({TRAIN_SIZE} samples). Training Model...")
                self.train_model()
                return "[System] Training Complete. Switched to Detection Mode."
            else:
                return f"[Calibration] collecting data... {int(progress)}%"

        # PHASE 3: Real-time Detection
        else:
            # Reshape for sklearn (1 sample, 3 features)
            X = np.array([features])

            # Predict: 1 = Normal, -1 = Anomaly
            prediction = self.clf.predict(X)
            score = self.clf.decision_function(X)[0]  # Negative score = more anomalous

            if prediction[0] == -1:
                return f"🔴 ANOMALY DETECTED! Score: {score:.3f} | Input: {features}"
            else:
                return f"🟢 Normal. Score: {score:.3f}"

    def train_model(self):
        # Convert buffer to numpy array
        X_train = np.array(self.buffer)

        # Fit the model to the "normal" data we collected
        self.clf.fit(X_train)

        self.is_trained = True
        print(f"[System] Model Trained on {len(X_train)} samples.")
        print(f"[System] Ready to detect anomalies based on learned patterns.")


def main():
    print(f"--- ML Host Server Listening on port {HOST_PORT} ---")
    print(f"--- Waiting for RPi Connection... ---")

    detector = MLAnomalyDetector()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST_IP, HOST_PORT))
        s.listen()
        conn, addr = s.accept()

        with conn:
            print(f"Connected by {addr}")
            print("Keep the sensor in 'Normal' state for calibration...")

            buffer = ""

            while True:
                try:
                    chunk = conn.recv(1024).decode('utf-8')
                    if not chunk:
                        break

                    buffer += chunk

                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        if line:
                            # Parse JSON from RPi
                            data_point = json.loads(line)

                            # Feed to ML Detector
                            status = detector.process_stream(data_point)

                            # Print status (overwrite line for clean UI if on Mac/Linux)
                            print(f"\r{status.ljust(80)}", end="")

                except json.JSONDecodeError:
                    pass
                except KeyboardInterrupt:
                    print("\nStopping...")
                    break
                except Exception as e:
                    print(f"\nError: {e}")
                    break


if __name__ == "__main__":
    main()