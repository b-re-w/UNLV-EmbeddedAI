import socket
import json
import numpy as np
from sklearn.ensemble import IsolationForest
import collections

# --- CONFIGURATION ---
HOST_IP = '0.0.0.0'  # Listen on all interfaces
PORT = 65432
BUFFER_SIZE = 50  # Number of samples to hold in sliding window
CALIBRATION_SIZE = 200  # Samples to train initial normal behavior


class VibrationDetector:
    def __init__(self):
        self.data_buffer = collections.deque(maxlen=BUFFER_SIZE)
        self.training_data = []
        self.model = IsolationForest(contamination=0.05, random_state=42)
        self.is_calibrated = False

    def process_packet(self, packet):
        # Extract features: We use simple raw X, Y, Z for this demo
        # In production, use FFT (Frequency domain) features
        features = [packet['x'], packet['y'], packet['z']]

        # 1. Calibration Phase
        if not self.is_calibrated:
            self.training_data.append(features)
            print(f"Calibrating... {len(self.training_data)}/{CALIBRATION_SIZE}", end='\r')

            if len(self.training_data) >= CALIBRATION_SIZE:
                print("\nTraining Anomaly Model...")
                self.model.fit(self.training_data)
                self.is_calibrated = True
                print("--- SYSTEM ARMED: DETECTING ANOMALIES ---")
            return "CALIBRATING"

        # 2. Detection Phase
        # Predict: 1 = Normal, -1 = Anomaly
        prediction = self.model.predict([features])[0]
        score = self.model.decision_function([features])[0]

        return "NORMAL" if prediction == 1 else "ANOMALY", score


def start_server():
    detector = VibrationDetector()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST_IP, PORT))
        s.listen()
        print(f"Listening on {PORT}...")

        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            buffer = ""

            while True:
                data = conn.recv(1024)
                if not data: break

                # Handle stream fragmentation
                buffer += data.decode('utf-8')
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if not line: continue

                    try:
                        packet = json.loads(line)
                        status, score = detector.process_packet(packet)

                        # Calculate simple RMS (Vibration Intensity) for visual reference
                        rms = np.sqrt(packet['x'] ** 2 + packet['y'] ** 2 + packet['z'] ** 2)

                        # Print status
                        if status == "ANOMALY":
                            print(f"⚠️ ANOMALY DETECTED! | Score: {score:.2f} | Intensity (RMS): {rms:.0f}")
                        elif status == "NORMAL":
                            print(f"✅ Normal Operation   | Score: {score:.2f} | Intensity (RMS): {rms:.0f}", end='\r')

                    except json.JSONDecodeError:
                        pass  # Partial packet received


if __name__ == "__main__":
    start_server()