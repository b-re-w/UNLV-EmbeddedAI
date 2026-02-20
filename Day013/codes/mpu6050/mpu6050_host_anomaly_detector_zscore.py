import socket
import json
import math
import collections
import statistics

# --- CONFIGURATION ---
HOST_IP = '0.0.0.0'  # Listen on all interfaces
HOST_PORT = 65432  # Must match RPi
WINDOW_SIZE = 50  # How many previous samples to keep for "Normal" baseline
Z_THRESHOLD = 3.5  # How many standard deviations trigger an alert


class AnomalyDetector:
    def __init__(self):
        # A rolling buffer to store recent vibration magnitudes
        self.history = collections.deque(maxlen=WINDOW_SIZE)

    def process(self, data):
        # 1. Calculate Vibration Magnitude (Vector Sum)
        # We subtract 1.0 from Z to remove gravity (approx)
        ax, ay, az = data['ax'], data['ay'], data['az']
        magnitude = math.sqrt(ax ** 2 + ay ** 2 + az ** 2)

        # 2. Add to history
        self.history.append(magnitude)

        # 3. Need enough data to establish a baseline
        if len(self.history) < WINDOW_SIZE:
            return f"Calibrating... ({len(self.history)}/{WINDOW_SIZE})"

        # 4. Statistical Anomaly Detection (Z-Score)
        mean = statistics.mean(self.history)
        stdev = statistics.stdev(self.history)

        # Avoid division by zero if sensor is perfectly still
        if stdev == 0: stdev = 0.001

        z_score = (magnitude - mean) / stdev

        # 5. Output
        status = f"Mag: {magnitude:.3f} | Z-Score: {z_score:.2f}"

        if abs(z_score) > Z_THRESHOLD:
            return f"🔴 ANOMALY DETECTED! {status}"
        else:
            return f"🟢 Normal. {status}"


def main():
    print(f"Host Server Listening on port {HOST_PORT}...")
    detector = AnomalyDetector()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST_IP, HOST_PORT))
        s.listen()
        conn, addr = s.accept()

        with conn:
            print(f"Connected by {addr}")
            buffer = ""

            while True:
                # Receive data stream
                chunk = conn.recv(1024).decode('utf-8')
                if not chunk:
                    break

                buffer += chunk

                # Handle "TCP Glue" (multiple messages arriving at once)
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if line:
                        try:
                            # Parse JSON
                            data_point = json.loads(line)

                            # Run Detection
                            result = detector.process(data_point)
                            print(result)

                        except json.JSONDecodeError:
                            pass


if __name__ == "__main__":
    main()