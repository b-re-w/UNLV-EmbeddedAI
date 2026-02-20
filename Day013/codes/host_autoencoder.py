import socket
import json
import numpy as np
import os

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from collections import deque

# --- CONFIGURATION ---
HOST_IP = '0.0.0.0'  # Listen on all network interfaces
PORT = 65432  # Must match the RPi sender port
BUFFER_SIZE = 1  # Process 1 sample at a time for real-time feel
CALIBRATION_SAMPLES = 500  # Number of "Normal" samples to train the Autoencoder
EPOCHS = 50  # Training iterations
BATCH_SIZE = 32


class AutoencoderDetector:
    def __init__(self, input_dim=3):
        self.training_data = []
        self.is_calibrated = False
        self.threshold = 0.0
        self.model = self._build_model(input_dim)

    def _build_model(self, input_dim):
        """
        Builds a simple Autoencoder.
        Input (3) -> Compressed (2) -> Output (3)
        """
        # Encoder
        input_layer = Input(shape=(input_dim,))
        encoder = Dense(2, activation="relu")(input_layer)  # Bottleneck

        # Decoder
        decoder = Dense(input_dim, activation="linear")(encoder)  # Reconstruction

        # Compile
        autoencoder = Model(inputs=input_layer, outputs=decoder)
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder

    def train(self):
        """Trains the model on collected normal data."""
        data = np.array(self.training_data)

        # Normalize data (Simple Min-Max scaling based on training set)
        self.min_val = np.min(data, axis=0)
        self.max_val = np.max(data, axis=0) + 1e-5  # Avoid divide by zero
        norm_data = (data - self.min_val) / (self.max_val - self.min_val)

        print(f"\nTraining Autoencoder on {len(data)} samples...")
        self.model.fit(norm_data, norm_data,
                       epochs=EPOCHS,
                       batch_size=BATCH_SIZE,
                       shuffle=True,
                       verbose=0)

        # Calculate Reconstruction Error on Training Data to set Threshold
        reconstructions = self.model.predict(norm_data)
        train_loss = np.mean(np.square(norm_data - reconstructions), axis=1)

        # Set Threshold: Mean Error + 3 Standard Deviations
        self.threshold = np.mean(train_loss) + 3 * np.std(train_loss)
        self.is_calibrated = True
        print(f"Training Complete. Threshold set to: {self.threshold:.6f}")

    def detect(self, sample):
        """
        Returns (Is_Anomaly, Error_Score)
        """
        # Normalize single sample using training stats
        norm_sample = (np.array([sample]) - self.min_val) / (self.max_val - self.min_val)

        # Reconstruct
        reconstruction = self.model.predict(norm_sample, verbose=0)

        # Calculate MSE (Reconstruction Error)
        error = np.mean(np.square(norm_sample - reconstruction))

        is_anomaly = error > self.threshold
        return is_anomaly, error


def start_server():
    detector = AutoencoderDetector(input_dim=3)  # X, Y, Z

    print(f"Waiting for connection on port {PORT}...")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST_IP, PORT))
        s.listen()

        conn, addr = s.accept()
        with conn:
            print(f"Connected to Sender: {addr}")
            buffer = ""

            while True:
                data = conn.recv(1024)
                if not data: break

                buffer += data.decode('utf-8')

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if not line: continue

                    try:
                        packet = json.loads(line)
                        raw_features = [packet['x'], packet['y'], packet['z']]

                        # --- PHASE 1: CALIBRATION ---
                        if not detector.is_calibrated:
                            detector.training_data.append(raw_features)
                            count = len(detector.training_data)
                            print(f"Collecting Normal Data... {count}/{CALIBRATION_SAMPLES}", end='\r')

                            if count >= CALIBRATION_SAMPLES:
                                detector.train()
                                print("\n--- SYSTEM ARMED: DEEP LEARNING ACTIVE ---")

                        # --- PHASE 2: DETECTION ---
                        else:
                            is_anomaly, error = detector.detect(raw_features)

                            # Visuals
                            status = "🔴 ANOMALY" if is_anomaly else "🟢 NORMAL "
                            bar = "█" * int(error / detector.threshold * 5)  # Visual bar

                            print(f"{status} | Loss: {error:.5f} | Thr: {detector.threshold:.5f} | {bar}")

                    except json.JSONDecodeError:
                        pass
                    except Exception as e:
                        print(f"Error: {e}")


if __name__ == "__main__":
    start_server()