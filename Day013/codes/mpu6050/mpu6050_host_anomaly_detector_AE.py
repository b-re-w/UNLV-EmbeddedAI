import socket
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.preprocessing import MinMaxScaler
import os

# Disable TF info logs for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- CONFIGURATION ---
HOST_IP = '0.0.0.0'  # Listen on all interfaces
HOST_PORT = 65432  # Must match RPi
BUFFER_SIZE = 1000  # Samples to collect for training (Normal Behavior)
EPOCHS = 10  # How long to train the neural network


class AutoencoderDetector:
    def __init__(self):
        self.buffer = []
        self.model = None
        self.threshold = 0.0
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_trained = False

    def build_model(self, input_dim):
        # A simple Autoencoder architecture
        # Input (3) -> Compressed (2) -> Reconstructed (3)
        input_layer = Input(shape=(input_dim,))

        # Encoder
        encoded = Dense(2, activation='relu')(input_layer)

        # Decoder
        decoded = Dense(input_dim, activation='sigmoid')(encoded)

        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder

    def process(self, data_point):
        # Extract features [ax, ay, az]
        features = [data_point['ax'], data_point['ay'], data_point['az']]

        # PHASE 1: Data Collection
        if not self.is_trained:
            self.buffer.append(features)
            progress = (len(self.buffer) / BUFFER_SIZE) * 100

            if len(self.buffer) >= BUFFER_SIZE:
                return self.train()
            else:
                return f"[Calibration] Collecting normal data... {int(progress)}%"

        # PHASE 2: Detection
        else:
            # 1. Scale input
            X = np.array([features])
            X_scaled = self.scaler.transform(X)

            # 2. Reconstruct
            reconstruction = self.model.predict(X_scaled, verbose=0)

            # 3. Calculate Error (Mean Squared Error)
            loss = np.mean(np.power(X_scaled - reconstruction, 2))

            # 4. Compare to Threshold
            status = f"Loss: {loss:.5f} | Threshold: {self.threshold:.5f}"

            if loss > self.threshold:
                return f"🔴 ANOMALY! {status}"
            else:
                return f"🟢 Normal. {status}"

    def train(self):
        print("\n[System] Training Autoencoder on normal data...")

        # Prepare Data
        data = np.array(self.buffer)

        # Fit Scaler (Normalize data between 0 and 1)
        self.scaler.fit(data)
        data_scaled = self.scaler.transform(data)

        # Build & Train Model
        self.model = self.build_model(input_dim=3)

        # Train the model to reconstruct the input (Input = Output)
        history = self.model.fit(
            data_scaled, data_scaled,
            epochs=EPOCHS,
            batch_size=32,
            shuffle=True,
            verbose=0
        )

        # Calculate Threshold:
        # We look at the "Reconstruction Error" on the training set.
        # Threshold = Max Training Loss + small buffer
        reconstructions = self.model.predict(data_scaled, verbose=0)
        train_loss = np.mean(np.power(data_scaled - reconstructions, 2), axis=1)

        # Set threshold to the worst error seen during training
        self.threshold = np.max(train_loss) * 1.1  # 10% safety margin

        self.is_trained = True
        return f"[System] Training Complete. Threshold set to {self.threshold:.5f}"


def main():
    print(f"--- Deep Learning Host Server Listening on {HOST_PORT} ---")
    detector = AutoencoderDetector()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST_IP, HOST_PORT))
        s.listen()
        conn, addr = s.accept()

        with conn:
            print(f"Connected by {addr}")
            buffer = ""

            while True:
                try:
                    chunk = conn.recv(1024).decode('utf-8')
                    if not chunk: break
                    buffer += chunk

                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        if line:
                            data = json.loads(line)
                            msg = detector.process(data)
                            print(f"\r{msg.ljust(80)}", end="")

                except Exception as e:
                    print(e)
                    break


if __name__ == "__main__":
    main()