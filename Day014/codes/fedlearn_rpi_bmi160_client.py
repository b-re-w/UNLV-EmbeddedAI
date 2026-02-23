import time
import numpy as np
import smbus2
from bmi160 import BMI160
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import flwr as fl

# --- CONFIGURATION ---
SERVER_IP = "192.168.1.100:8080"  # CHANGE THIS to your Host PC IP
BATCH_SIZE = 500  # How many samples to collect before training
EPOCHS_PER_ROUND = 2  # Local training epochs


# --- SENSOR SETUP ---
def get_sensor_data(n_samples):
    """
    Collects N samples from BMI160 to create a training dataset.
    Returns: Numpy array of shape (n_samples, 6) -> [ax, ay, az, gx, gy, gz]
    """
    i2c = smbus2.SMBus(1)
    bmi = BMI160(i2c)
    data_buffer = []

    print(f"Collecting {n_samples} samples from BMI160...", end="", flush=True)

    while len(data_buffer) < n_samples:
        try:
            # get_motion_6 returns (ax, ay, az, gx, gy, gz)
            raw = bmi.get_motion_6()
            # Simple normalization (approximate range for accel/gyro)
            # Adjust divisors based on your sensitivity settings
            norm_data = [
                raw[0] / 16384.0,  # Accel X
                raw[1] / 16384.0,  # Accel Y
                raw[2] / 16384.0,  # Accel Z
                raw[3] / 131.0,  # Gyro X
                raw[4] / 131.0,  # Gyro Y
                raw[5] / 131.0  # Gyro Z
            ]
            data_buffer.append(norm_data)
            time.sleep(0.01)  # 100Hz approx
        except Exception:
            pass

    print(" Done!")
    return np.array(data_buffer, dtype=np.float32)


# --- MODEL DEFINITION (Tiny Autoencoder) ---
def build_autoencoder():
    # Input: 6 features (Accel x,y,z + Gyro x,y,z)
    input_img = Input(shape=(6,))

    # Encoder: Compress to 3 neurons
    encoded = Dense(3, activation='relu')(input_img)

    # Decoder: Expand back to 6 neurons
    decoded = Dense(6, activation='tanh')(encoded)

    # Autoencoder maps Input -> Output
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder


# --- FLOWER CLIENT ---
class BMI160Client(fl.client.NumPyClient):
    def __init__(self, model):
        self.model = model
        self.x_train = np.array([])  # Placeholder

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        # 1. Update local model with global weights
        self.model.set_weights(parameters)

        # 2. Collect FRESH data from the sensor for this round
        # In FL, we train on data generated *locally* since the last round
        self.x_train = get_sensor_data(BATCH_SIZE)

        # 3. Train (Unsupervised: Target = Input)
        print("Training local model on sensor data...")
        history = self.model.fit(
            self.x_train,
            self.x_train,  # y = x for Autoencoder
            epochs=EPOCHS_PER_ROUND,
            batch_size=32,
            verbose=0
        )

        # 4. Return updated weights
        return self.model.get_weights(), len(self.x_train), {"loss": history.history['loss'][0]}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        # Collect a small batch for validation
        x_val = get_sensor_data(100)
        loss = self.model.evaluate(x_val, x_val, verbose=0)
        return loss, len(x_val), {"mse": loss}


# --- MAIN STARTUP ---
def main():
    # 1. Initialize Model
    model = build_autoencoder()

    # 2. Start Flower Client
    print(f"Connecting to FL Server at {SERVER_IP}...")
    fl.client.start_numpy_client(
        server_address=SERVER_IP,
        client=BMI160Client(model)
    )


if __name__ == "__main__":
    main()