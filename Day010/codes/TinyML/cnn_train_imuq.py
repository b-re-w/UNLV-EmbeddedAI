import os
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
DATA_FOLDER = "imu_class_data"
CLASSES = ["static", "verti", "horiz", "circ", "type", "draw", "write"]
WINDOW_SIZE = 50
STEP_SIZE = 25
NUM_FEATURES = 4  # qw, qx, qy, qz
EPOCHS = 20


# --- 1. DATA LOADING ---
def load_data():
    X, y = [], []
    print("Loading Data...")

    for idx, label in enumerate(CLASSES):
        path = os.path.join(DATA_FOLDER, label, "*.csv")
        files = glob.glob(path)

        if not files:
            print(f"Warning: No files found for class '{label}'")
            continue

        for f in files:
            try:
                df = pd.read_csv(f)
                # Ensure columns are ordered correctly
                # Adjust column names if your CSV header differs (e.g., 'QW', 'qw', etc.)
                data = df[['qw', 'qx', 'qy', 'qz']].values

                # Sliding Window
                for i in range(0, len(data) - WINDOW_SIZE, STEP_SIZE):
                    window = data[i: i + WINDOW_SIZE]
                    # Simple check to ensure full window
                    if len(window) == WINDOW_SIZE:
                        X.append(window)
                        y.append(idx)
            except Exception as e:
                print(f"Skipping {f}: {e}")

    return np.array(X), np.array(y)


# Load and Split
X, y = load_data()

if len(X) == 0:
    print("Error: No data loaded. Check folder structure and CSV headers.")
    exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training on {X_train.shape[0]} samples, Testing on {X_test.shape[0]} samples.")


# --- 2. DEFINE CNN MODEL ---
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(WINDOW_SIZE, NUM_FEATURES)),
        # 1D Conv is ideal for IMU time-series
        tf.keras.layers.Conv1D(16, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Dropout(0.1),

        tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same'),
        tf.keras.layers.GlobalAveragePooling1D(),

        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(len(CLASSES), activation='softmax')
    ])
    return model


model = create_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# --- 3. TRAIN ---
print("\nStarting Training...")
model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_test, y_test), batch_size=32)

# --- 4. SAVE MODEL ---
model_filename = "cnn_imu.h5"
model.save(model_filename)
print(f"\nSuccess! Model saved to {model_filename}")