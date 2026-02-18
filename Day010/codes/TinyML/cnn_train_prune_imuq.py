import os
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from sklearn.model_selection import train_test_split
import tempfile
import zipfile

# --- CONFIGURATION ---
DATA_FOLDER = "imu_class_data"
CLASSES = ["static", "verti", "horiz", "circ", "type", "draw", "write"]
WINDOW_SIZE = 50   # 50 samples per window
STEP_SIZE = 25     # 50% overlap
NUM_FEATURES = 4   # qw, qx, qy, qz
EPOCHS = 20

# --- 1. DATA LOADING ---
def load_data():
    X, y = [], []
    print("Loading Data...")
    for idx, label in enumerate(CLASSES):
        path = os.path.join(DATA_FOLDER, label, "*.csv")
        files = glob.glob(path)
        for f in files:
            try:
                df = pd.read_csv(f)
                # Ensure columns are ordered: qw, qx, qy, qz
                data = df[['qw', 'qx', 'qy', 'qz']].values
                # Sliding Window
                for i in range(0, len(data) - WINDOW_SIZE, STEP_SIZE):
                    X.append(data[i : i + WINDOW_SIZE])
                    y.append(idx)
            except Exception as e:
                print(f"Skipping {f}: {e}")
    return np.array(X), np.array(y)

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data Shape: {X_train.shape}")

# --- 2. DEFINE CNN MODEL ---
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(WINDOW_SIZE, NUM_FEATURES)),
        tf.keras.layers.Conv1D(16, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(len(CLASSES), activation='softmax')
    ])
    return model

# --- 3. APPLY PRUNING ---
# Pruning removes unnecessary connections (weights -> 0)
base_model = create_model()

# Define Pruning Schedule (50% -> 80% sparsity)
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.50, final_sparsity=0.80, begin_step=0, end_step=len(X_train)//32 * EPOCHS
    )
}

model_pruned = tfmot.sparsity.keras.prune_low_magnitude(base_model, **pruning_params)

model_pruned.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train (Fine-tune with pruning callbacks)
callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
print("\nTraining with Pruning...")
model_pruned.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_test, y_test), callbacks=callbacks)

# --- 4. STRIP & SAVE ---
# Remove pruning wrappers to get a standard Keras model back
final_model = tfmot.sparsity.keras.strip_pruning(model_pruned)
final_model.save("cnn_imu_pruned.h5")

# --- 5. SHOW EFFECTIVENESS ---
def get_gzipped_size(file):
    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(file)
    return os.path.getsize(zipped_file)

print(f"\nOriginal H5 Size: {os.path.getsize('cnn_imu_pruned.h5')/1024:.2f} KB")
print(f"Compressed (Gzip) Size: {get_gzipped_size('cnn_imu_pruned.h5')/1024:.2f} KB")
print("Note: The 'Compressed' size reflects the true benefit of pruning (sparse weights compress better).")