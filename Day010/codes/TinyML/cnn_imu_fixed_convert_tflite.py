import tensorflow as tf
import numpy as np
import os
import glob
import pandas as pd

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_PATH = "cnn_imu_pruned_manual.h5"  # Or "cnn_imu.h5"
TFLITE_PATH = "cnn_pruned_model_fixed.tflite"
HEADER_PATH = "model_data_fixed.h"

DATA_FOLDER = "imu_class_data"
CLASSES = ["static", "verti", "horiz", "circ", "type", "draw", "write"]
WINDOW_SIZE = 50
STEP_SIZE = 25
NUM_FEATURES = 4


# ==========================================
# 2. LOAD REAL DATA (CRITICAL STEP)
# ==========================================
def load_calibration_data():
    X = []
    print("Loading calibration data from CSVs...")
    # We only need a small subset (e.g., 300-500 samples) to calibrate
    # But let's load enough to ensure good coverage of all classes
    for label in CLASSES:
        path = os.path.join(DATA_FOLDER, label, "*.csv")
        files = glob.glob(path)

        for f in files[:5]:  # Load first 5 files from each class
            try:
                df = pd.read_csv(f)
                data = df[['qw', 'qx', 'qy', 'qz']].values
                for i in range(0, len(data) - WINDOW_SIZE, STEP_SIZE):
                    if len(data[i:i + WINDOW_SIZE]) == WINDOW_SIZE:
                        X.append(data[i: i + WINDOW_SIZE])
            except:
                pass

    X = np.array(X, dtype=np.float32)
    print(f"Calibration Dataset Size: {len(X)} samples")
    return X


# Load the data
X_calibration = load_calibration_data()


# ==========================================
# 3. DEFINE THE GENERATOR
# ==========================================
def representative_dataset():
    # Loop through the real data
    # The converter only needs about 100-200 samples to find the ranges
    for i in range(min(300, len(X_calibration))):
        # Get one window
        data = X_calibration[i]
        # Add Batch Dimension: (50, 4) -> (1, 50, 4)
        data = np.expand_dims(data, axis=0)
        yield [data]


# ==========================================
# 4. CONVERT & QUANTIZE
# ==========================================
print("Loading H5 model...")
model = tf.keras.models.load_model(MODEL_PATH)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Attach the REAL data generator
converter.representative_dataset = representative_dataset

# Enforce Full Integer Quantization
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

print("Converting with Real Calibration Data...")
tflite_model = converter.convert()

# Save
with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print(f"Success! Saved to {TFLITE_PATH}")

# ==========================================
# 5. GENERATE C HEADER
# ==========================================
# (Standard Hex Dump)
hex_str = ", ".join([f"0x{b:02x}" for b in tflite_model])
header_content = f"""
#ifndef MODEL_DATA_H
#define MODEL_DATA_H
#include <stdint.h>
const unsigned char model_data[] = {{ {hex_str} }};
const unsigned int model_data_len = {len(tflite_model)};
#endif
"""
with open(HEADER_PATH, "w") as f:
    f.write(header_content)
print("Header generated.")