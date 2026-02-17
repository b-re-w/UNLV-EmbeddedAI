import tensorflow as tf
import numpy as np
import os

# --- CONFIGURATION ---
MODEL_PATH = "cnn_imu_pruned_manual.h5"
TFLITE_PATH = "cnn_pruned_model.tflite"
HEADER_PATH = "model_data.h"
WINDOW_SIZE = 50
NUM_FEATURES = 4

# a) Read H5 Model
print(f"Loading {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)

# --- b) CONVERT & QUANTIZE (Float32 -> Int8) ---
def representative_dataset():
    # Generator function for the converter to measure dynamic range
    for _ in range(100):
        # Create random dummy data matching input shape (1, 50, 4)
        # In production, use REAL data from X_train here!
        data = np.random.rand(1, WINDOW_SIZE, NUM_FEATURES).astype(np.float32)
        yield [data]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset

# Enforce Full Integer Quantization (Best for ESP32)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# d) Convert to TFLite
tflite_model = converter.convert()

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

# c) Show Compression Results
h5_size = os.path.getsize(MODEL_PATH)
tflite_size = os.path.getsize(TFLITE_PATH)
print(f"\nOriginal H5 Size: {h5_size/1024:.2f} KB")
print(f"Quantized TFLite Size: {tflite_size/1024:.2f} KB")
print(f"Reduction: {(1 - tflite_size/h5_size)*100:.1f}%")

# --- e) GENERATE C HEADER (Hex Dump) ---
print(f"\nGenerating {HEADER_PATH}...")
with open(TFLITE_PATH, "rb") as f:
    bytes_data = f.read()

hex_str = ", ".join([f"0x{b:02x}" for b in bytes_data])
header_content = f"""
#ifndef MODEL_DATA_H
#define MODEL_DATA_H

#include <stdint.h>

const unsigned char model_data[] = {{ {hex_str} }};
const unsigned int model_data_len = {len(bytes_data)};

#endif
"""

with open(HEADER_PATH, "w") as f:
    f.write(header_content)

print("Done! Copy 'model_data.h' to your Arduino sketch folder.")