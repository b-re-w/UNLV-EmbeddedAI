import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import time

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATA_FOLDER = "imu_class_data"
CLASSES = ["static", "verti", "horiz", "circ", "type", "draw", "write"]
WINDOW_SIZE = 50
STEP_SIZE = 25
NUM_FEATURES = 4  # qw, qx, qy, qz

# Files generated from previous steps
MODEL_ORIGINAL = "cnn_imu.h5"  # From 1_train_standard.py
MODEL_PRUNED = "cnn_imu_pruned_manual.h5"  # From pruning script
MODEL_TFLITE = "cnn_pruned_model_fixed.tflite"  # From 2_convert_tflite.py


# ==========================================
# 2. LOAD TEST DATA (Same logic as training)
# ==========================================
def load_data():
    X, y = [], []
    print("Loading Test Data...")
    for idx, label in enumerate(CLASSES):
        path = os.path.join(DATA_FOLDER, label, "*.csv")
        files = glob.glob(path)
        for f in files:
            try:
                df = pd.read_csv(f)
                data = df[['qw', 'qx', 'qy', 'qz']].values
                for i in range(0, len(data) - WINDOW_SIZE, STEP_SIZE):
                    if len(data[i:i + WINDOW_SIZE]) == WINDOW_SIZE:
                        X.append(data[i: i + WINDOW_SIZE])
                        y.append(idx)
            except:
                pass
    return np.array(X, dtype=np.float32), np.array(y)


# Load and strictly use the TEST split for evaluation
X_all, y_all = load_data()
# Use random_state=42 to ensure this matches the split used during training
_, X_test, _, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

print(f"Test Set Size: {len(X_test)} samples")


# ==========================================
# 3. HELPER: TFLITE INFERENCE
# ==========================================
def evaluate_tflite(tflite_path, X_test):
    # Load Interpreter
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]['index']
    output_index = output_details[0]['index']

    # Check quantization params
    input_scale, input_zero = input_details[0]['quantization']
    output_scale, output_zero = output_details[0]['quantization']

    predictions = []

    start_time = time.time()

    for i in range(len(X_test)):
        # 1. Get single sample
        sample = X_test[i]  # Shape (50, 4)
        sample = np.expand_dims(sample, axis=0)  # Shape (1, 50, 4)

        # 2. Quantize Input (Float -> Int8)
        # q = (real / scale) + zero_point
        if input_scale != 0.0:
            sample = (sample / input_scale) + input_zero
        sample = np.clip(sample, -128, 127).astype(np.int8)

        # 3. Invoke
        interpreter.set_tensor(input_index, sample)
        interpreter.invoke()

        # 4. Get Output & Dequantize (Int8 -> Float)
        output = interpreter.get_tensor(output_index)
        if output_scale != 0.0:
            output = (output.astype(np.float32) - output_zero) * output_scale

        predictions.append(np.argmax(output))

    duration = time.time() - start_time
    return np.array(predictions), duration


# ==========================================
# 4. EVALUATION RUNNER
# ==========================================
results = []


def get_metrics(y_true, y_pred, model_name, time_taken):
    acc = accuracy_score(y_true, y_pred)
    # Weighted average accounts for class imbalance
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    return {
        "Model": model_name,
        "Accuracy": f"{acc:.4f}",
        "Precision": f"{prec:.4f}",
        "Recall": f"{rec:.4f}",
        "F1 Score": f"{f1:.4f}",
        "Time (sec)": f"{time_taken:.2f}"
    }


# --- A. Original Model ---
if os.path.exists(MODEL_ORIGINAL):
    print(f"\nEvaluating Original Model ({MODEL_ORIGINAL})...")
    model = tf.keras.models.load_model(MODEL_ORIGINAL)

    t0 = time.time()
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    t1 = time.time()

    results.append(get_metrics(y_test, y_pred, "Original (H5)", t1 - t0))

# --- B. Pruned Model ---
if os.path.exists(MODEL_PRUNED):
    print(f"Evaluating Pruned Model ({MODEL_PRUNED})...")
    # Note: If saved with pruning wrappers, you might need tfmot to load it.
    # Assuming standard save (strip_pruning was used):
    try:
        model_p = tf.keras.models.load_model(MODEL_PRUNED)

        t0 = time.time()
        y_pred_prob = model_p.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
        t1 = time.time()

        results.append(get_metrics(y_test, y_pred, "Pruned (H5)", t1 - t0))
    except:
        print("Could not load Pruned model. Did you run strip_pruning() before saving?")

# --- C. Quantized TFLite ---
if os.path.exists(MODEL_TFLITE):
    print(f"Evaluating Quantized TFLite ({MODEL_TFLITE})...")
    y_pred, duration = evaluate_tflite(MODEL_TFLITE, X_test)
    results.append(get_metrics(y_test, y_pred, "Quantized (Int8)", duration))

# ==========================================
# 5. DISPLAY RESULTS
# ==========================================
print("\n" + "=" * 80)
print(f"{'Model':<20} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10} | {'Time (s)':<10}")
print("-" * 80)

for res in results:
    print(
        f"{res['Model']:<20} | {res['Accuracy']:<10} | {res['Precision']:<10} | {res['Recall']:<10} | {res['F1 Score']:<10} | {res['Time (sec)']:<10}")
print("=" * 80 + "\n")

# Detailed Report for TFLite (Optional)
if os.path.exists(MODEL_TFLITE):
    print("Detailed Report for TFLite Model:")
    print(classification_report(y_test, y_pred, target_names=CLASSES))