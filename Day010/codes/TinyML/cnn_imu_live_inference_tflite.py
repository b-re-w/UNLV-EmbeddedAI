import serial
import serial.tools.list_ports
import numpy as np
import tensorflow as tf
from collections import deque
import sys
import time

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_PATH = "model.tflite"  # Your quantized TFLite file
CLASSES = ["static", "verti", "horiz", "circ", "type", "draw", "write"]

WINDOW_SIZE = 50  # Must match training
NUM_FEATURES = 4  # q0, q1, q2, q3 (Timestamp is ignored)
BAUD_RATE = 115200  # Match your Arduino Serial.begin
CONFIDENCE_THRESHOLD = 0.70


# ==========================================
# 2. TFLITE SETUP
# ==========================================
def load_tflite_model(path):
    print(f"Loading TFLite model: {path}...")
    try:
        # Load the TFLite interpreter
        interpreter = tf.lite.Interpreter(model_path=path)
        interpreter.allocate_tensors()

        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Check input requirements
        input_shape = input_details[0]['shape']
        print(f"Model Input Shape: {input_shape}")
        print(f"Input Type: {input_details[0]['dtype']}")

        return interpreter, input_details, output_details
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit()


# ==========================================
# 3. HELPER: QUANTIZATION
# ==========================================
# TFLite Int8 models expect inputs in int8 format (-128 to 127).
# We must manually convert our Float sensor data using the model's parameters.
def quantize_data(data, input_details):
    scale, zero_point = input_details[0]['quantization']

    # Formula: q = (real_value / scale) + zero_point
    # Note: We must handle the division and casting carefully
    if scale == 0: return data  # Fallback for non-quantized models

    q_data = (data / scale) + zero_point
    q_data = np.clip(q_data, -128, 127)  # Ensure it fits in int8
    return q_data.astype(np.int8)


def dequantize_output(output_data, output_details):
    scale, zero_point = output_details[0]['quantization']

    # Formula: real = (q - zero_point) * scale
    if scale == 0: return output_data

    return (output_data.astype(np.float32) - zero_point) * scale


# ==========================================
# 4. SERIAL PORT SELECTION
# ==========================================
def get_serial_port():
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No serial ports found!")
        sys.exit()

    print("\nAvailable Ports:")
    for i, p in enumerate(ports):
        print(f"[{i}] {p.device} - {p.description}")

    choice = input("Select port index: ")
    return ports[int(choice)].device


# ==========================================
# 5. MAIN LOOP
# ==========================================
def main():
    # Load Model
    interpreter, input_details, output_details = load_tflite_model(MODEL_PATH)
    input_index = input_details[0]['index']
    output_index = output_details[0]['index']

    # Connect Serial
    port = get_serial_port()
    ser = serial.Serial(port, BAUD_RATE, timeout=1)
    print(f"Connected to {port}. Waiting for data...")

    # Buffer: Deque handles sliding window automatically
    buffer = deque(maxlen=WINDOW_SIZE)

    print("\n--- STARTING INFERENCE ---\n")

    while True:
        try:
            if ser.in_waiting:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if not line: continue

                parts = line.split(',')
                # Expecting: ts, q0, q1, q2, q3 (5 values)
                if len(parts) == 5:
                    try:
                        # Parse Floats (Ignore timestamp at index 0)
                        q0 = float(parts[1])
                        q1 = float(parts[2])
                        q2 = float(parts[3])
                        q3 = float(parts[4])

                        # Add to buffer
                        buffer.append([q0, q1, q2, q3])

                        # Run Inference when buffer is full
                        if len(buffer) == WINDOW_SIZE:
                            # 1. Prepare Input Array (Batch=1, Steps=50, Features=4)
                            input_data = np.array(buffer, dtype=np.float32)
                            input_data = np.expand_dims(input_data, axis=0)

                            # 2. Manual Quantization (Float -> Int8)
                            # This is CRITICAL for Int8 TFLite models!
                            input_tensor = quantize_data(input_data, input_details)

                            # 3. Set Tensor & Invoke
                            interpreter.set_tensor(input_index, input_tensor)
                            interpreter.invoke()

                            # 4. Get Output & Dequantize (Int8 -> Float)
                            output_tensor = interpreter.get_tensor(output_index)
                            probabilities = dequantize_output(output_tensor, output_details)[0]

                            # 5. Process Result
                            class_idx = np.argmax(probabilities)
                            confidence = probabilities[class_idx]
                            label = CLASSES[class_idx]

                            # 6. Display
                            if confidence > CONFIDENCE_THRESHOLD:
                                # Overwrite line for clean output
                                print(f"\rAction: {label.upper()} ({confidence:.1%})        ", end="")
                            else:
                                print(f"\rAction: ... ({confidence:.1%})        ", end="")

                    except ValueError:
                        pass  # Corrupt line

        except KeyboardInterrupt:
            print("\nStopping...")
            break

    ser.close()


if __name__ == "__main__":
    main()