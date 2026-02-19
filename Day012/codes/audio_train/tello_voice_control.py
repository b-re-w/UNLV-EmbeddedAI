import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
import queue
import time
import sys
from djitellopy import Tello

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_PATH = "audio_classifier.h5"
SAMPLE_RATE = 16000
DURATION = 1.0  # 1 second window
HOP_LENGTH = 512  # Must match training
N_FFT = 2048  # Must match training
N_MFCC = 13  # Must match training
CONFIDENCE_THRESHOLD = 0.85  # High confidence to avoid false triggers
PREDICTION_INTERVAL = 0.5  # How often to predict (seconds)

# Command Mapping
# Adjust indices based on your training classes list:
# CLASSES = ["up", "down", "on", "off", "left", "right", "tello", "_background_noise_"]
# IMPORTANT: Update this list to match your exact training label encoder order!
CLASSES = ["_background_noise_", "down", "left", "off", "on", "right", "tello", "up"]

# Tello State
is_flying = False
tello = None

# Audio Buffer
q = queue.Queue()


# ==========================================
# 2. TELLO SETUP
# ==========================================
def connect_tello():
    global tello
    try:
        print("Connecting to Tello...")
        tello = Tello()
        tello.connect()
        print(f"Battery: {tello.get_battery()}%")
        return True
    except Exception as e:
        print(f"Tello Connection Failed: {e}")
        return False


# ==========================================
# 3. AUDIO PROCESSING
# ==========================================
def audio_callback(indata, frames, time, status):
    """This is called for each audio block"""
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())


def extract_features(audio_buffer):
    # Ensure exactly 1 second (16000 samples)
    if len(audio_buffer) > SAMPLE_RATE:
        audio_buffer = audio_buffer[-SAMPLE_RATE:]
    elif len(audio_buffer) < SAMPLE_RATE:
        audio_buffer = np.pad(audio_buffer, (0, SAMPLE_RATE - len(audio_buffer)), 'constant')

    # Compute MFCC (Must match training parameters exactly)
    mfcc = librosa.feature.mfcc(y=audio_buffer, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)

    # Transpose to (Time, Feats)
    mfcc = mfcc.T

    # Add Batch and Channel dimensions: (1, Time, Feats, 1)
    mfcc = mfcc[np.newaxis, ..., np.newaxis]
    return mfcc


# ==========================================
# 4. MAIN CONTROL LOOP
# ==========================================
def main():
    global is_flying

    # 1. Load Model
    print(f"Loading model: {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)

    # 2. Connect Drone
    if not connect_tello():
        print("Continuing in Simulation Mode (No Drone Connected)")

    # 3. Start Stream
    print("\n--- VOICE CONTROL ACTIVE ---")
    print("Commands: 'TELLO ON' (Takeoff), 'TELLO OFF' (Land)")
    print("In-Flight: 'UP', 'DOWN', 'LEFT', 'RIGHT' (10cm moves)")

    # Buffer to hold raw audio
    raw_audio_buffer = np.zeros(0)

    with sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE):
        last_pred_time = time.time()

        while True:
            # Gather data from queue
            while not q.empty():
                data = q.get()
                raw_audio_buffer = np.concatenate((raw_audio_buffer, data.flatten()))

            # Keep only the last 1 second in buffer
            if len(raw_audio_buffer) > SAMPLE_RATE:
                raw_audio_buffer = raw_audio_buffer[-SAMPLE_RATE:]

            # Run Inference every PREDICTION_INTERVAL
            if time.time() - last_pred_time > PREDICTION_INTERVAL and len(raw_audio_buffer) == SAMPLE_RATE:
                last_pred_time = time.time()

                # Preprocess
                input_tensor = extract_features(raw_audio_buffer)

                # Predict
                prediction = model.predict(input_tensor, verbose=0)
                class_idx = np.argmax(prediction)
                confidence = prediction[0][class_idx]
                command = CLASSES[class_idx]

                # Filter low confidence or background noise
                if confidence > CONFIDENCE_THRESHOLD and command != "_background_noise_":
                    print(f"HEARD: {command.upper()} ({confidence:.2f})")
                    execute_command(command)
                else:
                    # Optional: Print silence/noise for debugging
                    # print(".", end="", flush=True)
                    pass


# ==========================================
# 5. COMMAND EXECUTION
# ==========================================
def execute_command(cmd):
    global is_flying

    if tello:
        try:
            # --- SECURITY: REQUIRE "TELLO" + COMMAND ---
            # For simplicity in this script, we treat 'on'/'off' as 'tello on'/'tello off'
            # If you trained "tello" as a separate keyword, you would need a state machine here.

            if cmd == "on" and not is_flying:
                print(">>> TAKING OFF")
                tello.takeoff()
                is_flying = True

            elif cmd == "off" and is_flying:
                print(">>> LANDING")
                tello.land()
                is_flying = False

            elif is_flying:
                # Movement Commands (Only work if flying)
                distance = 20  # cm (Safe indoor distance)

                if cmd == "up":
                    print(f">>> Moving UP {distance}cm")
                    tello.move_up(distance)
                elif cmd == "down":
                    print(f">>> Moving DOWN {distance}cm")
                    tello.move_down(distance)
                elif cmd == "left":
                    print(f">>> Moving LEFT {distance}cm")
                    tello.move_left(distance)
                elif cmd == "right":
                    print(f">>> Moving RIGHT {distance}cm")
                    tello.move_right(distance)
                elif cmd == "tello":
                    print(">>> 'TELLO' detected - Awaiting command...")

            else:
                print(f"Ignored '{cmd}' (Not flying or invalid state)")

        except Exception as e:
            print(f"Command Error: {e}")
    else:
        # Simulation Print
        print(f"[SIMULATION] Executing: {cmd}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopping...")
        if tello and is_flying:
            print("Emergency Landing!")
            tello.land()