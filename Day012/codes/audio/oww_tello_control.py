import sys
import queue
import json
import numpy as np
import sounddevice as sd
from openwakeword.model import Model as OWWModel
from vosk import Model as VoskModel, KaldiRecognizer
from djitellopy import Tello

# --- CONFIGURATION ---
WAKE_WORD = "hey_jarvis"  # The trigger phrase
VOSK_PATH = "vosk_model"  # Path to your unzipped Vosk folder
SPEED = 30  # Drone speed (cm/s)

# --- GLOBAL STATE ---
q = queue.Queue()
is_command_mode = False  # False = Listening for Wake Word; True = Listening for Command


# --- AUDIO CALLBACK ---
# This runs in a separate thread to capture audio continuously
def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))


# --- DRONE CONTROL ---
def execute_command(text, drone):
    if not text: return
    text = text.lower()
    print(f" >> COMMAND RECOGNIZED: '{text}'")

    if not drone: return

    try:
        if "take off" in text:
            drone.takeoff()
        elif "land" in text:
            drone.land()
        elif "up" in text:
            drone.move_up(SPEED)
        elif "down" in text:
            drone.move_down(SPEED)
        elif "left" in text:
            drone.move_left(SPEED)
        elif "right" in text:
            drone.move_right(SPEED)
        elif "forward" in text:
            drone.move_forward(SPEED)
        elif "back" in text:
            drone.move_back(SPEED)
        elif "flip" in text:
            drone.flip_forward()
        elif "stop" in text:
            drone.send_rc_control(0, 0, 0, 0)
    except Exception as e:
        print(f"Command Error: {e}")


# --- MAIN ---
def main():
    global is_command_mode

    # 1. Setup Drone
    print("[INIT] Connecting to Tello...")
    drone = Tello()
    try:
        drone.connect()
        drone.streamoff()
        print(f"[SUCCESS] Tello Battery: {drone.get_battery()}%")
    except Exception as e:
        print(f"[WARN] Tello not found. Running in Simulation Mode. Error: {e}")
        drone = None

    # 2. Load AI Models
    print(f"[INIT] Loading OpenWakeWord ({WAKE_WORD})...")
    oww = OWWModel(wakeword_models=[WAKE_WORD])

    print(f"[INIT] Loading Vosk ({VOSK_PATH})...")
    if not os.path.exists(VOSK_PATH):
        print(f"[ERROR] Vosk model folder '{VOSK_PATH}' not found! Download it first.")
        return
    v_model = VoskModel(VOSK_PATH)
    v_rec = KaldiRecognizer(v_model, 16000)

    # 3. Start Microphone Stream
    # OpenWakeWord expects 16-bit PCM samples at 16kHz
    print("\n" + "=" * 50)
    print(f"  SYSTEM READY: Say '{WAKE_WORD}' then a command.")
    print("=" * 50 + "\n")

    with sd.RawInputStream(samplerate=16000, blocksize=1280, dtype='int16',
                           channels=1, callback=callback):
        while True:
            # Get audio chunk
            data = q.get()

            if not is_command_mode:
                # --- WAKE WORD DETECTION ---
                # Convert raw bytes to numpy array for OpenWakeWord
                audio_np = np.frombuffer(data, dtype=np.int16)

                # Predict
                prediction = oww.predict(audio_np)

                if prediction[WAKE_WORD] > 0.5:
                    print(f"\n[!] WAKE WORD DETECTED! Speak now...")
                    is_command_mode = True
                    v_rec.Reset()  # Clear previous buffer
            else:
                # --- COMMAND RECOGNITION (VOSK) ---
                if v_rec.AcceptWaveform(data):
                    # Full phrase recognized
                    result = json.loads(v_rec.Result())
                    cmd_text = result.get("text", "")

                    if cmd_text:
                        execute_command(cmd_text, drone)
                        print("[INFO] Returning to Wake Word mode...")
                        is_command_mode = False
                        oww.reset()  # Reset wake word buffer
                else:
                    # Partial result (optional print)
                    pass


import os

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopping...")