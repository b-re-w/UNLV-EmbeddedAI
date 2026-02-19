import sys
import json
import queue
import sounddevice as sd
from vosk import Model, KaldiRecognizer
from djitellopy import Tello

# --- Configuration ---
# Tello flight speed (cm/s)
SPEED = 30
# Model path (must be in the same folder)
MODEL_PATH = "model"

# --- Initialize Tello ---
print("[INFO] Connecting to Tello...")
tello = Tello()
try:
    tello.connect()
    tello.streamoff()
    print(f"[SUCCESS] Battery: {tello.get_battery()}%")
except Exception as e:
    print(f"[WARN] Tello connection failed: {e}")
    print("      (Running in Simulation Mode)")
    tello = None

# --- Audio Setup ---
# A thread-safe queue to pass audio from the callback to the main loop
q = queue.Queue()


def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))


# --- Command Execution ---
def parse_and_execute(text):
    if not text: return

    print(f" >> Heard: '{text}'")

    if not tello: return  # Skip if simulating

    try:
        if "take off" in text:
            tello.takeoff()
        elif "land" in text:
            tello.land()
        elif "up" in text:
            tello.move_up(SPEED)
        elif "down" in text:
            tello.move_down(SPEED)
        elif "left" in text:
            tello.move_left(SPEED)
        elif "right" in text:
            tello.move_right(SPEED)
        elif "forward" in text:
            tello.move_forward(SPEED)
        elif "back" in text:
            tello.move_back(SPEED)
        elif "stop" in text:
            tello.send_rc_control(0, 0, 0, 0)  # Emergency hover
        elif "flip" in text:
            tello.flip_forward()
    except Exception as e:
        print(f"Command Error: {e}")


# --- Main Loop ---
try:
    print(f"[INFO] Loading Vosk Model from '{MODEL_PATH}'...")
    model = Model(MODEL_PATH)
    rec = KaldiRecognizer(model, 16000)

    print("\n" + "=" * 40)
    print("  VOSK SYSTEM READY (Say 'Take Off')")
    print("=" * 40 + "\n")

    # Start the microphone stream
    # dtype='int16', channels=1, samplerate=16000 are required for Vosk
    with sd.RawInputStream(samplerate=16000, blocksize=8000, device=None,
                           dtype='int16', channels=1, callback=callback):
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                # Valid command phrase completed
                result = json.loads(rec.Result())
                parse_and_execute(result.get("text", ""))
            else:
                # Partial result (optional: use for even faster reaction)
                # partial = json.loads(rec.PartialResult())
                # print(partial["partial"])
                pass

except KeyboardInterrupt:
    print("\nDone.")
except Exception as e:
    print(f"\n[Error] {e}")
    print("Did you forget to download and rename the 'model' folder?")