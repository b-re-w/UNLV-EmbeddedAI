import time
import queue
import sounddevice as sd
import speech_recognition as sr
from djitellopy import Tello

# --- CONFIGURATION ---
SPEED = 30  # Drone speed in cm/s
DURATION = 3.5  # How long to listen for a command (seconds)
SAMPLE_RATE = 16000

# --- DRONE SETUP ---
print("[INFO] Connecting to Tello...")
tello = Tello()
try:
    tello.connect()
    tello.streamoff()
    print(f"[SUCCESS] Battery: {tello.get_battery()}%")
except Exception as e:
    print(f"[WARN] Connection failed (Simulating). Error: {e}")
    tello = None


# --- COMMAND MAPPING ---
def execute_command(text):
    if not text: return
    text = text.lower()
    print(f" >> RECOGNIZED: '{text}'")

    if not tello: return

    try:
        # Note: PocketSphinx often struggles with short words like "up".
        # We use longer phrases to improve accuracy.
        if "take off" in text or "takeoff" in text:
            tello.takeoff()
        elif "land" in text:
            tello.land()
        elif "go up" in text:
            tello.move_up(SPEED)
        elif "go down" in text:
            tello.move_down(SPEED)
        elif "go left" in text:
            tello.move_left(SPEED)
        elif "go right" in text:
            tello.move_right(SPEED)
        elif "go forward" in text:
            tello.move_forward(SPEED)
        elif "go back" in text:
            tello.move_back(SPEED)
        elif "flip" in text:
            tello.flip_forward()
        else:
            print(" -- Command not mapped")
    except Exception as e:
        print(f"Error: {e}")


# --- MAIN LOOP ---
print("\n" + "=" * 40)
print("  POCKETSPHINX CONTROL (OFFLINE)")
print("  Say: 'Take Off', 'Go Up', 'Land'...")
print("=" * 40 + "\n")

# Initialize Recognizer
r = sr.Recognizer()

try:
    while True:
        print("Listening...", end="", flush=True)

        # 1. Record Audio using SoundDevice (The Native Way)
        # We record a fixed chunk of audio (e.g., 3.5 seconds) into a numpy array
        recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
        sd.wait()  # Wait for recording to finish
        print(" Processing...")

        # 2. Convert Raw Audio to SpeechRecognition AudioData
        # (This bridges sounddevice -> pocketsphinx)
        audio_data = sr.AudioData(recording.tobytes(), SAMPLE_RATE, 2)  # 2 bytes width = int16

        # 3. Recognize using PocketSphinx
        try:
            # recognize_sphinx works offline
            command = r.recognize_sphinx(audio_data)
            execute_command(command)
        except sr.UnknownValueError:
            # Sphinx didn't understand the audio
            pass
        except sr.RequestError as e:
            print(f"Sphinx Error: {e}")

except KeyboardInterrupt:
    print("\nStopping...")
    if tello: tello.end()