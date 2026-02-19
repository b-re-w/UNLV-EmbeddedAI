from gtts import gTTS
from pydub import AudioSegment
from pydub.effects import speedup
import os

# Configuration
TEXT = "Tello"
OUTPUT_FILE = "tello_1s.wav"
TARGET_DURATION = 1000  # 1 second in milliseconds


def generate_audio():
    print(f"Generating audio for '{TEXT}'...")

    # 1. Generate Speech (MP3)
    tts = gTTS(text=TEXT, lang='en', slow=False)
    tts.save("temp.mp3")

    # 2. Convert to WAV & Process
    audio = AudioSegment.from_mp3("temp.mp3")

    # Trim silence from start and end
    def trim_silence(audio_segment, silence_thresh=-50):
        start_trim = 0
        end_trim = len(audio_segment)

        for i in range(len(audio_segment)):
            if audio_segment[i].dBFS > silence_thresh:
                start_trim = i
                break

        for i in range(len(audio_segment) - 1, 0, -1):
            if audio_segment[i].dBFS > silence_thresh:
                end_trim = i
                break

        return audio_segment[start_trim:end_trim]

    audio = trim_silence(audio)

    # 3. Adjust Duration to ~1s
    current_duration = len(audio)
    print(f"Original Duration: {current_duration}ms")

    if current_duration > TARGET_DURATION:
        # If too long, speed it up slightly (preserving pitch)
        speed_factor = current_duration / TARGET_DURATION
        # pydub's speedup is a bit rough, but works for small changes
        # Alternatively, simple crossfade cropping
        audio = audio[:TARGET_DURATION]
        print("Trimmed to 1s.")
    else:
        # If too short, add silence padding
        silence_needed = TARGET_DURATION - current_duration
        pad_start = silence_needed // 2
        pad_end = silence_needed - pad_start

        silence = AudioSegment.silent(duration=pad_start)
        audio = silence + audio + AudioSegment.silent(duration=pad_end)
        print(f"Padded with {silence_needed}ms silence.")

    # 4. Export
    # 16-bit PCM WAV (Standard for microcontrollers/embedded)
    audio.export(OUTPUT_FILE, format="wav", parameters=["-ac", "1", "-ar", "16000", "-sample_fmt", "s16"])

    # Cleanup
    if os.path.exists("temp.mp3"):
        os.remove("temp.mp3")

    print(f"Success! Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_audio()