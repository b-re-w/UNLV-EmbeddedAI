import subprocess
import time
import os

# --- CONFIGURATION ---
PORT = 8554


def start_stream():
    """
    Starts the native Raspberry Pi camera stream (H.264) via TCP.
    """
    print(f"Starting Camera Stream on Port {PORT}...")
    print("Waiting for connection from PC...")

    # Construct the command
    # -t 0: Run forever
    # --inline: Insert headers so the PC can join the stream at any time
    # --listen: Wait for the PC to initiate the connection
    # -o tcp://0.0.0.0:{PORT}: Output raw bytes to the network

    # Check if we should use the new 'rpicam' or old 'libcamera' command
    cmd_base = "rpicam-vid"

    # Simple check: try to find the command, fallback if missing
    if subprocess.call("which rpicam-vid > /dev/null 2>&1", shell=True) != 0:
        cmd_base = "libcamera-vid"

    cmd = [
        cmd_base,
        "-t", "0",  # No timeout (run forever)
        "--inline",  # PPS/SPS headers for streaming
        "--listen",  # Server mode
        "--width", "640",  # Low res for low latency
        "--height", "480",
        "--framerate", "30",
        "-o", f"tcp://0.0.0.0:{PORT}"  # Output to TCP
    ]

    try:
        # Run the command as a subprocess
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nStopping stream...")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    start_stream()