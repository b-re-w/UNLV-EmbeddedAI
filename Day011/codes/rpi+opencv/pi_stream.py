import subprocess
import time

# --- CONFIGURATION ---
PORT = 8554


def start_stream():
    """
    Streams raw H.264 video to the network using the Pi's hardware encoder.
    No OpenCV is required on the Pi.
    """
    print(f"Server listening on TCP port {PORT}...")
    print("Run the PC script now to connect.")

    # Select command based on OS version
    # 'rpicam-vid' is for Bookworm, 'libcamera-vid' for Bullseye
    cmd_base = "rpicam-vid"

    # Quick check to see if rpicam exists, else fallback
    try:
        subprocess.run(["which", "rpicam-vid"], check=True, stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        cmd_base = "libcamera-vid"

    # Command breakdown:
    # -t 0: Run forever
    # --inline: Add headers (SPS/PPS) so the stream can be joined anytime
    # --listen: Wait for the PC to connect to us
    # --width/height: 640x480 is optimal for low latency on Pi Zero
    # -o tcp://0.0.0.0:{PORT}: Output directly to network socket
    cmd = [
        cmd_base,
        "-t", "0",
        "--inline",
        "--listen",
        "--width", "640",
        "--height", "480",
        "--framerate", "30",
        "-o", f"tcp://0.0.0.0:{PORT}"
    ]

    try:
        # This will block and run until you press Ctrl+C
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("Stream stopped.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    start_stream()