import socket
import time
from picamera2 import Picamera2

# CONFIG
HOST_IP = '192.168.1.10' # REPLACE WITH YOUR HOST PC IP
PORT = 65432

# Setup Camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (1920, 1080), "format": "XRGB8888"})
picam2.configure(config)
picam2.start()

# Warmup
time.sleep(2)

# Capture JPEG data in memory
jpg_data = picam2.capture_buffer("main", format="jpeg")

# Send via Socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST_IP, PORT))
    s.sendall(jpg_data)
    print("Image sent!")

picam2.stop()