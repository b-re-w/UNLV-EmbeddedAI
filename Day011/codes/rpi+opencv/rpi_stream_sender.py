import socket
import struct
import time
import io
from picamera2 import Picamera2

# CONFIG
HOST_IP = '192.168.1.10'  # REPLACE WITH HOST PC IP
PORT = 8000

# Setup Camera (Picamera2)
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "XRGB8888"})
picam2.configure(config)
picam2.start()

# Connect to Host
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST_IP, PORT))
connection = client_socket.makefile('wb')

try:
    print("Streaming...")
    while True:
        # Capture standard JPEG
        # Note: capture_file is simpler for streaming loop than capture_buffer for direct bytes IO
        stream = io.BytesIO()
        picam2.capture_file(stream, format="jpeg")
        data = stream.getvalue()

        # Send Message: [Length of Image (4 bytes)][Image Data]
        # 'L' is unsigned long (4 bytes)
        connection.write(struct.pack('<L', len(data)))
        connection.write(data)

        # Simple throttle to prevent flooding (approx 30fps)
        # time.sleep(0.01)

except KeyboardInterrupt:
    pass
finally:
    connection.close()
    client_socket.close()
    picam2.stop()