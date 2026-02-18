import socket
import struct
import cv2
import numpy as np
import io
from ultralytics import YOLO

# Load YOLOv8 Nano model (small and fast for CPU)
model = YOLO("yolov8n.pt")

PORT = 8000

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('0.0.0.0', PORT))
server_socket.listen(0)

print(f"Listening for YOLO Stream on {PORT}...")
conn, addr = server_socket.accept()
connection = conn.makefile('rb')

try:
    while True:
        # Read payload length
        image_len_data = connection.read(4)
        if not image_len_data: break
        image_len = struct.unpack('<L', image_len_data)[0]

        # Read payload
        image_stream = io.BytesIO()
        image_stream.write(connection.read(image_len))

        # Decode image
        image_stream.seek(0)
        data = np.frombuffer(image_stream.getvalue(), dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)

        if frame is not None:
            # --- YOLO DETECTION ---
            # Run inference
            results = model(frame, verbose=False)

            # Plot results on the frame
            annotated_frame = results[0].plot()

            cv2.imshow('YOLOv8 Detection', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error: {e}")
finally:
    conn.close()
    server_socket.close()
    cv2.destroyAllWindows()