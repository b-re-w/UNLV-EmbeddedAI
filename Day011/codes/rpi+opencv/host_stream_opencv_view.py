import socket
import struct
import cv2
import numpy as np
import io

PORT = 8000

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('0.0.0.0', PORT))
server_socket.listen(0)

print(f"Listening on port {PORT}...")
conn, addr = server_socket.accept()
connection = conn.makefile('rb')

try:
    while True:
        # 1. Read the length of the image (4 bytes)
        image_len_data = connection.read(4)
        if not image_len_data: break

        image_len = struct.unpack('<L', image_len_data)[0]

        # 2. Read the image data based on the length
        image_stream = io.BytesIO()
        image_stream.write(connection.read(image_len))

        # 3. Rewind stream and decode
        image_stream.seek(0)
        data = np.frombuffer(image_stream.getvalue(), dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)

        if frame is not None:
            # --- OPEN CV PROCESSING EXAMPLE ---
            # Convert to Grayscale for demo
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cv2.imshow('RPi Stream (Gray)', gray_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(e)
finally:
    conn.close()
    server_socket.close()
    cv2.destroyAllWindows()