import socket

PORT = 65432

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(('0.0.0.0', PORT))
    s.listen()
    print("Waiting for image...")
    conn, addr = s.accept()
    with conn:
        with open('received_py.jpg', 'wb') as f:
            while True:
                data = conn.recv(4096)
                if not data: break
                f.write(data)
    print("Image saved as received_py.jpg")