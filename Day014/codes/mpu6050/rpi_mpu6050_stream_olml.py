import smbus, time, socket, json

# --- CONFIG ---
HOST_IP = '192.168.1.100'  # CHANGE THIS to your Host PC IP
HOST_PORT = 65432
BUS_ADDR = 0x68

# --- SETUP MPU6050 ---
bus = smbus.SMBus(1)
bus.write_byte_data(BUS_ADDR, 0x6B, 0x00)  # Wake up


def read_word(reg):
    h = bus.read_byte_data(BUS_ADDR, reg)
    l = bus.read_byte_data(BUS_ADDR, reg + 1)
    val = (h << 8) | l
    return val - 65536 if val > 32768 else val


print(f"Connecting to {HOST_IP}...")
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST_IP, HOST_PORT))
    print("Streaming...")
    while True:
        # Read Accel
        ax = read_word(0x3B) / 16384.0
        ay = read_word(0x3D) / 16384.0
        az = read_word(0x3F) / 16384.0

        # Send JSON
        payload = json.dumps({'ax': ax, 'ay': ay, 'az': az}) + "\n"
        s.sendall(payload.encode('utf-8'))
        time.sleep(0.02)  # ~50Hz