import smbus
import math
import time
import socket
import json

# --- CONFIGURATION ---
HOST_IP = '192.168.1.113'  # REPLACE with your Mac/Windows IP Address
HOST_PORT = 65432          # The port used by the server
SAMPLE_RATE = 0.05         # 20 Hz (0.05s delay)

# --- NATIVE MPU6050 CLASS ---
class MPU6050:
    def __init__(self, bus_id=1, addr=0x68):
        self.bus = smbus.SMBus(bus_id)
        self.addr = addr
        # Wake up the MPU6050 (it starts in sleep mode)
        self.bus.write_byte_data(self.addr, 0x6B, 0x00)

    def read_raw_data(self, addr):
        # Read two bytes (high and low) and combine them
        high = self.bus.read_byte_data(self.addr, addr)
        low = self.bus.read_byte_data(self.addr, addr + 1)
        value = (high << 8) | low
        # Convert to signed 16-bit integer
        if value > 32768:
            value = value - 65536
        return value

    def get_accel_data(self):
        # Register addresses for Accel X, Y, Z
        x = self.read_raw_data(0x3B)
        y = self.read_raw_data(0x3D)
        z = self.read_raw_data(0x3F)
        
        # Convert to G-force (Default sensitivity +/- 2g is 16384 LSB/g)
        ax = x / 16384.0
        ay = y / 16384.0
        az = z / 16384.0
        return ax, ay, az

# --- MAIN LOOP ---
def main():
    sensor = MPU6050()
    print(f"Connecting to Host {HOST_IP}:{HOST_PORT}...")

    # Create a TCP socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((HOST_IP, HOST_PORT))
            print("Connected! Streaming vibration data...")
            
            while True:
                ax, ay, az = sensor.get_accel_data()
                
                # Create a data packet
                data = {
                    "timestamp": time.time(),
                    "ax": round(ax, 3),
                    "ay": round(ay, 3),
                    "az": round(az, 3)
                }
                
                # Send as JSON string + newline (delimiter)
                message = json.dumps(data) + "\n"
                s.sendall(message.encode('utf-8'))
                
                time.sleep(SAMPLE_RATE)
                
        except ConnectionRefusedError:
            print("FAILED: Is the Host Server running?")
        except BrokenPipeError:
            print("Disconnected from Host.")
        except KeyboardInterrupt:
            print("\nStopping stream.")

if __name__ == "__main__":
    main()
