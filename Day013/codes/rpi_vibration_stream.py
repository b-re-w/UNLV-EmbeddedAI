import socket
import time
import json
import sys
from bmi160 import BMI160
import smbus2

# --- CONFIGURATION ---
HOST_IP = '192.168.1.100'  # REPLACE with your Laptop/Host PC IP
PORT = 65432  # Same port as receiver
SAMPLE_RATE = 0.05  # 20 Hz (Adjust as needed)


def get_imu_data(bmi):
    """Read raw acceleration data from BMI160."""
    try:
        data = bmi.get_motion_6()
        # Extract only accelerometer (x, y, z)
        # generic library returns (ax, ay, az, gx, gy, gz)
        return {
            "x": data[0],
            "y": data[1],
            "z": data[2],
            "timestamp": time.time()
        }
    except Exception as e:
        print(f"Sensor Error: {e}")
        return None


def main():
    # Setup I2C
    i2c = smbus2.SMBus(1)
    bmi = BMI160(i2c)

    # Initialize Socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        print(f"Connecting to Host {HOST_IP}:{PORT}...")
        sock.connect((HOST_IP, PORT))
        print("Connected! Streaming vibration data...")

        while True:
            packet = get_imu_data(bmi)

            if packet:
                # Serialize to JSON and send with a newline delimiter
                data_str = json.dumps(packet) + "\n"
                sock.sendall(data_str.encode('utf-8'))

            time.sleep(SAMPLE_RATE)

    except ConnectionRefusedError:
        print("Failed to connect. Is the Host script running?")
    except KeyboardInterrupt:
        print("Stopping stream...")
    finally:
        sock.close()


if __name__ == "__main__":
    main()