import time
import socket
import json
from smbus2 import SMBus
from apds9960.const import *
from apds9960 import APDS9960

# --- CONFIGURATION ---
HOST_IP = '192.168.1.100'  # REPLACE with your Host PC IP Address
PORT = 65432  # Must match the host script
LOCATION_ID = "Lab_RPi_Zero"


def setup_sensor():
    bus = SMBus(1)
    apds = APDS9960(bus)
    apds.enableLightSensor()
    return apds


def main():
    # 1. Setup Sensor
    try:
        apds = setup_sensor()
        print("Sensor initialized.")
    except Exception as e:
        print(f"Error initializing sensor: {e}")
        return

    # 2. Connect to Host
    while True:
        try:
            print(f"Connecting to {HOST_IP}:{PORT}...")
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((HOST_IP, PORT))
            print("Connected!")
            break
        except ConnectionRefusedError:
            print("Host not found. Retrying in 5 seconds...")
            time.sleep(5)

    # 3. Stream Data
    try:
        while True:
            # Read RGB and Clear light
            r = apds.readRedLight()
            g = apds.readGreenLight()
            b = apds.readBlueLight()
            c = apds.readAmbientLight()

            # Create Payload
            payload = {
                "location": LOCATION_ID,
                "timestamp": time.time(),  # Unix timestamp
                "red": r,
                "green": g,
                "blue": b,
                "clear": c
            }

            # Send JSON with newline delimiter
            json_str = json.dumps(payload) + "\n"
            client_socket.sendall(json_str.encode('utf-8'))

            print(f"Sent: R:{r} G:{g} B:{b} C:{c}")
            time.sleep(1)  # 1Hz Sample Rate

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Connection Error: {e}")
    finally:
        client_socket.close()


if __name__ == "__main__":
    main()