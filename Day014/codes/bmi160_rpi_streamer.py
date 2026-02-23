import socket
import time
import json
from BMI160_i2c import Driver
import smbus2

# --- CONFIGURATION ---
HOST_IP = '192.168.1.136'  # CHANGE TO HOST PC IP
PORT = 65432



def main():
    # Setup I2C and Sensor
    #i2c = smbus2.SMBus(1)
    #bmi = BMI160(i2c)
    sensor = Driver(0x69)
    # Connect to Host
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        print(f"Connecting to {HOST_IP}...")
        sock.connect((HOST_IP, PORT))
        print("Connected! Streaming...")

        while True:
            # Read 6-axis data (returns ax, ay, az, gx, gy, gz)
            data = sensor.getMotion6()

            # payload = {
            #     "ax": data[0],
            #     "ay": data[1],
            #     "az": data[2],
            #     "timestamp": time.time()
            # }
            # Send JSON
            payload = json.dumps({'ax': data[0], 'ay': data[1], 'az': data[2]}) + "\n"
            sock.sendall(payload.encode('utf-8'))
            time.sleep(0.02)  # ~50Hz
            # Send newline-delimited JSON
            #msg = json.dumps(payload) + "\n"
            #sock.sendall(msg.encode('utf-8'))

            # 50Hz sample rate (fast enough for vibration)
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("Stopping...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        sock.close()


if __name__ == "__main__":
    main()