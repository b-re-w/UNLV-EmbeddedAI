import socket
import json
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime

# --- CONFIGURATION ---
HOST_IP = '0.0.0.0'  # Listen on all interfaces
PORT = 65432

# InfluxDB Config
URL = "http://localhost:8086"
TOKEN = "YOUR_API_TOKEN_HERE"  # Paste token from Step 3B
ORG = "UNLV_Lab"
BUCKET = "light_data"


def write_to_db(write_api, data):
    """Writes the JSON data to InfluxDB"""
    # Convert Unix timestamp to datetime object
    dt_object = datetime.fromtimestamp(data['timestamp'])

    point = Point("apds9960_measurement") \
        .tag("location", data['location']) \
        .field("red", int(data['red'])) \
        .field("green", int(data['green'])) \
        .field("blue", int(data['blue'])) \
        .field("clear", int(data['clear'])) \
        .time(dt_object)

    write_api.write(bucket=BUCKET, org=ORG, record=point)


def main():
    # Setup InfluxDB Client
    client = InfluxDBClient(url=URL, token=TOKEN, org=ORG)
    write_api = client.write_api(write_options=SYNCHRONOUS)

    # Setup Socket Server
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST_IP, PORT))
    server.listen(1)

    print(f"Listening for RPi on port {PORT}...")

    conn, addr = server.accept()
    with conn:
        print(f"Connected by {addr}")
        buffer = ""

        while True:
            chunk = conn.recv(1024)
            if not chunk: break

            buffer += chunk.decode('utf-8')

            # Handle newline delimited JSON
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                if not line: continue

                try:
                    data = json.loads(line)
                    write_to_db(write_api, data)
                    print(f"Logged: R={data['red']} G={data['green']} B={data['blue']}")
                except json.JSONDecodeError:
                    print("Error decoding JSON")
                except Exception as e:
                    print(f"Error: {e}")


if __name__ == "__main__":
    main()