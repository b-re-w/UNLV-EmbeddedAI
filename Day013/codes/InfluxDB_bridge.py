import socket
import json
import datetime
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

# --- CONFIGURATION ---
HOST_IP = '0.0.0.0'  # Listen on all interfaces
PORT = 65432

# InfluxDB Settings (Match what you set up in UI)
INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "YOUR_ADMIN_TOKEN_HERE"
INFLUX_ORG = "UNLV_Lab"
INFLUX_BUCKET = "sensor_data"


def write_to_influx(data_dict):
    try:
        client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        write_api = client.write_api(write_options=SYNCHRONOUS)

        # Create Point
        p = Point("light_measurement") \
            .tag("location", data_dict.get("location", "unknown")) \
            .field("red", int(data_dict["red"])) \
            .field("green", int(data_dict["green"])) \
            .field("blue", int(data_dict["blue"])) \
            .field("clear", int(data_dict["clear"])) \
            .time(datetime.datetime.utcnow())

        write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=p)
        print(f"Written: R={data_dict['red']} G={data_dict['green']} B={data_dict['blue']}")

    except Exception as e:
        print(f"Database Error: {e}")


def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST_IP, PORT))
    server_socket.listen(5)
    print(f"Listening on {PORT}...")

    while True:
        client_socket, addr = server_socket.accept()
        # Read data (Buffer size 1024 bytes)
        data = client_socket.recv(1024)

        if data:
            try:
                # Decode and Parse JSON
                json_str = data.decode('utf-8').strip()
                data_dict = json.loads(json_str)
                write_to_influx(data_dict)
            except json.JSONDecodeError:
                print("Received invalid JSON")
            except Exception as e:
                print(f"Error: {e}")

        client_socket.close()


if __name__ == "__main__":
    start_server()