import serial
import serial.tools.list_ports
import pandas as pd
import matplotlib.pyplot as plt
import time


def find_m5_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if "M5Core2_Quat" in p.description or "ESP32" in p.description:
            return p.device
    return None


BAUD = 115200
OUT_FILE = "quaternion_bt_log.csv"
RECORD_SECONDS = 30

port = find_m5_port()
if not port:
    print("M5Core2 Bluetooth not found!")
    exit()

data_list = []

try:
    ser = serial.Serial(port, BAUD, timeout=1)
    ser.reset_input_buffer()
    print(f"Recording Quaternions from {port} for {RECORD_SECONDS}s...")

    start_time = time.time()
    while (time.time() - start_time) < RECORD_SECONDS:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            parts = line.split(',')
            if len(parts) == 5:  # ms, qw, qx, qy, qz
                data_list.append(parts)

    ser.close()
except Exception as e:
    print(f"Error: {e}")

if data_list:
    df = pd.DataFrame(data_list, columns=['ms', 'qw', 'qx', 'qy', 'qz'])
    df = df.apply(pd.to_numeric)
    df.to_csv(OUT_FILE, index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(df['ms'], df['qw'], label='qw', color='black')
    plt.plot(df['ms'], df['qx'], label='qx', color='red')
    plt.plot(df['ms'], df['qy'], label='qy', color='green')
    plt.plot(df['ms'], df['qz'], label='qz', color='blue')
    plt.title("Bluetooth Quaternion Stream")
    plt.legend()
    plt.show()