import serial
import serial.tools.list_ports
import csv
import pandas as pd
import matplotlib.pyplot as plt
import time


def find_m5_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        # Looking for the Bluetooth name or the common ESP32 driver description
        if "M5Core2_IMU" in p.description or "ESP32" in p.description or "Standard Serial over Bluetooth" in p.description:
            print(f"Found M5Core2 on port: {p.device}")
            return p.device
    return None


# --- Configuration ---
BAUD = 115200
OUT_FILE = "bt_imu_log.csv"
RECORD_SECONDS = 30

port = find_m5_port()
if not port:
    print("M5Core2 Bluetooth port not found. Make sure it's paired and 'Streaming'!")
    exit()

data_list = []

try:
    ser = serial.Serial(port, BAUD, timeout=1)
    ser.reset_input_buffer()
    print(f"Recording {RECORD_SECONDS} seconds via Bluetooth...")

    start_time = time.time()
    while (time.time() - start_time) < RECORD_SECONDS:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            parts = line.split(',')
            if len(parts) == 7:  # ms, ax, ay, az, gx, gy, gz
                data_list.append(parts)

    ser.close()
except Exception as e:
    print(f"Error: {e}")

# Save and Plot
if data_list:
    cols = ['ms', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']
    df = pd.DataFrame(data_list, columns=cols)
    df = df.apply(pd.to_numeric)
    df.to_csv(OUT_FILE, index=False)

    # Plotting Accelerometer for variety
    plt.figure(figsize=(10, 5))
    plt.plot(df['ms'], df['ax'], label='Accel X')
    plt.plot(df['ms'], df['ay'], label='Accel Y')
    plt.plot(df['ms'], df['az'], label='Accel Z')
    plt.title("Bluetooth IMU Stream - Accelerometer Data")
    plt.legend()
    plt.show()
else:
    print("No data received. Check Bluetooth pairing.")