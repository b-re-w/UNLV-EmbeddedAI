import serial
import serial.tools.list_ports
import pandas as pd
import time

# --- Settings ---
BAUD = 921600
OUT_FILE = "mesh_quat_data.csv"
RECORD_SECONDS = 30

def find_esp_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if any(x in p.description for x in ["USB", "CP210", "CH340"]):
            return p.device
    return None

port = find_esp_port()
if not port:
    print("Receiver ESP32 not found!")
    exit()

data_list = []

try:
    ser = serial.Serial(port, BAUD, timeout=1)
    ser.reset_input_buffer()
    print(f"Recording Multi-Device ESP-NOW stream for {RECORD_SECONDS}s...")

    start_time = time.time()
    while (time.time() - start_time) < RECORD_SECONDS:
        if ser.in_waiting > 0:
            try:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                parts = line.split(',')
                # Check for 6 values: ID, ms, qw, qx, qy, qz
                if len(parts) == 6:
                    data_list.append(parts)
            except:
                continue
    ser.close()
except Exception as e:
    print(f"Error: {e}")

if data_list:
    df = pd.DataFrame(data_list, columns=['id', 'ms', 'qw', 'qx', 'qy', 'qz'])
    df = df.apply(pd.to_numeric)
    df.to_csv(OUT_FILE, index=False)
    print(f"Capture Finished! Saved {len(df)} total samples.")
    print(f"Unique Devices Detected: {df['id'].unique()}")
else:
    print("No data captured. Ensure transmitters are powered and on the same channel.")