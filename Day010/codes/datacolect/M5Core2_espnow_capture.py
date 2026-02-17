import serial
import serial.tools.list_ports
import pandas as pd
import time

# --- Settings ---
BAUD = 921600 # Matches Receiver
OUT_FILE = "esp_now_quats.csv"
RECORD_SECONDS = 30

def find_esp_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if "USB" in p.description or "CP210" in p.description or "CH340" in p.description:
            return p.device
    return None

port = find_esp_port()
if not port:
    print("Receiver not found!")
    exit()

data_list = []

try:
    ser = serial.Serial(port, BAUD, timeout=1)
    ser.reset_input_buffer()
    print(f"Recording ESP-NOW stream for {RECORD_SECONDS}s...")

    start_time = time.time()
    while (time.time() - start_time) < RECORD_SECONDS:
        if ser.in_waiting > 0:
            try:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                parts = line.split(',')
                if len(parts) == 5:
                    data_list.append(parts)
            except:
                continue
    ser.close()
except Exception as e:
    print(f"Error: {e}")

if data_list:
    df = pd.DataFrame(data_list, columns=['ms', 'qw', 'qx', 'qy', 'qz'])
    df = df.apply(pd.to_numeric)
    df.to_csv(OUT_FILE, index=False)
    print(f"Done! Saved {len(df)} samples to {OUT_FILE}")
else:
    print("No data captured.")