import socket
import csv
import time
import pandas as pd
import matplotlib.pyplot as plt

# --- Settings ---
UDP_IP = "0.0.0.0"  # Listen on all available interfaces
UDP_PORT = 12345
OUT_FILE = "quaternion_wifi_log.csv"
RECORD_SECONDS = 30

# Initialize Socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(2.0)

data_list = []
print(f"Listening for WiFi UDP packets on port {UDP_PORT}...")

try:
    start_time = time.time()
    while (time.time() - start_time) < RECORD_SECONDS:
        try:
            data, addr = sock.recvfrom(1024)
            line = data.decode('utf-8').strip()
            parts = line.split(',')

            if len(parts) == 5:  # ms, qw, qx, qy, qz
                data_list.append(parts)
        except socket.timeout:
            continue

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    sock.close()

# Save and Plot
if data_list:
    df = pd.DataFrame(data_list, columns=['ms', 'qw', 'qx', 'qy', 'qz'])
    df = df.apply(pd.to_numeric)
    df.to_csv(OUT_FILE, index=False)
    print(f"Saved {len(df)} samples to {OUT_FILE}")

    plt.figure(figsize=(12, 6))
    for col in ['qw', 'qx', 'qy', 'qz']:
        plt.plot(df['ms'], df[col], label=col)
    plt.title("WiFi UDP Quaternion Stream")
    plt.legend()
    plt.show()
else:
    print("No data received. Check IP/WiFi settings.")