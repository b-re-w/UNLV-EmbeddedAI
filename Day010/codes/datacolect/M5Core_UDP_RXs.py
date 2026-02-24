import socket
import csv
import time
import pandas as pd
import matplotlib.pyplot as plt

# --- Settings ---
UDP_IP = "0.0.0.0"  # Listen on all available interfaces
UDP_PORT = 12345
OUT_FILE = "../quaternion_wifi_log_multi.csv"
RECORD_SECONDS = 10

# Initialize Socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(2.0)

data_list = []
print(f"Listening for WiFi UDP packets on port {UDP_PORT} for {RECORD_SECONDS} seconds...")

try:
    start_time = time.time()
    while (time.time() - start_time) < RECORD_SECONDS:
        try:
            data, addr = sock.recvfrom(1024)
            line = data.decode('utf-8').strip()
            parts = line.split(',')

            # Now expecting 6 parts: node_id, ms, qw, qx, qy, qz
            if len(parts) == 6:
                data_list.append(parts)
        except socket.timeout:
            continue

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    sock.close()

# Save and Plot
if data_list:
    # Build dataframe with the new node_id column
    df = pd.DataFrame(data_list, columns=['node_id', 'ms', 'qw', 'qx', 'qy', 'qz'])
    df = df.apply(pd.to_numeric)

    # Save combined data to CSV
    df.to_csv(OUT_FILE, index=False)
    print(f"Saved {len(df)} samples to {OUT_FILE}")

    # Discover unique transmitting nodes
    nodes = df['node_id'].unique()

    # Create subplots for each node dynamically
    fig, axes = plt.subplots(len(nodes), 1, figsize=(12, 6 * len(nodes)), sharex=False)

    # Force axes to be a list even if there's only 1 node transmitting
    if len(nodes) == 1:
        axes = [axes]

    # Plot data separately for each node
    for i, node in enumerate(nodes):
        node_df = df[df['node_id'] == node]
        for col in ['qw', 'qx', 'qy', 'qz']:
            axes[i].plot(node_df['ms'], node_df[col], label=col)

        axes[i].set_title(f"Node {int(node)} - WiFi UDP Quaternion Stream")
        axes[i].set_xlabel("Time (ms)")
        axes[i].set_ylabel("Quaternion Value")
        axes[i].legend()

    plt.tight_layout()
    plt.show()
else:
    print("No data received. Check IP/WiFi settings and ensure transmitters are running.")