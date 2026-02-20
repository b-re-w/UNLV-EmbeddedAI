import smbus
import time
import sqlite3
import os

# --- CONFIGURATION ---
# We use an ABSOLUTE PATH so we know exactly where the file is for the Host to find
DB_PATH = "/home/pi/sensor_data.db"
I2C_BUS = 1
DEVICE_ADDR = 0x39

# --- REGISTERS ---
REG_ENABLE = 0x80
REG_ATIME = 0x81
REG_CONTROL = 0x8F
REG_CDATAL = 0x94


class APDS9960:
    def __init__(self, bus_id=1, addr=0x39):
        self.bus = smbus.SMBus(bus_id)
        self.addr = addr
        self._setup()

    def _setup(self):
        try:
            self.bus.write_byte_data(self.addr, REG_ENABLE, 0x03)  # Power ON + ALS
            self.bus.write_byte_data(self.addr, REG_ATIME, 0xD5)  # 100ms integration
            self.bus.write_byte_data(self.addr, REG_CONTROL, 0x01)  # 4x Gain
            print("Sensor initialized.")
            time.sleep(0.5)
        except OSError:
            print("Error: APDS9960 not found. Check wiring.")

    def read_rgbc(self):
        data = self.bus.read_i2c_block_data(self.addr, REG_CDATAL, 8)
        c = data[1] << 8 | data[0]
        r = data[3] << 8 | data[2]
        g = data[5] << 8 | data[4]
        b = data[7] << 8 | data[6]
        return c, r, g, b


def init_db():
    conn = sqlite3.connect(DB_PATH)
    # WAL Mode allows reading (by Host download) while writing (by Sensor)
    conn.execute('PRAGMA journal_mode=WAL;')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS readings (
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            clear INTEGER,
            red INTEGER,
            green INTEGER,
            blue INTEGER
        )
    ''')
    conn.commit()
    return conn


def main():
    print(f"--- Logging to {DB_PATH} ---")
    sensor = APDS9960(I2C_BUS, DEVICE_ADDR)
    conn = init_db()
    cursor = conn.cursor()

    try:
        while True:
            c, r, g, b = sensor.read_rgbc()

            cursor.execute('''
                INSERT INTO readings (clear, red, green, blue)
                VALUES (?, ?, ?, ?)
            ''', (c, r, g, b))

            conn.commit()
            print(f"Logged: C={c} R={r} G={g} B={b}")
            time.sleep(1.0)  # Log every 1 second

    except KeyboardInterrupt:
        print("\nStopping.")
        conn.close()


if __name__ == "__main__":
    main()