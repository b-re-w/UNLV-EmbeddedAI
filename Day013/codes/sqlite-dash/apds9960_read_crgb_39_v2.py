import smbus
import time

# --- CONFIGURATION ---
I2C_BUS = 1
DEVICE_ADDR = 0x39  # The specific address you requested

# --- REGISTER MAP ---
REG_ENABLE  = 0x80
REG_ATIME   = 0x81
REG_CONTROL = 0x8F
REG_CDATAL  = 0x94  # Clear Data Low Byte

def main():
    try:
        bus = smbus.SMBus(I2C_BUS)
    except Exception as e:
        print(f"Failed to open I2C Bus {I2C_BUS}: {e}")
        return

    print(f"Reading APDS9960 at 0x{DEVICE_ADDR:02X}...")

    try:
        # 1. Power ON the sensor and Enable ALS (Ambient Light Sensor)
        # Write 0x03 (PON | AEN) to Enable Register (0x80)
        bus.write_byte_data(DEVICE_ADDR, REG_ENABLE, 0x03)
        
        # 2. Set Integration time (0xD5 = ~100ms)
        bus.write_byte_data(DEVICE_ADDR, REG_ATIME, 0xD5)

        # 3. Set Gain to 4x (0x01)
        bus.write_byte_data(DEVICE_ADDR, REG_CONTROL, 0x01)
        
        time.sleep(0.5) # Wait for sensor to boot and gather light

        while True:
            # 4. Read 8 bytes of data starting at CDATAL (0x94)
            # Order: Clear(L), Clear(H), Red(L), Red(H), Green(L), Green(H), Blue(L), Blue(H)
            data = bus.read_i2c_block_data(DEVICE_ADDR, REG_CDATAL, 8)
            
            # Convert bytes to 16-bit integers
            clear = data[1] << 8 | data[0]
            red   = data[3] << 8 | data[2]
            green = data[5] << 8 | data[4]
            blue  = data[7] << 8 | data[6]

            print(f"Clear: {clear}, Red: {red}, Green: {green}, Blue: {blue}")
            time.sleep(1)

    except OSError:
        print(f"Error: Device not found at address 0x{DEVICE_ADDR:02X}. Check wiring.")
    except KeyboardInterrupt:
        print("\nStopping.")

if __name__ == "__main__":
    main()
