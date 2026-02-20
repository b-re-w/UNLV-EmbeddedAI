import smbus
import time

# --- CONFIGURATION ---
I2C_BUS = 1
DEVICE_ADDR = 0x39

# --- APDS9960 REGISTER MAP ---
REG_ENABLE  = 0x80
REG_ATIME   = 0x81
REG_CONTROL = 0x8F
REG_CDATAL  = 0x94 # Start of data registers (Clear Low)

class APDS9960:
    def __init__(self, bus_id=1, addr=0x39):
        try:
            self.bus = smbus.SMBus(bus_id)
            self.addr = addr
            self._setup()
        except Exception as e:
            print(f"Failed to initialize I2C: {e}")
            exit(1)

    def _setup(self):
        try:
            # 1. Power ON + Enable ALS (Ambient Light Sensor)
            # 0x03 = Power ON (Bit 0) + ALS Enable (Bit 1)
            self.bus.write_byte_data(self.addr, REG_ENABLE, 0x03)
            
            # 2. Set Integration Time
            # 0xD5 = ~100ms. Lower values = longer time/higher sensitivity.
            self.bus.write_byte_data(self.addr, REG_ATIME, 0xD5)
            
            # 3. Set Gain
            # 0x01 = 4x Gain. (0x00=1x, 0x01=4x, 0x02=16x, 0x03=60x)
            self.bus.write_byte_data(self.addr, REG_CONTROL, 0x01)
            
            print(f"Sensor initialized at address 0x{self.addr:02X}")
            time.sleep(0.5) # Allow sensor to gather first batch of light
            
        except OSError:
            print(f"Error: APDS9960 not found at 0x{self.addr:02X}. Check wiring.")
            exit(1)

    def read_rgbc(self):
        # Read 8 bytes: Clear(2), Red(2), Green(2), Blue(2)
        # Block read is faster than 4 separate reads
        data = self.bus.read_i2c_block_data(self.addr, REG_CDATAL, 8)
        
        # Combine Low and High bytes (Little Endian)
        c = data[1] << 8 | data[0]
        r = data[3] << 8 | data[2]
        g = data[5] << 8 | data[4]
        b = data[7] << 8 | data[6]
        
        return c, r, g, b

def main():
    sensor = APDS9960(I2C_BUS, DEVICE_ADDR)
    
    print("Reading APDS9960... (Press Ctrl+C to stop)")
    print("-" * 40)
    print(f"{'CLEAR':<8} {'RED':<8} {'GREEN':<8} {'BLUE':<8}")
    print("-" * 40)
    
    try:
        while True:
            c, r, g, b = sensor.read_rgbc()
            
            # Print formatted output using f-strings for alignment
            print(f"{c:<8} {r:<8} {g:<8} {b:<8}")
            
            time.sleep(0.5) # Read every 0.5 seconds
            
    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == "__main__":
    main()
