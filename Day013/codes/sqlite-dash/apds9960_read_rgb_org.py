from apds9960.const import *
from apds9960 import APDS9960
import smbus
import time

# 1. Initialize I2C Bus (Bus 1 is standard for Raspberry Pi)
bus = smbus.SMBus(1)

# 2. Initialize the Sensor
apds = APDS9960(bus)


def setup():
    print("Initializing APDS-9960...")

    # Enable the Light Sensor (ALS)
    # This automatically turns on the power (PON) and ALS Enable (AEN)
    apds.enableLightSensor()

    # Optional: Adjust Gain and Integration Time for clarity
    # GAIN_1X, GAIN_4X, GAIN_16X, GAIN_64X
    apds.setAmbientLightGain(APDS9960_AGAIN_4X)

    # Integration time affects resolution.
    # higher time = higher resolution (values up to 65535)
    # APDS9960_ATIME_219MS is a good balance
    # (Default is often fast/low res, setting it explicitly helps)
    # Note: Some libraries might not expose setAmbientLightIntTime directly
    # without looking up the constant map, but defaults usually work.

    print("Sensor Ready. Reading RGB + Clear values...")
    print("Press Ctrl+C to stop.")


def loop():
    while True:
        # 3. Read raw 16-bit values (0-65535)
        # 'Clear' is the total intensity (ambient light)
        clear_val = apds.readAmbientLight()
        red_val = apds.readRedLight()
        green_val = apds.readGreenLight()
        blue_val = apds.readBlueLight()

        # 4. Print Data
        print(f"Clear: {clear_val} | R: {red_val} G: {green_val} B: {blue_val}")

        # Wait a bit before next read
        time.sleep(0.5)


if __name__ == "__main__":
    try:
        setup()
        loop()
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"\n[Error] {e}")
        print("Check your wiring: 3.3V, GND, SDA, SCL")
