#include <M5Unified.h>
#include <BluetoothSerial.h>

BluetoothSerial SerialBT;

// Timing
unsigned long lastUpdate = 0;
const unsigned long interval = 20000; // 50Hz (20ms)

void setup() {
    auto cfg = M5.config();
    M5.begin(cfg);

    // Initialize Bluetooth with a name
    // You will look for "M5Core2_IMU" in your PC Bluetooth settings
    if (!SerialBT.begin("M5Core2_IMU")) {
        M5.Display.println("BT Init Failed!");
        while (1);
    }

    M5.Display.setTextDatum(middle_center);
    M5.Display.setFont(&fonts::FreeSansBold12pt7b);
    M5.Display.println("BT Streaming...");
    M5.Display.println("Target: M5Core2_IMU");
}

void loop() {
    M5.update();

    unsigned long now = micros();
    if (now - lastUpdate >= interval) {
        lastUpdate = now;

        float ax, ay, az, gx, gy, gz;
        M5.Imu.getAccel(&ax, &ay, &az);
        M5.Imu.getGyro(&gx, &gy, &gz);

        // Format: Timestamp, ax, ay, az, gx, gy, gz
        // We use SerialBT.printf just like standard Serial
        SerialBT.printf("%lu,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n", 
                        millis(), ax, ay, az, gx, gy, gz);
                        
        // Also print to USB for debugging
        Serial.printf("%lu,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n", 
                      millis(), ax, ay, az, gx, gy, gz);
    }
}