#include <M5Unified.h>
#include <BluetoothSerial.h>
#include <math.h>

BluetoothSerial SerialBT;

// Quaternion state variables [cite: 18]
float qw = 1.0f, qx = 0.0f, qy = 0.0f, qz = 0.0f;
unsigned long lastUpdate = 0;
const float beta = 0.02f; // Accel influence for drift correction

void setup() {
    auto cfg = M5.config();
    M5.begin(cfg);

    // Initialize Bluetooth
    if (!SerialBT.begin("M5Core2_Quat_IMU")) {
        M5.Display.println("BT Init Failed!");
        while (1);
    }

    M5.Display.setTextDatum(middle_center);
    M5.Display.println("BT Quat Stream...");
    M5.Display.println("Target: M5Core2_Quat_IMU");
}

void loop() {
    M5.update();

    unsigned long now = micros();
    float dt = (now - lastUpdate) / 1000000.0f;
    lastUpdate = now;

    float ax, ay, az, gx, gy, gz;
    M5.Imu.getAccel(&ax, &ay, &az);
    M5.Imu.getGyro(&gx, &gy, &gz);

    // Convert Gyro to rad/s
    float gx_rad = gx * M_PI / 180.0f;
    float gy_rad = gy * M_PI / 180.0f;
    float gz_rad = gz * M_PI / 180.0f;

    // 1. Gyro Integration
    float dqw = 0.5f * (-qx * gx_rad - qy * gy_rad - qz * gz_rad);
    float dqx = 0.5f * ( qw * gx_rad + qy * gz_rad - qz * gy_rad);
    float dqy = 0.5f * ( qw * gy_rad - qx * gz_rad + qz * gx_rad);
    float dqz = 0.5f * ( qw * gz_rad + qx * gy_rad - qy * gx_rad);

    qw += dqw * dt;
    qx += dqx * dt;
    qy += dqy * dt;
    qz += dqz * dt;

    // 2. Normalization
    float norm = sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
    qw /= norm; qx /= norm; qy /= norm; qz /= norm;

    // 3. Accelerometer Correction
    float accel_norm = sqrt(ax*ax + ay*ay + az*az);
    if (accel_norm > 0.1f) {
        float target_pitch = atan2(-ax, sqrt(ay*ay + az*az));
        float target_roll  = atan2(ay, az);

        float cp = cos(target_pitch * 0.5f);
        float sp = sin(target_pitch * 0.5f);
        float cr = cos(target_roll * 0.5f);
        float sr = sin(target_roll * 0.5f);

        float aqw = cr * cp;
        float aqx = sr * cp;
        float aqy = cr * sp;
        float aqz = -sr * sp;

        qw = (1.0f - beta) * qw + beta * aqw;
        qx = (1.0f - beta) * qx + beta * aqx;
        qy = (1.0f - beta) * qy + beta * aqy;
        qz = (1.0f - beta) * qz + beta * aqz;
    }

    // Stream over Bluetooth: ms, qw, qx, qy, qz
    SerialBT.printf("%lu,%.4f,%.4f,%.4f,%.4f\n", millis(), qw, qx, qy, qz);
    delay(20); // 50Hz stability
}