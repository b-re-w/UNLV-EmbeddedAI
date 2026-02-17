#include <M5Unified.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <math.h>

// --- Configuration ---
const char* ssid     = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";
const char* host_ip  = "192.168.1.XX"; // The IP of your PC
const int udp_port   = 12345;

WiFiUDP udp;
float qw = 1.0f, qx = 0.0f, qy = 0.0f, qz = 0.0f;
unsigned long lastUpdate = 0;
const float beta = 0.02f;

void setup() {
    auto cfg = M5.config();
    M5.begin(cfg);
    WiFi.begin();
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        M5.Display.print(".");
    }
    M5.Display.println("\nWiFi Connected");
    M5.Display.println(WiFi.localIP());
}

void loop() {
    M5.update();

    unsigned long now = micros();
    float dt = (now - lastUpdate) / 1000000.0f; // [cite: 22]
    lastUpdate = now;

    float ax, ay, az, gx, gy, gz;
    M5.Imu.getAccel(&ax, &ay, &az); // [cite: 23]
    M5.Imu.getGyro(&gx, &gy, &gz); // [cite: 24]

    // Manual Quaternion Math [cite: 25, 26, 30, 31, 32]
    float gx_rad = gx * M_PI / 180.0f;
    float gy_rad = gy * M_PI / 180.0f;
    float gz_rad = gz * M_PI / 180.0f;

    float dqw = 0.5f * (-qx * gx_rad - qy * gy_rad - qz * gz_rad);
    float dqx = 0.5f * ( qw * gx_rad + qy * gz_rad - qz * gy_rad);
    float dqy = 0.5f * ( qw * gy_rad - qx * gz_rad + qz * gx_rad);
    float dqz = 0.5f * ( qw * gz_rad + qx * gy_rad - qy * gx_rad);

    qw += dqw * dt; qx += dqx * dt; qy += dqy * dt; qz += dqz * dt;

    float norm = sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
    qw /= norm; qx /= norm; qy /= norm; qz /= norm;

    // Accelerometer Correction [cite: 33, 34, 38, 42]
    float accel_norm = sqrt(ax*ax + ay*ay + az*az);
    if (accel_norm > 0.1f) {
        float target_pitch = atan2(-ax, sqrt(ay*ay + az*az));
        float target_roll  = atan2(ay, az);
        float cp = cos(target_pitch * 0.5f); float sp = sin(target_pitch * 0.5f);
        float cr = cos(target_roll * 0.5f); float sr = sin(target_roll * 0.5f);

        float aqw = cr * cp; float aqx = sr * cp; float aqy = cr * sp; float aqz = -sr * sp;

        qw = (1.0f - beta) * qw + beta * aqw;
        qx = (1.0f - beta) * qx + beta * aqx;
        qy = (1.0f - beta) * qy + beta * aqy;
        qz = (1.0f - beta) * qz + beta * aqz;
    }

    // Send UDP Packet
    udp.beginPacket(host_ip, udp_port);
    udp.printf("%lu,%.4f,%.4f,%.4f,%.4f", millis(), qw, qx, qy, qz);
    udp.endPacket();

    delay(10); // ~100Hz [cite: 44]
}