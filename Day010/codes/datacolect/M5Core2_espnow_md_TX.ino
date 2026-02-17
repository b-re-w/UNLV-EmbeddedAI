#include <M5Unified.h>
#include <esp_now.h>
#include <WiFi.h>

// --- UNIQUE ID FOR THIS DEVICE ---
#define DEVICE_ID 1  // Set to 1, 2, or 3 for each unit

typedef struct struct_message {
    uint8_t id;      // Device Identifier
    uint32_t ms;     // Timestamp
    float qw, qx, qy, qz;
} struct_message;

struct_message myData;
uint8_t broadcastAddress[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

// Orientation variables
float qw = 1.0f, qx = 0.0f, qy = 0.0f, qz = 0.0f;
unsigned long lastUpdate = 0;

void setup() {
    auto cfg = M5.config();
    M5.begin(cfg);
    WiFi.begin();
    WiFi.mode(WIFI_STA);
    if (esp_now_init() != ESP_OK) return;

    esp_now_peer_info_t peerInfo = {};
    memcpy(peerInfo.peer_addr, broadcastAddress, 6);
    peerInfo.channel = 0;
    peerInfo.encrypt = false;
    esp_now_add_peer(&peerInfo);

    M5.Display.printf("Transmitter ID: %d\n", DEVICE_ID);
}

void loop() {
    M5.update();
    unsigned long now = micros();
    float dt = (now - lastUpdate) / 1000000.0f;
    lastUpdate = now;

    float ax, ay, az, gx, gy, gz;
    M5.Imu.getAccel(&ax, &ay, &az);
    M5.Imu.getGyro(&gx, &gy, &gz);

    // Quaternion Math (Simplified derivative integration)
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

    // Package Data
    myData.id = DEVICE_ID;
    myData.ms = millis();
    myData.qw = qw; myData.qx = qx; myData.qy = qy; myData.qz = qz;

    esp_now_send(broadcastAddress, (uint8_t *) &myData, sizeof(myData));
    delay(10); // 100Hz per device
}