#include <M5Unified.h>
#include <esp_now.h>
#include <WiFi.h>

// Structure to match the receiver
typedef struct struct_message {
    uint32_t ms;
    float qw, qx, qy, qz;
} struct_message;

struct_message myData;
// Change it to your receiver (refer to M5Core2_espnow_RX.ino)
uint8_t broadcastAddress[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF}; // Broadcast to all

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
}

void loop() {
    M5.update();
    unsigned long now = micros();
    float dt = (now - lastUpdate) / 1000000.0f;
    lastUpdate = now;

    float ax, ay, az, gx, gy, gz;
    M5.Imu.getAccel(&ax, &ay, &az);
    M5.Imu.getGyro(&gx, &gy, &gz);

    // Manual Quaternion Math
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

    // Prepare data
    myData.ms = millis();
    myData.qw = qw; myData.qx = qx; myData.qy = qy; myData.qz = qz;

    // Send via ESP-NOW
    esp_now_send(broadcastAddress, (uint8_t *) &myData, sizeof(myData));
    
    delay(5); // Ultra-fast ~200Hz stream
}