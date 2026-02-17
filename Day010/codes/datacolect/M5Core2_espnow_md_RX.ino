#include <WiFi.h>
#include <esp_now.h>

typedef struct struct_message {
    uint8_t id;
    uint32_t ms;
    float qw, qx, qy, qz;
} struct_message;

struct_message incomingData;

// Callback function when data is received
void OnDataRecv(const uint8_t * mac, const uint8_t *incoming, int len) {
    if (len == sizeof(incomingData)) {
        memcpy(&incomingData, incoming, sizeof(incomingData));
        // Serial Format: ID, Timestamp, qw, qx, qy, qz
        Serial.printf("%d,%lu,%.4f,%.4f,%.4f,%.4f\n",
                      incomingData.id, incomingData.ms,
                      incomingData.qw, incomingData.qx,
                      incomingData.qy, incomingData.qz);
    }
}

void setup() {
    Serial.begin(921600); // High speed to handle multiple streams
    WiFi.begin();
    WiFi.mode(WIFI_STA);

    if (esp_now_init() != ESP_OK) return;
    esp_now_register_recv_cb(esp_now_recv_cb_t(OnDataRecv));
}

void loop() {
    // Receiver only reacts to callbacks
}