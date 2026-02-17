#include <M5Unified.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLE2902.h>
#include <math.h>

// BLE UUIDs (Generate your own or use these defaults)
#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"

BLECharacteristic *pCharacteristic;
bool deviceConnected = false;
float qw = 1.0f, qx = 0.0f, qy = 0.0f, qz = 0.0f;
unsigned long lastUpdate = 0;

class MyServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) { deviceConnected = true; };
    void onDisconnect(BLEServer* pServer) {
      deviceConnected = false;
      pServer->getAdvertising()->start(); // Restart advertising so phone can find it again
    }
};

void setup() {
    auto cfg = M5.config();
    M5.begin(cfg);

    // Initialize BLE
    BLEDevice::init("M5Core2_Quat_Mobile");
    BLEServer *pServer = BLEDevice::createServer();
    pServer->setCallbacks(new MyServerCallbacks());
    BLEService *pService = pServer->createService(SERVICE_UUID);
    pCharacteristic = pService->createCharacteristic(
                        CHARACTERISTIC_UUID,
                        BLECharacteristic::PROPERTY_NOTIFY
                      );
    pCharacteristic->addDescriptor(new BLE2902());
    pService->start();
    pServer->getAdvertising()->start();

    M5.Display.println("BLE Ready: M5Core2_Quat_Mobile");
}

void loop() {
    M5.update();
    unsigned long now = micros();
    float dt = (now - lastUpdate) / 1000000.0f; [cite: 22]
    lastUpdate = now;

    float ax, ay, az, gx, gy, gz;
    M5.Imu.getAccel(&ax, &ay, &az); [cite: 23]
    M5.Imu.getGyro(&gx, &gy, &gz); [cite: 24]

    // Manual Quaternion Math [cite: 26, 31, 33]
    float gx_rad = gx * M_PI / 180.0f; [cite: 25]
    float gy_rad = gy * M_PI / 180.0f;
    float gz_rad = gz * M_PI / 180.0f;

    float dqw = 0.5f * (-qx * gx_rad - qy * gy_rad - qz * gz_rad); [cite: 26]
    float dqx = 0.5f * ( qw * gx_rad + qy * gz_rad - qz * gy_rad); [cite: 27]
    float dqy = 0.5f * ( qw * gy_rad - qx * gz_rad + qz * gx_rad); [cite: 28]
    float dqz = 0.5f * ( qw * gz_rad + qx * gy_rad - qy * gx_rad); [cite: 29]

    qw += dqw * dt; [cite: 30]
    qx += dqx * dt;
    qy += dqy * dt;
    qz += dqz * dt;

    float norm = sqrt(qw*qw + qx*qx + qy*qy + qz*qz); [cite: 31]
    qw /= norm; qx /= norm; qy /= norm; qz /= norm; [cite: 32]

    if (deviceConnected) {
        char buffer[64];
        snprintf(buffer, sizeof(buffer), "%.3f,%.3f,%.3f,%.3f", qw, qx, qy, qz);
        pCharacteristic->setValue(buffer);
        pCharacteristic->notify(); // Push data to phone
    }
    delay(40); // 25Hz is usually sufficient for mobile BLE updates
}