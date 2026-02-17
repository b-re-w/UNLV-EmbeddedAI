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
    BLEDevice::init("M5Core2_Quat_Mobile10");
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

    qw += dqw * dt; 
    qx += dqx * dt;
    qy += dqy * dt;
    qz += dqz * dt;

    float norm = sqrt(qw*qw + qx*qx + qy*qy + qz*qz); 
    qw /= norm; qx /= norm; qy /= norm; qz /= norm; 

    if (deviceConnected) {
        char buffer[64];
        snprintf(buffer, sizeof(buffer), "%.3f,%.3f,%.3f,%.3f", qw, qx, qy, qz);
        pCharacteristic->setValue(buffer);
        pCharacteristic->notify(); // Push data to phone
    }
    delay(40); // 25Hz is usually sufficient for mobile BLE updates
}