#include <M5Unified.h>
#include <Wire.h>
#include <WiFi.h>
#include <ArduinoJson.h>
#include <Adafruit_APDS9960.h>

// --- CONFIGURATION ---
const char* ssid     = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";
const char* host     = "192.168.1.100"; // IP address of your PC/Mac
const int   port     = 65432;           // Same port as Python script

Adafruit_APDS9960 apds;
WiFiClient client;

void setup() {
  auto cfg = M5.config();
  M5.begin(cfg);

  M5.Display.setTextSize(2);
  M5.Display.println("Initializing...");

  // Initialize I2C for Port A (Core2 uses 32, 33)
  Wire.begin(32, 33);

  if(!apds.begin()){
    M5.Display.println("Failed to find APDS9960 sensor!");
    while(1) delay(100);
  }
  apds.enableColor(true); // Enable color sensing

  // Connect to WiFi
  M5.Display.print("Connecting to WiFi");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    M5.Display.print(".");
  }
  M5.Display.println("\nWiFi Connected!");
  M5.Display.println(WiFi.localIP());
}

void loop() {
  M5.update();

  // 1. Read Sensor Data
  uint16_t r, g, b, c;

  // Wait for color data to be ready
  while(!apds.colorDataReady()) {
    delay(5);
  }
  apds.getColorData(&r, &g, &b, &c);

  // 2. Create JSON Document
  StaticJsonDocument<200> doc;
  doc["location"] = "Lab_Station_1";
  doc["timestamp"] = millis(); // Relative time (Python will handle real timestamp)
  doc["red"] = r;
  doc["green"] = g;
  doc["blue"] = b;
  doc["clear"] = c;

  String jsonString;
  serializeJson(doc, jsonString);

  // 3. Send via TCP
  if (client.connect(host, port)) {
    client.println(jsonString);
    client.stop(); // Close connection after sending (Short-lived connection)
    M5.Display.fillScreen(BLACK);
    M5.Display.setCursor(0,0);
    M5.Display.printf("Sent:\nR:%d G:%d B:%d", r, g, b);
  } else {
    M5.Display.fillScreen(RED);
    M5.Display.setCursor(0,0);
    M5.Display.println("Connection Failed");
  }

  delay(1000); // 1 Hz sample rate
}