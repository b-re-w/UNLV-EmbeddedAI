#include <M5Unified.h>
#include "model_data.h"

#include <TensorFlowLite_ESP32.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

// --- CONFIGURATION ---
#define WINDOW_SIZE 50
#define NUM_FEATURES 4  // qw, qx, qy, qz
#define NUM_CLASSES 7
const char* CLASSES[] = {"static", "verti", "horiz", "circ", "type", "draw", "write"};

#define CONFIDENCE_THRESHOLD 0.70

// TFLite Globals
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
tflite::AllOpsResolver resolver;
const int kTensorArenaSize = 30 * 1024; // Adjust if needed
uint8_t tensor_arena[kTensorArenaSize];

// Data Buffer (Circular Buffer)
float input_buffer[WINDOW_SIZE][NUM_FEATURES];
int buffer_head = 0;

void setup() {
    auto cfg = M5.config();
    M5.begin(cfg);

    M5.Display.setTextSize(2);
    M5.Display.println("Init AI Model...");

    // 1. Init IMU (M5Unified automatically selects MPU6886 for Core2)
    M5.Imu.begin();

    // 2. Load TFLite Model
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        M5.Display.println("Schema Error!");
        while(1);
    }

    // 3. Setup Interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        M5.Display.println("Alloc Failed!");
        while(1);
    }

    M5.Display.println("Ready!");
    delay(1000);
    M5.Display.clear();
}

void loop() {
    M5.update();

    // 1. Get Quaternion Data
    // M5Unified provides helper to get Quaternions directly
    float q[4]; // w, x, y, z
    M5.Imu.getQuaternion(&q[0], &q[1], &q[2], &q[3]);

    // 2. Add to Circular Buffer
    // Note: Verify your training order! Python script used: qw, qx, qy, qz
    input_buffer[buffer_head][0] = q[0]; // qw
    input_buffer[buffer_head][1] = q[1]; // qx
    input_buffer[buffer_head][2] = q[2]; // qy
    input_buffer[buffer_head][3] = q[3]; // qz

    buffer_head++;
    if (buffer_head >= WINDOW_SIZE) buffer_head = 0;

    // 3. Run Inference (Every loop or set interval)
    TfLiteTensor* input = interpreter->input(0);

    // Get Quantization Params
    float input_scale = input->params.scale;
    int input_zero_point = input->params.zero_point;

    // Flatten Circular Buffer into Linear Input Tensor
    for (int i = 0; i < WINDOW_SIZE; i++) {
        int idx = (buffer_head + i) % WINDOW_SIZE; // Unwind buffer from oldest to newest
        for (int f = 0; f < NUM_FEATURES; f++) {
            float val = input_buffer[idx][f];

            // Quantize: (val / scale) + zero_point
            int8_t q_val = (int8_t)(val / input_scale + input_zero_point);

            // Map to flat array
            input->data.int8[(i * NUM_FEATURES) + f] = q_val;
        }
    }

    // 4. Invoke
    if (interpreter->Invoke() == kTfLiteOk) {
        TfLiteTensor* output = interpreter->output(0);

        // Dequantize Output
        float out_scale = output->params.scale;
        int out_zero_point = output->params.zero_point;

        float max_score = 0;
        int max_index = -1;

        for (int i = 0; i < NUM_CLASSES; i++) {
            float score = (output->data.int8[i] - out_zero_point) * out_scale;
            if (score > max_score) {
                max_score = score;
                max_index = i;
            }
        }

        // 5. Display Result
        // Serial Output
        Serial.printf("Pred: %s (%.2f)\n", CLASSES[max_index], max_score);

        // LCD Output
        M5.Display.setCursor(0, 0);
        if (max_score > CONFIDENCE_THRESHOLD) {
            M5.Display.setTextColor(GREEN, BLACK);
            M5.Display.setTextSize(3);
            M5.Display.printf("%s      \n", CLASSES[max_index]); // Spaces to clear prev text

            M5.Display.setTextSize(2);
            M5.Display.setTextColor(WHITE, BLACK);
            M5.Display.printf("Conf: %.0f%%   ", max_score * 100);
        } else {
            M5.Display.setTextColor(YELLOW, BLACK);
            M5.Display.setTextSize(3);
            M5.Display.printf("...       \n");
            M5.Display.setTextSize(2);
            M5.Display.printf("Conf: %.0f%%   ", max_score * 100);
        }
    }

    delay(20); // Approx 50Hz sample rate (Matches training)
}