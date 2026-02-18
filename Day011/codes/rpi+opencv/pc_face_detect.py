import cv2

# --- CONFIGURATION ---
PI_IP = "192.168.1.15"  # <--- REPLACE WITH YOUR PI'S IP ADDRESS
PORT = 8554

# 1. Connect to the stream
# We use the TCP URL format that OpenCV understands
stream_url = f"tcp://{PI_IP}:{PORT}"
print(f"Connecting to {stream_url}...")

cap = cv2.VideoCapture(stream_url)

# 2. Load Face Detector (Haar Cascade)
# This is lighter and faster than deep learning models for simple tests
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if not cap.isOpened():
    print("Error: Could not connect to the Pi.")
    print("Make sure the Pi script is running and waiting for a connection.")
    exit()

print("Stream received! Press 'q' to quit.")

while True:
    # 3. Read Frame
    ret, frame = cap.read()
    if not ret:
        print("Frame lost. Stopping.")
        break

    # 4. Face Detection Logic
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detectMultiScale parameters:
    # scaleFactor=1.1: Image size reduced by 10% each pass
    # minNeighbors=5: Higher = fewer detections but higher quality
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    # Draw boxes
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 5. Display
    cv2.imshow("Remote Face Detection", frame)

    # Handle Latency (Optional)
    # If the buffer gets too full, the video lags.
    # This 'grab' loop skips frames to keep the feed "live".
    # for _ in range(2): cap.grab()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()