
import cv2
import time

# --- Setup ---
cap = cv2.VideoCapture(1)  # 0 is usually the default webcam
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Video Writer Setup (Initially None)
writer = None
recording = False

print("------------------------------------------------------")
print(" CONTROLS:")
print(" 'i' = Capture Image")
print(" 'v' = Start/Stop Video Recording")
print(" 'q' = Quit")
print("------------------------------------------------------")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()

        # Visual indicator if recording
        if recording:
            cv2.circle(display_frame, (30, 30), 10, (0, 0, 255), -1)  # Red Dot
            writer.write(frame)

        cv2.imshow("Webcam Capture Tool", display_frame)
        key = cv2.waitKey(1) & 0xFF

        # [I] Save Image
        if key == ord('i'):
            filename = f"capture_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")

        # [V] Toggle Video Recording
        elif key == ord('v'):
            if not recording:
                # Start Recording
                filename = f"video_{int(time.time())}.mp4"
                writer = cv2.VideoWriter(
                    filename,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    30,
                    (width, height)
                )
                recording = True
                print(f"Started Recording: {filename}")
            else:
                # Stop Recording
                writer.release()
                recording = False
                print("Stopped Recording")

        # [Q] Quit
        elif key == ord('q'):
            break

finally:
    if writer:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()
