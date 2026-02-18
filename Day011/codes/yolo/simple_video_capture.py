import cv2

# Initialize the camera object (0 is usually the default built-in camera)
cam = cv2.VideoCapture(1)

# Check if the camera opened successfully
if not cam.isOpened():
    print("Error: Could not access the webcam.")
else:
    cv2.namedWindow("Camera Feed - Press SPACE to save, ESC to exit")
    img_counter = 0

    while True:
        # Capture frame-by-frame
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break

        # Display the frame
        cv2.imshow("Camera Feed - Press SPACE to save, ESC to exit", frame)

        # Wait for a key press
        k = cv2.waitKey(1)

        # Check for ESC key press (ASCII 27) to exit
        if k % 256 == 27:
            print("Escape hit, closing...")
            break
        # Check for SPACE bar key press (ASCII 32) to save image
        elif k % 256 == 32:
            img_name = f"opencv_frame_{img_counter}.png"
            cv2.imwrite(img_name, frame)
            print(f"{img_name} written!")
            img_counter += 1

    # Release the camera and close all windows
    cam.release()
    cv2.destroyAllWindows()
