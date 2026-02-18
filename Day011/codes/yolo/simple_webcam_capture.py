import cv2

# Initialize the camera capture object (0 for default webcam)
cap = cv2.VideoCapture(1)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open video device.")
else:
    # Read a single frame from the camera
    ret, frame = cap.read()

    if ret:
        # Save the captured frame as an image file
        image_path = 'webcam_capture.png'
        cv2.imwrite(image_path, frame)
        print(f"Image saved as {image_path}")

        # Optional: Display the captured image
        cv2.imshow("Captured Image", frame)
        cv2.waitKey(2000)  # Wait for 2 seconds
    else:
        print("Error: Could not read a frame.")

# Release the camera resource and close all windows
cap.release()
cv2.destroyAllWindows()
