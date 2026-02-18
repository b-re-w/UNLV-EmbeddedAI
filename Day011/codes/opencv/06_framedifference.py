import cv2

cap = cv2.VideoCapture(0)

# Initialize the first frame
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

while True:
    ret, frame = cap.read()
    if not ret: break

    # Convert to grayscale and blur to remove noise
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # 1. Compute absolute difference between current and previous frame
    diff = cv2.absdiff(prev_gray, gray)

    # 2. Threshold the difference to get binary motion mask
    # Pixels with a difference > 25 are turned white (255)
    thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]

    # 3. Dilate the image to fill in holes (morphological operation)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # 4. Find contours (draw boxes around motion)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 500:  # Ignore small movements
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Motion Feed", frame)
    cv2.imshow("Threshold (Motion Mask)", thresh)

    # Update previous frame
    prev_gray = gray.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()