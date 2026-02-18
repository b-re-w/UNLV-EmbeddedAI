import cv2

cap = cv2.VideoCapture(0)

# Create the Background Subtractor Object
# history: How many previous frames influence the background model
# varThreshold: Threshold for detecting shadows/motion (lower = more sensitive)
backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. Apply the algorithm to a frame
    # This automatically updates the background model and returns a mask
    fgMask = backSub.apply(frame)

    # 2. Remove shadows (which are gray, value 127) to keep only pure motion (white, 255)
    # Binary thresholding removes the gray shadows
    _, fgMask = cv2.threshold(fgMask, 250, 255, cv2.THRESH_BINARY)

    # Optional: Clean up noise
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, (5, 5))

    cv2.imshow('Original Frame', frame)
    cv2.imshow('FG Mask (Motion)', fgMask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()