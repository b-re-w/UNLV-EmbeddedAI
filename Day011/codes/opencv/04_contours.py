import cv2
import numpy as np

img = cv2.imread('shapes.jpg')
if img is None: # Create dummy image with shapes
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255), -1) # Square
    cv2.circle(img, (300, 300), 50, (255, 255, 255), -1) # Circle

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 1. Thresholding (Critical Pre-step)
# Contours work best on binary white objects on black background
ret, thresh = cv2.threshold(gray, 127, 255, 0)

# 2. Find Contours
# RETR_TREE: Retrieves all contours and creates a full hierarchy list
# CHAIN_APPROX_SIMPLE: Compresses horizontal/vertical segments (saves memory)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

print(f"Number of contours found: {len(contours)}")

# 3. Draw Contours
# -1 means draw ALL contours. (0, 255, 0) is Green. 3 is thickness.
cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

cv2.imshow('Contours', img)
cv2.waitKey(0)
cv2.destroyAllWindows()