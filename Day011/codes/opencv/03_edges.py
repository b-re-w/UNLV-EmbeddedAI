import cv2
import numpy as np

img = cv2.imread('image.jpg', 0) # Load directly as gray
if img is None: img = np.zeros((300, 300), dtype=np.uint8); cv2.rectangle(img, (50, 50), (250, 250), 255, -1)

# 1. Canny Edge Detection
# 100 = Lower Threshold (Discard edges below this gradient)
# 200 = Upper Threshold (Accept edges above this gradient)
# Values between 100-200 are accepted ONLY if connected to a strong edge.
edges = cv2.Canny(img, 100, 200)

cv2.imshow('Original Gray', img)
cv2.imshow('Canny Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()