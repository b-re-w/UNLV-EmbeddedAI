import cv2
import numpy as np

img = cv2.imread('image.jpg')
if img is None: img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8) # Noisy dummy

# 1. Gaussian Blur (The Standard)
# (5, 5) is the kernel size (must be odd). 0 is standard deviation (auto-calc).
gaussian = cv2.GaussianBlur(img, (5, 5), 0)

# 2. Median Blur (The De-speckler)
# Great for removing salt-and-pepper noise
median = cv2.medianBlur(img, 5)

# 3. Bilateral Filter (The Edge Preserver)
# Blurs surfaces but keeps edges sharp. Slow but pretty.
# 9 = Diameter, 75 = SigmaColor, 75 = SigmaSpace
bilateral = cv2.bilateralFilter(img, 9, 75, 75)

# Display
cv2.imshow('Original', img)
cv2.imshow('Gaussian (Soft)', gaussian)
cv2.imshow('Bilateral (Clean Edges)', bilateral)
cv2.waitKey(0)
cv2.destroyAllWindows()