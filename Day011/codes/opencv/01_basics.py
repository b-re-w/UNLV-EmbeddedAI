import cv2
import numpy as np
from matplotlib import pyplot as plt

# 1. Load Image
# '0' loads it directly as grayscale, but let's load color to show conversion
img = cv2.imread('image.jpg')
if img is None:
    # Generate a dummy image if no file exists
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(img, (150, 150), 100, (255, 255, 255), -1)

# 2. Color Space Conversion (BGR -> Gray)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. Simple Thresholding
# Logic: If pixel > 127, make it 255 (white), else 0 (black)
ret, thresh_binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 4. Adaptive Thresholding (Advanced)
# Logic: Calculates threshold for small regions. Better for varying lighting.
thresh_adaptive = cv2.adaptiveThreshold(gray, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)

# Visualization
titles = ['Original Image', 'Gray', 'Simple Threshold', 'Adaptive Threshold']
images = [img, gray, thresh_binary, thresh_adaptive]

plt.figure(figsize=(10, 8))
for i in range(4):
    plt.subplot(2, 2, i+1)
    if i == 0:
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()