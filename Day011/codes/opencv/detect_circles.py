import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def detect_and_draw_circles(image_path):
    # 1. Load image
    img = cv.imread(image_path)
    if img is None:
        print("Image file not found.")
        return

    # 2. Convert to Grayscale
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 3. Edge Detection (Canny)
    # Thresholds 50 and 150 determine weak vs strong edges
    edges = cv.Canny(img_gray, 50, 150)

    # 4. Detect Circles (Hough Transform)
    # dp=1: Resolution of accumulator inverse ratio
    # minDist=20: Minimum distance between detected centers
    circles = cv.HoughCircles(
        edges,
        cv.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=50
    )

    # 5. Draw Detected Circles
    # Work on a copy to preserve the original image for display
    output_img = img.copy()

    if circles is not None:
        # Convert coordinates to integers
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]

            # Draw the outer circle (Green, thickness 5)
            cv.circle(output_img, center, radius, (0, 255, 0), 5)

            # Add a label near the circle
            cv.putText(output_img, "Circle", (i[0] - 30, i[1]),
                       cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

    # 6. Display Results
    # Convert BGR to RGB for Matplotlib
    output_img_rgb = cv.cvtColor(output_img, cv.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 5))
    plt.imshow(output_img_rgb)
    plt.title(f"Detected {len(circles[0]) if circles is not None else 0} Circles")
    plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == "__main__":
    # You can use 'smarties.png' or any image with circular objects
    detect_and_draw_circles("smarties.png")