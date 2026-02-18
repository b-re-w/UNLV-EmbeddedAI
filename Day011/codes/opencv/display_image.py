import cv2 as cv
from matplotlib import pyplot as plt


def show_image(image, title="Image", is_bgr=True):
    """
    Helper function to display an image using Matplotlib.
    Handles BGR to RGB conversion automatically.
    """
    # Convert BGR to RGB if necessary
    if is_bgr:
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    plt.imshow(image)
    plt.title(title)
    # Hide tick values on X and Y axis
    plt.xticks([]), plt.yticks([])
    plt.show()


# Main Execution
if __name__ == "__main__":
    # Load image
    img = cv.imread("shapes.png", cv.IMREAD_COLOR)

    if img is not None:
        # Display the image correctly
        show_image(img, title="Corrected RGB Output")

        # Convert to Grayscale for processing
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Display grayscale (cmap='gray' is required for matplotlib 2D arrays)
        plt.imshow(img_gray, cmap='gray')
        plt.title("Grayscale Version")
        plt.xticks([]), plt.yticks([])
        plt.show()
    else:
        print("Image not found.")