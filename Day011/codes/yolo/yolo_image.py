from ultralytics import YOLO
import cv2

# 1. Load the Model
# 'yolov8n.pt' will automatically download the first time you run this.
# The 'n' stands for Nano (fastest).
print("Loading Model...")
model = YOLO('yolov8n.pt')

# 2. Load an Image
# You can use a local path or a URL
image_source = 'https://ultralytics.com/images/zidane.jpg'

# 3. Run Inference
# results is a list (in case you passed multiple images)
results = model(image_source)

# 4. Show Results
# Visualize the results on the image
for result in results:
    # plot() draws the bounding boxes and labels on the image
    annotated_frame = result.plot()

    # Display using OpenCV
    cv2.imshow("YOLO Detection", annotated_frame)
    cv2.waitKey(0)  # Wait for any key press to close

cv2.destroyAllWindows()
