import cv2
import numpy as np

def generate_aruco_markers(id_sequence):
    # 1. Choose a dictionary (Standard is 6x6 bits, 250 unique IDs)
    # Other options: DICT_4X4_50, DICT_5X5_100, etc.
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

    print(f"Generating markers for IDs: {id_sequence}")

    for marker_id in id_sequence:
        # 2. Generate the marker image
        # parameters: (dictionary, marker_id, size_in_pixels)
        # 200x200 pixels is a good standard size
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 200)

        # 3. Save the image to disk
        filename = f"aruco_marker_{marker_id}.png"
        cv2.imwrite(filename, marker_img)
        print(f"Saved {filename}")

# --- Example Usage ---
# Provide your sequence of numbers here
my_numbers = [0, 5, 12, 42, 100]

generate_aruco_markers(my_numbers)