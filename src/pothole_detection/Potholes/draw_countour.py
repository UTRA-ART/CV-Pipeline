from email.mime import base
import cv2
import numpy as np
import os
from pathlib import Path
import glob

T_LABEL_PATH = "/Users/leon.lee.21/Desktop/CV-Pipeline/src/pothole_detection/Potholes/processed/data/t_labels"
BLACK_FOLDER = "/Users/leon.lee.21/Desktop/CV-Pipeline/src/pothole_detection/Potholes/processed/data/t_black"
IMAGE_FOLDER = "/Users/leon.lee.21/Desktop/CV-Pipeline/src/pothole_detection/Potholes/processed/data/t_images"

for filename in os.listdir(BLACK_FOLDER):
    # Check if the file is an image file
    if not filename.endswith(".jpg"):
        continue
    
    # Load the image
    img = cv2.imread(os.path.join(BLACK_FOLDER, filename), 0)  # Load the image in grayscale

    # Apply thresholding to extract the circle
    thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]

    # Find contours of the circle
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over all the contours and draw a rectangle around each of them
    for contour in contours:
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)

    img_files = glob.glob(IMAGE_FOLDER + "*.jpg") #list of filenames in BLACK_FOLDER

    base_name, extension = os.path.splitext(filename) # base_name of current black_image 
    #print(filename)
    print(base_name)
    files = glob.glob(IMAGE_FOLDER + "/*" + base_name[-7:] + ".jpg")
    print(files[0])
    #print(files)


    # DRAW ON TRANSFORMED IMAGE
    image = cv2.imread(files[0])

    # Loop over all the contours and draw a rectangle around each of them
    filename = base_name[2:] + ".txt"
    print(filename)
    file = open(os.path.join(T_LABEL_PATH, filename), "w")
    for contour in contours:
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        file.write(
            f"0 {x/1280} {y/720} {w/1280} {h/720}\n"
        )
        # Draw a rectangle around the contour
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    file.close()

    # Show the image with the rectangles
    # cv2.imshow("Image with Contours", image)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
