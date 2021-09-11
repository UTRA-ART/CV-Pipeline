import cv2
import numpy as np
import matplotlib.pyplot as plt


INPUT = '../test_images/straight_lines2.jpg'
INPUT = '../test_images/test1.jpg'


def edges_method():
    img = cv2.imread(INPUT)

    img = cv2.blur(img, (5,5))
    edges = cv2.Canny(image=img, threshold1=100, threshold2=250) # Canny Edge Detection
    cv2.imshow('Edges', edges)
    return edges


def anannays_method():
    img = cv2.imread(INPUT)
    img = cv2.blur(img, (5,5))
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Flip colours
    img_flip = 255 - img_grey

    # Threshold
    _, img_thresh = cv2.threshold(img_flip, 50, 255, cv2.THRESH_BINARY_INV)

    cv2.imshow('Threshold', img_thresh)
    return img_thresh


if __name__ == '__main__':
    img_thresh = anannays_method()
    img_edges = edges_method()

    diff = img_thresh - img_edges
    cv2.imshow('Diff', diff)
