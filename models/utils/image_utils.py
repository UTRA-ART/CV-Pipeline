'''
Contains tools for resizing, warping, and handling image data
'''
import os
import glob 
import math

import cv2
import numpy as np

def resize(img, width, height=None, pad=False):
    '''
    Resizes single image

    # Arguments:
    img: numpy array of size H x W x 3
    width: width to reshape to 
    height: height to reshape to. If not provided, image ratio is kept. 
    pad: bool, indicates whether the image should be padded to keep the original ratio. Only relevant if height is set

    # Returns:
    resized image
    '''
    h, w, _ = img.shape 
    dest_width = width
    if height is None:
        dest_height = dest_width * h / w 
    else:
        dest_height = height

    if pad and height is not None:
        hratio = dest_height / h
        wratio = dest_width / w

        ratio = min(hratio, wratio)

        _dest_height = h * ratio
        _dest_width = w * ratio
        img = cv2.resize(img, (int(_dest_width), int(_dest_height)))
        height_pad = dest_height - img.shape[0]
        width_pad = dest_width - img.shape[1]

        img = np.pad(img, ((math.floor(height_pad / 2), math.ceil(height_pad / 2)), (math.floor(width_pad / 2), math.ceil(width_pad / 2)), (0, 0)))
        return img
    else:
        return cv2.resize(img, (int(dest_width), int(dest_height)))
    
def clean_web_images_classifier(path, size=250):
    '''
    Given a path to images scraped from the web, this script formats images for use for the classifier pipeline. 
    
    # Arguments:
    path: folder path containing recently scrapped images 
    size: size of smallest image dimension 
    '''

    images = glob.glob(os.path.join(path, '*.PNG'))

    for image in images:
        img = cv2.imread(image)
        h, w, c = img.shape 
        if h > w:
            cropped = resize(img, size)
        else:
            cropped = resize(img, int(size * w / h), height=size)
        cv2.imwrite(image, cropped)

if __name__=='__main__':
    clean_web_images_classifier('dev')