import numpy as np
import cv2

def crop(img):
    '''Crops image to be largest square possible'''
    iH, iW, _ = img.shape
    crop_size = min(iH, iW)
    ret = img[(iH - crop_size)//2:(iH - crop_size)//2 + crop_size,(iW - crop_size)//2:(iW - crop_size)//2+crop_size]
    return ret

def scale(img, width=250):
    iH, iW, _ = img.shape

    final_window_width = width

    dest_width = final_window_width
    dest_height = round((final_window_width / iW) * iH)

    res = cv2.resize(img, dsize=(dest_width, dest_height), interpolation=cv2.INTER_CUBIC)
    return res