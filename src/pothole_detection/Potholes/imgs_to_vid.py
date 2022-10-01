import cv2
import numpy as np
import os

dir = 'C:\\Users\\ammar\\Downloads\\zed_images_potholes\\zed_images_potholes\\2'

imgs = []
width, height = None, None
for file in os.listdir(dir):
    img = cv2.imread(os.path.join(dir, file))
    imgs.append(img)
    height, width, layers = img.shape


print(f'{width}, {height}')

out = cv2.VideoWriter('potholes.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
                      5, (width, height))

for img in imgs:
    out.write(img)

out.release()
