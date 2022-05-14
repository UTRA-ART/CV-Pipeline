import os
import cv2

if __name__ == "__main__":
    dir_old = "/Users/jasonyuan/Desktop/UNet_Data/input"
    dir_current = "/Users/jasonyuan/Desktop/UNet_Data/Inputs_Cleaned"
    dir_new = "/Users/jasonyuan/Desktop/UNet_Data/Uncleaned_Inputs"

    for filename in os.listdir(dir_old):
        if filename == ".DS_Store":
            continue
        elif filename in os.listdir(dir_current):
            continue
        else:
            img = cv2.imread(os.path.join(dir_old,filename))
            img = cv2.resize(img,(1280,720),interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(dir_new,filename),img)
