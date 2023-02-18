import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import os

name = 'label_data_0313.json'

def show(image):
    cv2.imshow('image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# noisy function is based of https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
def noisy(noise_typ,image):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.99
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 1.9 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)
        noisy = image + image * gauss
        return noisy

# obtain perspective transform matrix
# len = img.shape[0]
# wid = img.shape[1]
length = 720
width = 1280
new_len = length/3
new_wid = width/3

pt1 = np.array([[new_wid,new_len],[width - new_wid,new_len],[width,length],[0,length]])
pt1 = np.array([i.astype(np.float32) for i in pt1])

pt2 = np.array([[0,0],[width,0],[width,length],[0,length]])
pt2 = np.array([i.astype(np.float32) for i in pt2])



matrix = cv2.getPerspectiveTransform(pt1,pt2)

json_gt = [json.loads(line) for line in open(name)]
for n in range(len(json_gt)):
    gt = json_gt[n]
    gt_lanes = gt['lanes']
    y_samples = gt['h_samples']
    raw_file = gt['raw_file']
    if not os.path.exists(raw_file):
        continue
    img = cv2.imread(raw_file)

    gt_lanes_vis = [[(x, y) for (x, y) in zip(lane, y_samples)
                    if x >= 0] for lane in gt_lanes]

    img_vis = img.copy()
    overlay = img.copy()
    label = np.zeros(np.shape(img))
    alpha = 0.9
    for i in range(len(gt_lanes_vis)):
        cv2.polylines(overlay, np.int32([gt_lanes_vis[i]]), isClosed=False, color=(250, 249, 246), thickness = 7)
        cv2.polylines(label, np.int32([gt_lanes_vis[i]]), isClosed=False, color=(255,255,255), thickness = 7)

    label = cv2.cvtColor(label.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    by_x = {}  # position info for segments with starting x position as key
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])  # getting position info for each contour
        by_x[x] = i

    by_x_sort = dict(sorted(by_x.items()))  # sort the contours from left to right

    num =0
    for start_x in by_x_sort:
        cv2.drawContours(label, contours, by_x_sort[start_x], 1 * (num + 1), -1) # for numbered label
        # cv2.drawContours(label, contours, by_x_sort[start_x], 255, -1) # for pure white label
        num+=1
    label = cv2.warpPerspective(label, matrix, (width, length), flags=cv2.INTER_LINEAR)

    alpha = 1
    beta = 2
    img_vis = cv2.convertScaleAbs(img_vis, alpha=alpha, beta=beta)
    img_vis = cv2.addWeighted(overlay, alpha, img_vis, 1 - alpha, 0)

    alpha = 3.5
    beta = 0
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    adjusted = cv2.bitwise_not(adjusted)
    img_vis = cv2.bitwise_not(img_vis)

    alpha = 0.95
    res = cv2.addWeighted(adjusted, alpha, img_vis, 1 - alpha, 0)
    res = cv2.bitwise_not(res)

    hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0,0,254], dtype=np.uint8)
    upper_white = np.array([0, 0, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_white, upper_white)
    res = cv2.bitwise_and(res, res, mask= mask)

    res = cv2.bitwise_or(res, img)
    alpha = 0.16
    res = cv2.addWeighted(img, alpha, res, 1 - alpha, 0)

    res = noisy("poisson", res/255)
    res = noisy("gauss", res)
    res = cv2.warpPerspective(res, matrix, (width, length), flags=cv2.INTER_LINEAR)

    # add to branch cleanup_year23
    # CV-Pipeline/src/lane_detection

    import os

    try:
        os.makedirs("adjusted")
    except FileExistsError:
        pass

    try:
        os.makedirs("label")
    except FileExistsError:
        pass

    cv2.imwrite("adjusted/" +raw_file.split("/")[2]+".jpg", res * 255)
    cv2.imwrite("label/" + raw_file.split("/")[2]+".jpg", label)
