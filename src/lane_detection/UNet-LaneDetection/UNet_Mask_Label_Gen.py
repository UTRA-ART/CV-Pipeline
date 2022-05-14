import time
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
import os

def define_region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]

    # src = np.array([[(width//9+math.floor(2*(height//5))/(math.tan(1/4*math.pi)),3*(height//5)),
    #                 (8*(width//9)-math.floor(2*(height//5))/(math.tan(1/4*math.pi)),3*(height//5)),
    #                 (8*(width//9),height),
    #                 (width//9,height)]],
    #                 dtype='float32')  # Extract a trapezoid like region (Top Left, Top Right, Bottom Right, Bottom Left)

    src = np.array([[(width//10+math.floor(2*(height//5))/(math.tan(1/4*math.pi)),3*(height//5)),
                    (9*(width//10)-math.floor(2*(height//5))/(math.tan(1/4*math.pi)),3*(height//5)),
                    (width,height),
                    (0,height)]],
                    dtype='float32')


    # temp = cv2.polylines(image,np.int32(src),True,(255,0,0))

    # plt.figure("Annotated")
    # plt.imshow(temp)
    # plt.show()

    dst = np.array([[(0,0),
                    (image.shape[1],0),
                    (image.shape[1],image.shape[0]),
                    (0,image.shape[0])]],
                    dtype='float32')   # To be transformed to a rectangle
    # dst = np.array([[(0,0),
    #                 (330,0),
    #                 (330,180),
    #                 (0,180)]],
    #                 dtype='float32')   # To be transformed to a rectangle

    M = cv2.getPerspectiveTransform(src,dst)
    M_inv = cv2.getPerspectiveTransform(dst,src)
    warped_image = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

    return warped_image,M,M_inv

if __name__ == "__main__":
    # img_path = "/Users/jasonyuan/Desktop/UTRA/ART stuff/lane_dataset grass/image_rect_color_screenshot_09.12.2017 2.png"
    # dir = "/Users/jasonyuan/Desktop/UNet_Data/Inputs_Cleaned"
    # save_dir_input = "/Users/jasonyuan/Desktop/UNet_Data/Dataset2/inputs"
    # save_dir_label = "/Users/jasonyuan/Desktop/UNet_Data/Dataset2/labels"

    dir = "/Users/jasonyuan/Desktop/inputs"
    dir2 = "/Users/jasonyuan/Desktop/labels"
    save_dir_input = "/Users/jasonyuan/Desktop/Synthetic Inputs"
    save_dir_label = "/Users/jasonyuan/Desktop/Renamed"
    # May need to add the whole directory looping stuff later
    n_count = 1

    # image = cv2.imread("/Users/jasonyuan/Desktop/Group_2_Batch_6..9/Batch_6/Synthetic_Lane_286.png")
    # # cv2.imshow("ROI",roi_im)
    # # cv2.imshow("OG",image)
    # # Convert to a mask of binary 0 and 1 --------------------------------------
    # red_channel = image[:,:,2]
    # mask = np.zeros(red_channel.shape)
    # # mask[np.asarray((image[:,:,0]!=255)&(image[:,:,1]!=255)&(image[:,:,2]==255)).nonzero()] = 1
    # mask[np.where(np.all((image==[0,38,255]),axis=2))] = 1
    #
    # cv2.imshow("Red Channel",red_channel)
    # cv2.imshow("BW",mask)
    # cv2.waitKey(0)

    for filename in os.listdir(dir):
        if filename == ".DS_Store":
            continue
        name = filename.split(".")[0]
        num = name.split("_")[-1]
        img = cv2.imread(dir2+"/"+"Lane_Label_"+num+".png",cv2.IMREAD_GRAYSCALE)

        cv2.imwrite(os.path.join(save_dir_label,name+"_Label.png"),img)

    # if not os.path.isdir(save_dir_input):
    #     os.makedirs(save_dir_input)
    # if not os.path.isdir(save_dir_label):
    #     os.makedirs(save_dir_label)
    #
    # for filename in os.listdir(dir):
    #     if filename == ".DS_Store":
    #         continue
    #     else:
    #         name = filename.split(".")[0]
    #         image = cv2.imread(os.path.join(dir,filename))
    #         image = cv2.resize(image,(1280,720),interpolation=cv2.INTER_AREA)
    #         red_channel = image[:,:,2]
    #         mask = np.zeros(red_channel.shape)
    #         # mask[np.asarray((image[:,:,0]!=255)&(image[:,:,1]!=255)&(image[:,:,2]==255)).nonzero()] = 255
    #         mask[red_channel>250] = 255
    #
    #         # cv2.imwrite(os.path.join(save_dir_input,num+".png"),input_img)
    #         cv2.imwrite(os.path.join(save_dir_label,name+"_Label.png"),mask)
    #
    #         # roi_im = np.copy(image)
    #         # roi_im,M,M_inv = define_region_of_interest(np.copy(image))
    #         # roi_im_blurred = cv2.GaussianBlur(roi_im,(3,3),0)
    #
    #         # Convert to a mask of binary 0 and 1 --------------------------------------
    #         # thresh,im_bw = cv2.threshold(cv2.cvtColor(roi_im,cv2.COLOR_BGR2GRAY),128,255,cv2.THRESH_BINARY)
    #         # im_bw = cv2.medianBlur(im_bw,ksize=5)
    #         # hls = cv2.cvtColor(roi_im_blurred,cv2.COLOR_BGR2HLS)
    #         # s_channel = hls[:,:,1]
    #         # s_binary = np.zeros(s_channel.shape)
    #         # s_binary[(s_channel >= 45) & (s_channel <= 70)] = 1
    #         # mask_bw = np.zeros(im_bw.shape)
    #         # mask_bw[im_bw == 255] = 1
    #         # combined_hls_thresh = cv2.bitwise_or(s_binary,mask_bw)
    #         # cv2.imwrite(os.path.join(save_dir_input,input_name),roi_im)
    #         # cv2.imwrite(os.path.join(save_dir_label,label_name),combined_hls_thresh*255)
    #
    #         n_count += 1
    #
    #         # print(im_bw.shape)
    #
    #         # img_Salt = cv2.imread("/Users/jasonyuan/Desktop/train/masks/0a1742c740.png",cv2.IMREAD_GRAYSCALE)
    #         # print(img_Salt.shape)
    #         # print(np.sum(img_Salt == 255))
    #         # print(np.sum(img_Salt > 1))

    # --------------------------------------------------------------------------
    # # This part is the overlaying of the UNet output onto the original image
    # image_new = cv2.imread("/Users/jasonyuan/Desktop/test.png",cv2.IMREAD_GRAYSCALE)
    # # im_new_g = cv2.cvtColor(image_new,cv2.COLOR_BGR2GRAY)
    # thresh,im_new_thresh = cv2.threshold(image_new,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # mask_new_bw = np.zeros(im_new_thresh.shape)
    # mask_new_bw[im_new_thresh == 255] = 1

    # print(mask_new_bw)
    # cv2.imshow("Annotated",mask_new_bw)
    # print(len((mask_bw*255 == im_bw)[(mask_bw*255 == im_bw) == True]))
    # print(mask_bw.shape)
    # print(im_bw.shape)
    # print(np.sum(mask_bw),np.sum(im_bw))
    # cv2.imshow("Binary Black and White 2",mask_bw)
    # cv2.imshow("Binary Black and White",im_bw)

    # re_trans_im = cv2.warpPerspective(mask_new_bw, M_inv, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    # # print(re_trans_im.shape)
    # im_with_mask = np.copy(image)
    # im_with_mask[np.where(re_trans_im != 1)] = [0,0,255]

    # cv2.imshow("Unwarped",image)
    # cv2.imshow("ROI",roi_im)
    # cv2.imshow("Black and White Mask",mask_bw)
    # # cv2.imshow("Re-Transform",re_trans_im)
    # # cv2.imshow("ROI is Red",im_with_mask)
    # cv2.waitKey(0)
