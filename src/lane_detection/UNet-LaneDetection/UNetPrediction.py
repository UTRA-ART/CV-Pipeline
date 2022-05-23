from UNetModel import UNet
from UNet_Mask_Label_Gen import define_region_of_interest
import torch
import cv2
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import time
import onnx
import onnxruntime as ort

video_path = "/Users/jasonyuan/Desktop/output_video.mp4"
# video_path = "/Users/jasonyuan/Desktop/Section_of_dashcam.mp4"
save_path = "/Users/jasonyuan/Desktop/Processed Frames"
image_path = "/Users/jasonyuan/Desktop/Seattle Lane Driving Data"


def find_edge_channel2(img):

    gray_im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_im = cv2.GaussianBlur(gray_im, (5, 5), 0)

    med = np.median(gray_im)
    l = int(max(0, (1 - 0.205) * med))
    u = int(min(255, (1 + 0.205) * med))
    edges_mask = cv2.Canny(gray_im, l, u)

    edges_mask_inv = cv2.bitwise_not(edges_mask)

    return edges_mask, edges_mask_inv


def find_edge_channel(img):
    edges_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    width = img.shape[1]
    height = img.shape[0]

    gray_im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray_im = cv2.GaussianBlur(gray_im,(3,3),0)
    # Separate into quadrants
    med1 = np.median(gray_im[: height // 2, : width // 2])
    med2 = np.median(gray_im[: height // 2, width // 2 :])
    med3 = np.median(gray_im[height // 2 :, width // 2 :])
    med4 = np.median(gray_im[height // 2 :, : width // 2])

    l1 = int(max(0, (1 - 0.205) * med1))
    u1 = int(min(255, (1 + 0.205) * med1))
    e1 = cv2.Canny(gray_im[: height // 2, : width // 2], l1, u1)

    l2 = int(max(0, (1 - 0.205) * med2))
    u2 = int(min(255, (1 + 0.205) * med2))
    e2 = cv2.Canny(gray_im[: height // 2, width // 2 :], l2, u2)

    l3 = int(max(0, (1 - 0.205) * med3))
    u3 = int(min(255, (1 + 0.205) * med3))
    e3 = cv2.Canny(gray_im[height // 2 :, width // 2 :], l3, u3)

    l4 = int(max(0, (1 - 0.205) * med4))
    u4 = int(min(255, (1 + 0.205) * med4))
    e4 = cv2.Canny(gray_im[height // 2 :, : width // 2], l4, u4)

    # Stitch the edges together
    edges_mask[: height // 2, : width // 2] = e1
    edges_mask[: height // 2, width // 2 :] = e2
    edges_mask[height // 2 :, width // 2 :] = e3
    edges_mask[height // 2 :, : width // 2] = e4

    edges_mask_inv = cv2.bitwise_not(edges_mask)

    return edges_mask, edges_mask_inv


def predict_lanes(frame, unet):
    frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
    # roi2,M2,M_inv2 = define_region_of_interest(frame)
    frame_copy = np.copy(frame)
    # roi,M,M_inv = define_region_of_interest(frame_copy)
    # print(roi.shape)
    # roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    # roi = roi.reshape(1,1,180,330)
    # roi = roi/255.
    # input = torch.Tensor(roi)

    test_edges, test_edges_inv = find_edge_channel(frame_copy)
    frame_copy = np.append(
        frame_copy,
        test_edges.reshape(test_edges.shape[0], test_edges.shape[1], 1),
        axis=2,
    )
    frame_copy = np.append(
        frame_copy,
        test_edges_inv.reshape(test_edges_inv.shape[0], test_edges_inv.shape[1], 1),
        axis=2,
    )
    frame_copy = cv2.resize(frame_copy, (330, 180))

    input = torch.Tensor((frame_copy / 255.0).transpose(2, 0, 1)).reshape(
        1, 5, 180, 330
    )
    # x = (frame_copy/255.).transpose(2,0,1).reshape(1,5,180,330).astype(np.float32)

    # input = torch.Tensor((cv2.cvtColor(input_img_copy,cv2.COLOR_BGR2GRAY)/255.).reshape(1,1,180,330))
    # ort_sess = ort.InferenceSession('/Users/jasonyuan/Desktop/unet_with_sigmoid.onnx')
    # output = ort_sess.run(None, {'Inputs': x})[0]

    unet.eval()
    output = unet(input)
    # print(output)
    # print(unet)

    output = torch.sigmoid(output)
    output = output.detach().numpy()
    pred_mask = np.where(output > 0.5, 1, 0)

    # print(output)
    # print(ground_truth.shape)
    # print(pred_mask.size())
    pred_mask = (pred_mask.squeeze(0)).transpose(1, 2, 0).squeeze().astype("float32")
    # pred_mask = cv2.resize(pred_mask,(1280,720),interpolation=cv2.INTER_AREA)

    overlayed_mask = np.copy(cv2.resize(frame, (330, 180)))
    # overlayed_mask = np.copy(input_img)
    overlayed_mask[np.where(pred_mask == 1)[0], np.where(pred_mask == 1)[1], 2] = 255
    overlayed_mask[np.where(pred_mask == 1)[0], np.where(pred_mask == 1)[1], 1] = 0
    overlayed_mask[np.where(pred_mask == 1)[0], np.where(pred_mask == 1)[1], 0] = 0

    # print(pred_mask.sum())
    # cv2.imshow("Input", frame)
    # cv2.imshow("Overlayed", overlayed_mask)
    # cv2.waitKey(0)

    return overlayed_mask, pred_mask


def sortkey(x):
    if x == ".DS_Store":
        return -1
    else:
        return int(x.split(".")[0].split("Lane")[1])


if __name__ == "__main__":
    # cap = cv2.VideoCapture(video_path)
    frame_rate = 1
    prev = 0
    n = 0
    unet = UNet()
    unet.load_state_dict(
        torch.load(
            "/Users/jasonyuan/Desktop/UNet Weights/unet_model_batch64_scheduled_lr0.05_epochs40_e14_best.pt",
            map_location=torch.device("cpu"),
        )
    )

    frame = cv2.imread(
        "/Users/jasonyuan/Desktop/UTRA:Projects/ART stuff/lane_dataset grass/image_rect_color_screenshot_09.12.2017 23.png"
    )
    annotated, pred = predict_lanes(frame, unet)

    cv2.imwrite("/Users/jasonyuan/Desktop/Test9.png", pred * 255)

    cv2.imshow("Annotated", annotated)
    cv2.imshow("Og", frame)
    cv2.imshow("Pred", pred)
    cv2.waitKey(0)

    # out = cv2.VideoWriter('/Users/jasonyuan/Desktop/Processed_Lane_vid_2.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 10, (640,360))
    #
    # # for file in sorted(os.listdir(image_path),key=sortkey):
    # #     if file == ".DS_Store":
    # #         continue
    # #     print(file)
    # #     frame = cv2.imread(os.path.join(image_path,file))
    # #     annotated = predict_lanes(frame,unet)
    # #     out.write(cv2.resize(annotated,(640,360),interpolation=cv2.INTER_AREA))
    #
    # start = time.time()
    # while cap.isOpened():
    #     time_elapsed = time.time() - prev
    #     if time_elapsed > 1./frame_rate:
    #         prev = time.time()
    #         ret,frame = cap.read()
    #         if ret == False:
    #             print("ret was False")
    #             break
    #
    #         overlayed_mask = predict_lanes(frame,unet)
    #         out.write(cv2.resize(overlayed_mask,(640,360),interpolation=cv2.INTER_AREA))
    #
    #         # frame = cv2.resize(frame,(1280,720),interpolation=cv2.INTER_AREA)
    #         # frame_copy = np.copy(frame)
    #         # # roi2,M2,M_inv2 = define_region_of_interest(frame)
    #         # # roi,M,M_inv = define_region_of_interest(frame_copy)
    #         # # print(roi.shape)
    #         # # roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    #         # # roi = roi.reshape(1,1,180,330)
    #         # # roi = roi/255.
    #         # # input = torch.Tensor(roi)
    #         #
    #         # test_edges,test_edges_inv = find_edge_channel2(frame_copy)
    #         #
    #         # frame_copy = np.append(frame_copy,test_edges.reshape(test_edges.shape[0],test_edges.shape[1],1),axis=2)
    #         # frame_copy = np.append(frame_copy,test_edges_inv.reshape(test_edges_inv.shape[0],test_edges_inv.shape[1],1),axis=2)
    #         #
    #         # frame_copy = cv2.resize(frame_copy,(330,180))
    #         #
    #         # input = (torch.Tensor(frame_copy/255.).permute(2,0,1)).reshape(1,5,180,330)
    #         #
    #         # # input = torch.Tensor((cv2.cvtColor(input_img_copy,cv2.COLOR_BGR2GRAY)/255.).reshape(1,1,180,330))
    #         #
    #         # unet.eval()
    #         # output = unet(input)
    #         # # print(output)
    #         # # print(unet)
    #         #
    #         # output = torch.sigmoid(output)
    #         # output = output.detach().numpy()
    #         # pred_mask = np.where(output>0.5,1,0)
    #         #
    #         # # print(output)
    #         # # print(ground_truth.shape)
    #         # # print(pred_mask.size())
    #         # pred_mask = (pred_mask.squeeze(0)).transpose(1,2,0).squeeze().astype('float32')
    #         # # pred_mask = cv2.resize(pred_mask,(1280,720),interpolation=cv2.INTER_AREA)
    #         #
    #         # overlayed_mask = cv2.resize(np.copy(frame),(330,180),interpolation=cv2.INTER_AREA)
    #         # # overlayed_mask = np.copy(frame)
    #         # overlayed_mask[np.where(pred_mask==1)[0],np.where(pred_mask==1)[1],2] = 255
    #         # overlayed_mask[np.where(pred_mask==1)[0],np.where(pred_mask==1)[1],1] = 0
    #         # overlayed_mask[np.where(pred_mask==1)[0],np.where(pred_mask==1)[1],0] = 0
    #         # # overlayed_mask = cv2.warpPerspective(overlayed_mask,M_inv2,(1280,720),cv2.INTER_LINEAR)
    #         #
    #         # # overlayed = np.copy(frame)
    #         # # overlayed[np.where(overlayed_mask != [0,0,0])] = overlayed_mask[np.where(overlayed_mask != [0,0,0])]
    #         # # overlayed = cv2.resize(overlayed,(1280,720),interpolation=cv2.INTER_AREA)
    #         #
    #         # # cv2.imshow("Video",frame)
    #         # # cv2.imshow("Edges",test_edges)
    #         # cv2.imshow("Annotated",cv2.resize(overlayed_mask,(640,360),interpolation=cv2.INTER_AREA))
    #         # if cv2.waitKey(1) & 0xFF == ord('q'):
    #         #     break
    #
    #         # cv2.imwrite(os.path.join(save_path,"Frame_{}.png".format(n)),overlayed_mask)
    #         # n += 1
    #
    # end = time.time()
    #
    # print("Total Processing time: {}".format(end-start))
    #
    # out.release()
    # cap.release()
    cv2.destroyAllWindows()


# C++ Implementation of find_edge_channel ######################################

# cv::CV_8UC1 getMedian(std::vector<cv::CV_8UC1> input2vec) {
#   std::nth_element(input2vec.begin(), input2vec.begin() + input2vec.size() / 2, input2vec.end());
#   return input2vec[input2vec.size() / 2];
# }
#
# std::vector<cv::CV_8UC1> convertToQuadrant(cv::Mat input, int r_min, int r_max,
#                                             int c_min, int c_max) {
#
#   std::vector<cv::CV_8UC1> output;
#
#   for (int r = r_min; r < r_max; r++) {
#     for (int c = c_min; c < c_max; c++) {
#       output.emplace_back(input[r][c]);
#     }
#   }
#
#   return output;
# }
#
# std::pair<cv::Mat, cv::Mat> find_edge_channel(cv::Mat img) {
#
#     width = img.cols;
#     height = img.rows
#
#     cv::Mat edges_mask;
#     cv::Mat edges_mask_inv;
#     cv::Mat gray_im;
#
#     cv::cvtColor(img,gray_im,cv::COLOR_BGR2GRAY);
#
#     // gray_im = cv2.GaussianBlur(gray_im,(3,3),0)
#     // Separate into quadrants
#     std::vector<cv::CV_8UC1> quad1 = convertToQuadrant(gray_im,0,static_cast<int>(height/2),
#                                                             0,static_cast<int>(width/2));
#
#     std::vector<cv::CV_8UC1> quad2 = convertToQuadrant(gray_im,0,static_cast<int>(height/2),
#                                                             static_cast<int>(width/2),width);
#
#     std::vector<cv::CV_8UC1> quad3 = convertToQuadrant(gray_im,static_cast<int>(height/2),height,
#                                                             static_cast<int>(width/2),width);
#
#     std::vector<cv::CV_8UC1> quad4 = convertToQuadrant(gray_im,static_cast<int>(height/2),height,
#                                                             0,static_cast<int>(width/2));
#     double med1 = static_cast<double>(getMedian(quad1));
#     double med2 = static_cast<double>(getMedian(quad2));
#     double med3 = static_cast<double>(getMedian(quad3));
#     double med4 = static_cast<double>(getMedian(quad4));
#
#     double l1 = std::max(0,std::floor((1-0.205)*med1));
#     double u1 = std::min(255,std::floor((1+0.205)*med1));
#     std::vector<int> quad1_int(quad1.begin(),quad1.end());
#     cv::Mat quad1_mat(quad1_int,cv::CV_8UC1);
#     cv::Mat e1;
#     cv::Canny(quad1_mat,e1,l1,u1);
#
#     double l2 = std::max(0,std::floor((1-0.205)*med2));
#     double u2 = std::min(255,std::floor((1+0.205)*med2));
#     std::vector<int> quad2_int(quad2.begin(),quad2.end());
#     cv::Mat quad2_mat(quad2_int,cv::CV_8UC1);
#     cv::Mat e2;
#     cv::Canny(quad2_mat,e2,l2,u2);
#
#     double l3 = std::max(0,std::floor((1-0.205)*med3));
#     double u3 = std::min(255,std::floor((1+0.205)*med3));
#     std::vector<int> quad3_int(quad3.begin(),quad3.end());
#     cv::Mat quad3_mat(quad3_int,cv::CV_8UC1);
#     cv::Mat e3;
#     cv::Canny(quad3_mat,e3,l3,u3);
#
#     double l4 = std::max(0,std::floor((1-0.205)*med4));
#     double u4 = std::min(255,std::floor((1+0.205)*med4));
#     std::vector<int> quad4_int(quad4.begin(),quad4.end());
#     cv::Mat quad4_mat(quad4_int,cv::CV_8UC1);
#     cv::Mat e4;
#     cv::Canny(quad4_mat,e4,l4,u4);
#
#     // Stitch the edges together
#     int new_rows = std::max(e1.rows+e3.rows,e2.rows+r4.rows);
#     int new_cols = std::max(e1.cols+e2.cols,e3.cols+e4.cols);
#     edges_mask = cv::zeros(new_rows, new_cols, cv::CV_8UC1);
#
#     e1.copyTo(edges_mask(cv::Rect(0,0,e1.cols,e1.rows)));
#     e2.copyTo(edges_mask(cv::Rect(e1.cols,0,e2.cols,e2.rows)));
#     e3.copyTo(edges_mask(cv::Rect(0,e1.rows,e3.cols,e3.rows)));
#     e4.copyTo(edges_mask(cv::Rect(e1.cols,e1.rows,e4.cols,e4.rows)));
#
#     cv::bitwise_not(edges_mask,edges_mask_inv);
#
#     return std::make_pair(edges_mask, edges_mask_inv);
# }
