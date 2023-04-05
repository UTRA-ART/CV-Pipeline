from UNet import UNet
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
from UNetDataset import LaneDataset

video_path = "/Users/jasonyuan/Desktop/output_video.mp4"
# video_path = "/Users/jasonyuan/Desktop/Section_of_dashcam.mp4"
save_path = "output/"
# image_path = "C:\\Users\\ammar\\Documents\\CodingProjects\\ART\\CV-Pipeline\\src\\lane_detection\\UNet-LaneDetection\\input\\additional-data\\inputs"
# image_path = "C:\\Users\\ammar\\Documents\\CodingProjects\\ART\\CV-Pipeline\\src\\lane_detection\\UNet-LaneDetection\\input\\additional-data\\inputs\\Gravel_10.png"
image_path = "D:\\UTRA\\CompetitionLaneLineLabelled\\inputs"
# image_path = "D:\\UTRA\\UNet dataset\\inputs\\25.jpg"
# image_path = r"input\unet-lanes-v3\Dataset 3\Day Time\inputs"
# image_path = r"C:\Users\ammar\Documents\CodingProjects\ART\Data\LaneDataForPothole\Toronto Dashcam video_0"
# image_path = r'input\lanes4.jpg'
# image_path = r"input\unet-lanes-v3\Dataset 3\Past Comp Data\inputs\Lane_Input_1054.png"
# image_path = r'input\tusimple_unet_v2\inputs\35.jpg'

img_dir = True

# weights_path = 'runs/1668151763.9445446/1668151763.9445446unet_gray_model_batch64_sheduled_lr0.1_epochs15.pt'
# weights_path = r"runs\1668662675.3081253\1668662675.3081253unet_gray_model_batch64_sheduled_lr0.1_last.pt"
# weights_path = r"C:\Users\ammar\Documents\CodingProjects\ART\CV-Pipeline\src\lane_detection\UNet-LaneDetection\runs\1668886756.1157043\1668886756.1157043unet_gray_model_batch64_sheduled_lr0.1_last.pt"
weights_path = "D:\\UTRA CV-Pipeline\\src\\lane_detection\\runs\\Mar_17_08_2023\\unet_batch1_lr0.1_ep79.pt"


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
    med2 = np.median(gray_im[: height // 2, width // 2:])
    med3 = np.median(gray_im[height // 2:, width // 2:])
    med4 = np.median(gray_im[height // 2:, : width // 2])

    l1 = int(max(0, (1 - 0.205) * med1))
    u1 = int(min(255, (1 + 0.205) * med1))
    e1 = cv2.Canny(gray_im[: height // 2, : width // 2], l1, u1)

    l2 = int(max(0, (1 - 0.205) * med2))
    u2 = int(min(255, (1 + 0.205) * med2))
    e2 = cv2.Canny(gray_im[: height // 2, width // 2:], l2, u2)

    l3 = int(max(0, (1 - 0.205) * med3))
    u3 = int(min(255, (1 + 0.205) * med3))
    e3 = cv2.Canny(gray_im[height // 2:, width // 2:], l3, u3)

    l4 = int(max(0, (1 - 0.205) * med4))
    u4 = int(min(255, (1 + 0.205) * med4))
    e4 = cv2.Canny(gray_im[height // 2:, : width // 2], l4, u4)

    # Stitch the edges together
    edges_mask[: height // 2, : width // 2] = e1
    edges_mask[: height // 2, width // 2:] = e2
    edges_mask[height // 2:, width // 2:] = e3
    edges_mask[height // 2:, : width // 2] = e4

    edges_mask_inv = cv2.bitwise_not(edges_mask)

    return edges_mask, edges_mask_inv


def predict_lanes(frame, unet):
    frame_copy = np.copy(frame)
    cv2.imwrite('gradient.png', frame_copy[:, :, 3])
    frame_copy = cv2.resize(frame_copy, (256, 160))

    input = torch.Tensor((frame_copy/255.0).transpose(2, 0, 1)).reshape(
        1, 4, 160, 256
    )
    # x = (frame_copy/255.).transpose(2,0,1).reshape(1,5,180,330).astype(np.float32)

    # input = torch.Tensor((cv2.cvtColor(input_img_copy,cv2.COLOR_BGR2GRAY)/255.).reshape(1,1,180,330))
    # ort_sess = ort.InferenceSession('/Users/jasonyuan/Desktop/unet_with_sigmoid.onnx')
    # output = ort_sess.run(None, {'Inputs': x})[0]

    unet.eval()

    input = input.to(device="cuda")
    unet = unet.to(device="cuda")

    times = []
    frames = 1
    for i in range(0, frames):
        t1 = time.time()
        output = unet(input)[0][0]
        # output = ort_session.run(None, {'Inputs': input.cpu().numpy()})[0][0][0]
        t2 = time.time()
        times.append(t2 - t1)
    print(f'Done. Inference @ {1 / np.mean(times)} fps ')

    # print(output)
    # print(unet)
    print(output.shape)
    output = torch.tensor(output)
    output = torch.sigmoid(output)
    output = output.detach().cpu().numpy()
    pred_mask = np.where(output > 0.5, 1, 0).astype("float32")
    print(np.any(pred_mask))
    # print(output)
    # print(ground_truth.shape)
    # print(pred_mask.size())
    # pred_mask = (pred_mask.squeeze(0)).transpose(
    #     1, 2, 0).squeeze().astype("float32")
    # pred_mask = cv2.resize(pred_mask,(1280,720),interpolation=cv2.INTER_AREA)

    bgr_frame = cv2.cvtColor(frame[:,:,:3], cv2.COLOR_HSV2BGR)
    overlayed_mask = np.copy(cv2.resize(bgr_frame, (256, 160)))
    # overlayed_mask = np.copy(input_img)
    overlayed_mask[np.where(pred_mask == 1)[0],
                   np.where(pred_mask == 1)[1], 2] = 255
    overlayed_mask[np.where(pred_mask == 1)[0],
                   np.where(pred_mask == 1)[1], 1] = 0
    overlayed_mask[np.where(pred_mask == 1)[0],
                   np.where(pred_mask == 1)[1], 0] = 0

    # print(pred_mask.sum())
    # cv2.imshow("Input", frame)
    # cv2.imshow("Overlayed", overlayed_mask)
    # cv2.waitKey(0)

    return overlayed_mask, pred_mask, 1/np.mean(times)


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
        torch.load(weights_path,
                   map_location=torch.device("cuda"),
                   )
    )
    imagePaths = []
    for img in os.listdir(image_path):
        img_path = os.path.join(image_path, img)
        imagePaths.append(img_path)
    test_dataset = LaneDataset(imagePaths, None)
    # model = onnx.load('unet_v1.onnx')
    # onnx.checker.check_model(model)
    # ort_session = ort.InferenceSession('unet_v1.onnx', providers=['CUDAExecutionProvider'])
    # print(ort.get_device())
    count = 0
    all_fps = []
    out = cv2.VideoWriter('competition_ordered_tusimple.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 2, (640, 360))
    if img_dir:

        for img_idx in range(test_dataset.__len__()):
            if count == 200: break
            frame, _, path = test_dataset.__getitem__(img_idx)
            print((frame.shape[0], frame.shape[1]))
            annotated, pred, fps = predict_lanes(frame, unet)
            h, w = frame.shape[0:2]
            print(frame.shape)
            pred = np.squeeze(pred)
            annotated = np.squeeze(annotated)
            print(f"annotated shape is {annotated.shape}")
            annotated = cv2.resize(annotated, (640, 360))
            pred = cv2.resize(pred, (640, 360))
            all_fps.append(fps)
            # filename = image_path.split('/')[-1]
            # cv2.imwrite(save_path+filename, annotated)
            # cv2.imshow('test', pred*255)
            # cv2.waitKey(0)
            out.write(annotated)
            count += 1

        avg_fps = sum(all_fps)/count
        print(f"All done, avg fps is {avg_fps}")
        out.release()
    else:

        frame = cv2.imread(image_path)
        print((frame.shape[0], frame.shape[1]))
        annotated, pred, _ = predict_lanes(frame, unet)
        pred = np.squeeze(pred)
        annotated = np.squeeze(annotated)
        h, w = frame.shape[0:2]
        print(frame.shape)
        annotated = cv2.resize(annotated, (w, h))
        pred = cv2.resize(pred, (w, h))
        print(pred.shape)

        filename = image_path.split('/')[-1]
        cv2.imwrite(save_path+filename, annotated)
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
