from UNet import UNet
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
from UNetPrediction import predict_lanes

weights_path = "D:\\UTRA CV-Pipeline\\src\\lane_detection\\runs\\Mar_17_08_2023\\unet_batch1_lr0.1_ep79.pt"

class Inference():
    def __init__(self, root = None, debug=False):
        self.model = UNet()
        self.model.load_state_dict(
                torch.load(weights_path,
                   map_location=torch.device("cuda"),
                )
        )
        self.model.eval()
        self.model.to(device="cuda")

        self.debug = debug

    def inference(self, frame):
        annotated, pred = self.predict_lanes(frame)
        annotated = cv2.resize(annotated, (640, 360))
        pred = cv2.resize(pred, (640, 360))
        return annotated

    def predict_lanes(self, frame):
        cv2.imwrite('gradient.png', frame[:, :, 3])
        frame = cv2.resize(frame, (256, 160))

        input = torch.Tensor((frame/255.0).transpose(2, 0, 1)).reshape(
            1, 4, 160, 256
        )
        input = input.to(device="cuda")
        
        output = self.model(input)[0][0]
        output = torch.tensor(output)
        output = torch.sigmoid(output)
        output = output.detach().cpu().numpy()
        pred_mask = np.where(output > 0.5, 1, 0).astype("float32")
        
        if self.debug:
            bgr_frame = cv2.cvtColor(frame[:,:,:3], cv2.COLOR_HSV2BGR)
            overlayed_mask = np.copy(cv2.resize(bgr_frame, (256, 160)))
            overlayed_mask[np.where(pred_mask == 1)[0],
                        np.where(pred_mask == 1)[1], 2] = 255
            overlayed_mask[np.where(pred_mask == 1)[0],
                        np.where(pred_mask == 1)[1], 1] = 0
            overlayed_mask[np.where(pred_mask == 1)[0],
                        np.where(pred_mask == 1)[1], 0] = 0


        return overlayed_mask, pred_mask


# Testing 
image_path = "D:\\UTRA\\CompetitionLaneLineLabelled\\inputs"
imagePaths = []
for img in os.listdir(image_path):
    img_path = os.path.join(image_path, img)
    imagePaths.append(img_path)
test_dataset = LaneDataset(imagePaths, None)

Output = Inference()
for img_idx in range(test_dataset.__len__()):
    frame, _, path = test_dataset.__getitem__(img_idx)
    annotated = Output.inference(frame)
    cv2.imshow("Predict", annotated)
    if cv2.waitKey(0) == 'q':
        cv2.destroyWindow("Predict")
