from tool.darknet2pytorch import Darknet
import torch
WEIGHTS = Darknet('cfg/yolov4.cfg')
WEIGHTS.load_weights('weights/yolov4.weights')

torch.save(WEIGHTS, 'weights/yolov4.pth')