from tool.darknet2pytorch import Darknet
import torch
from torch import nn
from torch import optim


def convert_to_torch(cfg, weights, out_dir): # (.cfg, .weights, .pth)
    WEIGHTS = Darknet('cfg/yolov4.cfg')
    WEIGHTS.load_weights('weights/yolov4.weights')

    torch.save(WEIGHTS, out_dir)

def transfer_train(num_classes):
    model = torch.load('weights/yolov4.pth')
    num_ftrs = num_classes

    print(model)

    model.fc = nn.Linear(num_ftrs, num_classes)
    model.cuda()

    criterion = nn.CrossEntropyLoss
    optimizer = optim.SGD(model.parameters(), lr = 0.001)

    #Scheduler

transfer_train(10)