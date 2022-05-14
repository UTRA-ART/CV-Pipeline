import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from UNetDataset import LaneDataset
from UNetModel import UNet
import cv2
import numpy as np
import os
import PIL
import matplotlib.pyplot as plt
import random

torch.manual_seed(123)
random.seed(29)

def sortKey(x):
    x1 = x.split("/")[-1]
    x2 = x1.split("_")[-1]
    val = int(x2.split(".")[0])
    return val

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()

def accuracy(model,dataloader):
    if torch.cuda.is_available():
        model.cuda()

    model.eval()
    total_correct = 0
    total_inputs = 0
    for i,data in enumerate(dataloader,0):
        image,label = data

        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()

        out = model(image)
        pred = torch.sigmoid(out)
        pred_np = pred.detach().cpu().numpy()
        labels_np = label.detach().cpu().numpy()

        masked_pred = np.where(pred_np>0.017,1,0) # Thresholds prediction
        correct = (masked_pred==labels_np.astype('int64')).sum()
        incorrect = (masked_pred!=labels_np.astype('int64')).sum()

        total_correct += correct
        total_inputs += correct+incorrect

    if total_inputs == 0:
      return 0

    return total_correct/total_inputs

# ----------------- Collect all the file names into two lists ------------------
input_dir = "../input/unet-lanes-v2/Dataset2/inputs"
mask_dir = "../input/unet-lanes-v2/Dataset2/labels"

imagePaths = []
maskPaths = []

for filename in os.listdir(input_dir):
    if filename != ".DS_Store":
        img_path = os.path.join(input_dir,filename)
        imagePaths.append(img_path)

for filename in os.listdir(mask_dir):
    if filename != ".DS_Store":
        mask_path = os.path.join(mask_dir,filename)
        maskPaths.append(mask_path)

imagePaths = sorted(imagePaths,key=sortKey)
maskPaths = sorted(maskPaths,key=sortKey)

# ------------- Instantiate the custom dataset and dataloaders -----------------
# Do an 85% - 15% split of the images for training and validation
transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((180,330)),
                                transforms.ToTensor()])
all_idx = np.arange(0,len(imagePaths)).tolist()
random.shuffle(all_idx)

split = int(np.ceil(1*len(all_idx)))
split2 = int(np.ceil(0.75*len(all_idx)))
train_idx = all_idx[:split]
# valid_idx = all_idx[split:]
# train3_idx = all_idx[split2:]

train_images = []
train_labels = []
val_images = []
val_labels = []
# train3_images = []
# train3_labels = []

for idx in train_idx:
    train_images.append(imagePaths[idx])
    train_labels.append(maskPaths[idx])
# for idx in valid_idx:
#     val_images.append(imagePaths[idx])
#     val_labels.append(maskPaths[idx])
# for idx in train3_idx:
#     train3_images.append(imagePaths[idx])
#     train3_labels.append(maskPaths[idx])

trainset = LaneDataset(train_images,train_labels,transforms=transform)
# valset = LaneDataset(val_images,val_labels,transforms=transform)
# trainset3 = LaneDataset(train3_images,train3_labels,transforms=transform)

trainloader = DataLoader(trainset,
                        batch_size=16,
                        shuffle=True,
                        num_workers=0)
# validloader = DataLoader(valset,
#                         batch_size=16,
#                         shuffle=True,
#                         num_workers=0)
# trainloader3 = DataLoader(trainset3,
#                         batch_size=10,
#                         shuffle=True,
#                         num_workers=0)

# ---------------------- Initialize the training loop --------------------------
l_rate = 0.001
num_epochs = 65
train_loss = []
train_error = []
val_loss = []
val_error = []
epochs = []

unet = UNet()
if torch.cuda.is_available():
    unet.cuda()

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(unet.parameters(),lr=l_rate)
# exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

for e in range(num_epochs):
    total_train_loss = 0
    total_val_loss = 0
    total_train_error = 0
    total_val_error = 0
    num_train_iterations = 0
    num_val_iterations = 0

    unet.train()
    for i,data in enumerate(trainloader,0):
#         print("Training Iteration: {}".format(i+1))

        images,labels = data
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()
        out = unet(images)
        pred = torch.sigmoid(out)
#         loss = criterion(out,labels)*0.4 + 0.6*
# #         loss = criterion(out,labels)*0.5 + (1-0.5)*dice_loss((pred.detach()>0.5).int(),labels)
#         loss.backward()
#         optimizer.step()

        # Check if the mask is truly binary
        test_label = labels.detach().cpu().numpy()
        num_not_binary = np.where(((test_label>0)&(test_label<1)),1,0).sum()
        # For calculating error
        pred_np = pred.detach().cpu()
        labels_np = labels.detach().cpu()

        masked_pred = np.where(pred_np>0.5,1,0) # Thresholds prediction
        masked_pred = (pred_np>0.5).int()
        correct = torch.sum(torch.bitwise_and(masked_pred,labels_np.type(torch.int32))).item()
        incorrect = torch.sum(torch.bitwise_or(masked_pred,labels_np.type(torch.int32))).item()

#         if (labels_np==1).sum().item() == 0:
#             print((labels_np>0).sum().item())
#             plt.figure()
#             plt.subplot(1,2,1)
#             plt.imshow(images.detach().cpu().numpy().squeeze().transpose(1,2,0))
#             plt.subplot(1,2,2)
#             plt.imshow(labels_np.numpy().squeeze())
#         print(labels)
        error = incorrect/(correct+incorrect)
#         loss = criterion(out,labels)
        loss = criterion(out,labels)*0.5 + 0.5*dice_loss((pred.detach()>0.5).float(),labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        total_train_error += error
        num_train_iterations += 1

    train_error.append(total_train_error/num_train_iterations)
    train_loss.append(total_train_loss/num_train_iterations)
#     exp_lr_scheduler.step()

#     with torch.no_grad():
#         unet.eval()

#         for i,data in enumerate(validloader,0):
# #             print("Validation Iteration: {}".format(i+1))

#             images,labels = data
#             if torch.cuda.is_available():
#                 images = images.cuda()
#                 labels = labels.cuda()

#             out = unet(images)
#             loss = criterion(out,labels)

#             # For calculating error
#             pred = torch.sigmoid(out)
#             pred_np = pred.detach().cpu().numpy()
#             labels_np = labels.detach().cpu().numpy()

#             masked_pred = np.where(pred_np>0.5,1,0)
#             correct = (np.bitwise_and(masked_pred,labels_np.astype('int64')).sum())
#             incorrect = (np.bitwise_or(masked_pred,labels_np.astype('int64')).sum())

#             error = incorrect/(correct+incorrect)

#             total_val_loss += loss.item()
#             total_val_error += error
#             num_val_iterations += 1

#         val_error.append(total_val_error/num_val_iterations)
#         val_loss.append(total_val_loss/num_val_iterations)

    print("Epoch: {}".format(e+1))
    print("Training Error: {} | Training Loss: {} | Number of non-binary: {}".format(total_train_error/num_train_iterations,total_train_loss/num_train_iterations,num_not_binary))
#     print("Validation Error: {} | Validation Loss: {}".format(total_val_error/num_val_iterations,total_val_loss/num_val_iterations))

    epochs.append(e+1)

# f_train_accuracy = accuracy(unet,trainloader)
# # f_val_accuracy = accuracy(unet,validloader)
# # print("Final Training Error: {} | Final Validation Error: {}".format(f_train_accuracy,f_val_accuracy))
# print("Final Training Error: {}".format(f_train_accuracy))

# ------------------ Plot the training and validation curves -------------------
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

ax1.plot(epochs,train_error,label="Training")
# ax1.plot(epochs,val_error,label="Validation")
ax1.set_title("Model Error Curves")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Error")
ax1.legend()

ax2.plot(epochs,train_loss,label="Training")
# ax2.plot(epochs,val_loss,label="Validation")
ax2.set_title("Model Loss Curves")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()

fig.tight_layout()
plt.show()

# torch.save(unet.state_dict(),"Some Path")

# Loading model weights
# unet.load_state_dict(torch.load("Some Path"))
