#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Modules


# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import PIL
import matplotlib.pyplot as plt
import random
import math
import tqdm
import time

import wandb


# # Defining the UNet model and the custom dataset class

# In[12]:


class LaneDataset(Dataset):
    def __init__(self,imagePath,maskPath,prob=0,transforms=None):
        self.imagePath = imagePath # Array of filepaths for the input images
        self.maskPath = maskPath # Array of filepaths for the mask images
        self.transforms = transforms
        self.prob = prob

    def __len__(self):
        return len(self.imagePath)

    def __getitem__(self,idx):
        img_path = self.imagePath[idx]
        mask_path = self.maskPath[idx]

        image = cv2.imread(img_path)
        if image is None:
            print(img_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)

        edges,edges_inv = self.find_edge_channel(image)
        
        gradient_map = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=-1) # Gradient map along x
#         gradient_map = cv2.Laplacian(gray, cv2.CV_64F)
        gradient_map = np.uint8(np.absolute(gradient_map))
        
        output_image = np.zeros((gray.shape[0],gray.shape[1],3),dtype=np.uint8)
        output_image[:,:,0] = gray
        output_image[:,:,1] = edges
        output_image[:,:,2] = gradient_map
    
        output_image, mask = self.prob_rotate(output_image,mask)
        output_image, mask = self.prob_flip(output_image,mask)

        if self.transforms != None:
            output_image = self.transforms(output_image)
            mask = self.transforms(mask)
        
        mask_binary = (mask>0).type(torch.float)
        
        return (output_image,mask_binary, img_path)
    
    def prob_flip(self,img,lbl):
        if random.random() > self.prob:
            return img,lbl
        flip_img = cv2.flip(img,1)
        flip_lbl = cv2.flip(lbl,1)
        return flip_img,flip_lbl
    
    def prob_rotate(self,img,lbl):
        if random.random() > self.prob: 
            return img,lbl
        
        rotations = [-90,-45,45,90,180]
        angle = random.choice(rotations)
        center_img = (img.shape[1]//2,img.shape[0]//2)
        center_lbl = (lbl.shape[1]//2,lbl.shape[0]//2)
        
        rotate_matrix_img = cv2.getRotationMatrix2D(center=center_img,angle=angle,scale=1)
        rotate_matrix_lbl = cv2.getRotationMatrix2D(center=center_lbl,angle=angle,scale=1)
        
        rotated_img = cv2.warpAffine(img,rotate_matrix_img,(img.shape[1],img.shape[0]))
        rotated_lbl = cv2.warpAffine(lbl,rotate_matrix_lbl,(lbl.shape[1],lbl.shape[0]))
        
        return rotated_img,rotated_lbl
    
    def find_edge_channel(self,img):
        edges_mask = np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8)
        width = img.shape[1]
        height = img.shape[0]
        
        gray_im = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        # Separate into quadrants
        med1 = np.median(gray_im[:height//2,:width//2])
        med2 = np.median(gray_im[:height//2,width//2:])
        med3 = np.median(gray_im[height//2:,width//2:])
        med4 = np.median(gray_im[height//2:,:width//2])

        l1 = int(max(0,(1-0.205)*med1))
        u1 = int(min(255,(1+0.205)*med1))
        e1 = cv2.Canny(gray_im[:height//2,:width//2],l1,u1)

        l2 = int(max(0,(1-0.205)*med2))
        u2 = int(min(255,(1+0.205)*med2))
        e2 = cv2.Canny(gray_im[:height//2,width//2:],l2,u2)

        l3 = int(max(0,(1-0.205)*med3))
        u3 = int(min(255,(1+0.205)*med3))
        e3 = cv2.Canny(gray_im[height//2:,width//2:],l3,u3)

        l4 = int(max(0,(1-0.205)*med4))
        u4 = int(min(255,(1+0.205)*med4))
        e4 = cv2.Canny(gray_im[height//2:,:width//2],l4,u4)

        # Stitch the edges together
        edges_mask[:height//2,:width//2] = e1
        edges_mask[:height//2,width//2:] = e2
        edges_mask[height//2:,width//2:] = e3
        edges_mask[height//2:,:width//2] = e4
        
        edges_mask_inv = cv2.bitwise_not(edges_mask)
        
        return edges_mask, edges_mask_inv


# In[13]:


from operator import indexOf
from numpy import ndim
import torch.nn as nn
import torch


class ConvStage(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(ConvStage, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[16, 32, 64, 128]):
        super(UNet, self).__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        # Setting up the architecture for the encoder
        for feature in features:
            double_conv = ConvStage(
                in_channels=in_channels, out_channels=feature)
            self.encoder.append(double_conv)
            in_channels = feature

        self.bottleneck = ConvStage(
            in_channels=features[-1], out_channels=features[-1] * 2
        )

        # Setting up the architecrue for the decoder
        for feature in reversed(features):
            up_conv = nn.ConvTranspose2d(
                in_channels=feature * 2, out_channels=feature, kernel_size=2, stride=2
            )
            self.decoder.append(up_conv)
            double_conv = ConvStage(
                in_channels=feature * 2, out_channels=feature)
            self.decoder.append(double_conv)

        self.segmentation = nn.Conv2d(
            in_channels=features[0], out_channels=out_channels, kernel_size=1, stride=1
        )

    def forward(self, x):
        # Make sure that the inputted size is compatible
        assert x.shape[2] % 16 == 0 and x.shape[3] % 16 == 0

        copies = []

        # Forward pass through the encoder
        for i, down in enumerate(self.encoder):
            x = down(x)
            # Store a copy
            copies.append(x)
            x = self.pool(x)

        # The bottleneck
        x = self.bottleneck(x)

        # Reverse the coppies
        copies = copies[::-1]

        # Forward pass through the decoder
        for j, up in enumerate(self.decoder):
            if j % 2 == 0:
                x = up(x)
                x = torch.cat((copies[j // 2], x), axis=1)
            else:
                x = up(x)

        return self.segmentation(x)


def test():
    x = torch.rand((3, 3, 160, 256))
    model = UNet(in_channels=2, out_channels=1)
    pred = model(x)
    print(x.shape, pred.shape)
    
# test()


# In[6]:


# def dice_loss(pred, target, smooth = 1.):
#     pred = pred.contiguous()
#     target = target.contiguous()    

#     intersection = (pred * target).sum(dim=2).sum(dim=2)
    
#     loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
#     return loss.mean()

def dice_loss(inputs, target):
    num = target.size(0)
    inputs = inputs.reshape(num, -1)
    target = target.reshape(num, -1)
    smooth = 1.0
    intersection = (inputs * target)
    dice = (2. * intersection.sum(1) + smooth) / (inputs.sum(1) + target.sum(1) + smooth)
    dice = 1 - dice.sum() / num
    return dice


# In[7]:


def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
        
    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)
    
    loss = bce * bce_weight + dice * (1 - bce_weight)
    
    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    
    return loss


# # Training Code and Loop

# In[33]:


torch.manual_seed(250)
random.seed(2)

def sortKey(x):
    x1 = x.split("/")[-1]
    x2 = x1.split("_")[-1]
    val = int(x2.split(".")[0])
    return val

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



def load_data(location):
    dataPaths = []
    labelPaths = []

    for folder in os.listdir(location):
        if os.path.isdir(location+"/"+folder):
            if folder == "inputs":
                for filename in os.listdir(location+"/"+folder):
                    dataPaths = dataPaths + [location+"/"+folder+"/"+filename]
                    # mask_name = filename.split(".")[0]+"_Label.png"
                    mask_name = filename
                    labelPaths = labelPaths + [location+"/labels/"+mask_name]
    return dataPaths, labelPaths


# ----------------- Collect all the file names into two lists ------------------

base_path = "input/unet-lanes-v3/Dataset 3"
tusimple_path = "input/tusimple"
competition_path = "input/additional-data"


imagePaths = []
maskPaths = []

validPaths = []
valid_lblPaths = []

imagePaths, maskPaths =load_data(tusimple_path)
# load_data(imagePaths, maskPaths, competition_path)

    
print(len(imagePaths),len(maskPaths))

# ------------- Instantiate the custom dataset and dataloaders -----------------
# Do an 85% - 15% split of the images for training and validation
transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((160,256)),
                                transforms.ToTensor()])

all_idx = np.arange(0,len(imagePaths)).tolist()
all_valid_idx = np.arange(0,len(validPaths)).tolist()
random.shuffle(all_idx)
random.shuffle(all_valid_idx)

split = int(np.ceil(0.2*len(all_idx)))
valid_idx_additional = all_idx[:split]

train_images = []
train_labels = []

valid_images = []
valid_labels = []

for idx in all_idx[split:]:
    train_images.append(imagePaths[idx])
    train_labels.append(maskPaths[idx])

for idx in all_valid_idx:
    valid_images.append(validPaths[idx])
    valid_labels.append(valid_lblPaths[idx])

for idx in valid_idx_additional:
    valid_images.append(imagePaths[idx])
    valid_labels.append(maskPaths[idx])
    
print(f'{len(train_images)} train images,{len(train_labels)} train labels')
print(f'{len(valid_images)} val images,{len(valid_labels)} val labels')

trainset = LaneDataset(train_images,train_labels,prob=0.15,transforms=transform)
validset = LaneDataset(valid_images,valid_labels,prob=0.6,transforms=transform)

batch = 64

trainloader = DataLoader(trainset,
                        batch_size=batch,
                        num_workers=0)

validloader = DataLoader(validset,
                        batch_size=batch,
                        num_workers=0)


# ---------------------- Initialize the training loop --------------------------
timeStarted = time.time()
wandb.init(project='unet_shrinking', name=f'{timeStarted}')
config = wandb.config

os.mkdir(f'./runs/{timeStarted}/')

config.l_rate = 0.1
config.momentum = 0.9
num_epochs = 40   # Start smaller to actually make sure that the model is not overfitting due to data similarities

train_loss = []
train_error = []
val_loss = []
val_error = []
epochs = []
lr_vals = []
min_loss = np.inf

if torch.cuda.is_available():
    print('Using CUDA.')
    device = torch.device('cuda')
model = UNet()
# model.load_state_dict(torch.load("runs/1667872806.510513/1667872806.510513unet_gray_model_batch64_sheduled_lr0.1_last.pt"))

if torch.cuda.is_available():
    model.cuda()
    model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(),lr=config.l_rate,momentum=config.momentum)
# optimizer = optim.Adam(unet.parameters(),lr=l_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=5)

for e in range(num_epochs):
    
    print("Epoch: {}".format(e+1))
    
    total_train_loss = 0
    total_val_loss = 0
    total_train_error = 0
    total_val_error = 0
    num_train_iterations = 0
    num_val_iterations = 0

    model.train()
    with tqdm.tqdm(trainloader, unit="batch") as tepoch:
        for i,data in enumerate(tepoch,0):
    #         print("Training Iteration: {}".format(i+1))

            images,labels, path = data
            if torch.cuda.is_available():
                images = images.cuda()
                images = images.to(device)
                labels = labels.cuda()
                labels = labels.to(device)

            optimizer.zero_grad()
            out = model(images)
            pred = torch.sigmoid(out)

            # Check if the mask is truly binary
            test_label = labels.detach().cpu().numpy()
            num_not_binary = np.where(((test_label>0)&(test_label<1)|(test_label>1)),1,0).sum()

            # For calculating error
            pred_np = pred.detach()
            labels_np = labels.detach()

    #         masked_pred = np.where(pred_np>0.5,1,0) # Threshold prediction
            masked_pred = (pred_np>0.5).int()
            correct = torch.sum(torch.bitwise_and(masked_pred,labels_np.type(torch.int32))).item()
            incorrect = torch.sum(torch.bitwise_xor(masked_pred,labels_np.type(torch.int32))).item()
            if correct + incorrect == 0:
                print(f'{correct} and {incorrect}')
                print(out.shape, masked_pred.shape)
    #         Debugging -----------------------------------------------
            if (labels_np==1).sum().item() == 0:
                print((labels_np>0).sum().item())
                print(path)
                plt.figure()
                plt.subplot(1,2,1)
                plt.imshow(images.detach().cpu().numpy().squeeze().transpose(1,2,0))
                plt.subplot(1,2,2)
                plt.imshow(labels_np.numpy().squeeze())
#             print(labels)

            
            error = incorrect/(correct+incorrect)
            loss = criterion(out,labels) + dice_loss(masked_pred,labels)*math.exp(error)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_error += error
            num_train_iterations += 1

    train_error.append(total_train_error/num_train_iterations)
    train_loss.append(total_train_loss/num_train_iterations)
#     exp_lr_scheduler.step()

    print(f"Training Error: {train_error[-1]} | Training Loss: {train_loss[-1]} | Number of non-binary: {num_not_binary}")
    
    with torch.no_grad():
        model.eval()
        
        with tqdm.tqdm(validloader, unit="batch") as tepoch:
            for i,data in enumerate(tepoch,0):
    #             print("Validation Iteration: {}".format(i+1))

                images,labels, path = data
                if torch.cuda.is_available():
                    images = images.cuda(device)
                    labels = labels.cuda(device)

                out = model(images)
                pred = torch.sigmoid(out)

                # Check if the mask is truly binary
                test_label = labels.detach().cpu().numpy()
                num_not_binary = np.where(((test_label>0)&(test_label<1)|(test_label>1)),1,0).sum()

                # For calculating error
                pred_np = pred.detach()
                labels_np = labels.detach()

                masked_pred = (pred_np>0.5).int()

                correct = torch.sum(torch.bitwise_and(masked_pred,labels_np.type(torch.int32))).item()
                incorrect = torch.sum(torch.bitwise_xor(masked_pred,labels_np.type(torch.int32))).item()
                error = incorrect/(correct+incorrect)

                loss = criterion(out,labels) + dice_loss(masked_pred,labels)*math.exp(error)

                total_val_loss += loss.item()
                total_val_error += error
                num_val_iterations += 1
                
        val_error.append(total_val_error/num_val_iterations)
        val_loss.append(total_val_loss/num_val_iterations)
    
    scheduler.step(total_val_loss/num_val_iterations)
    
    lr_vals.append(optimizer.param_groups[0]['lr'])
    
    print(f"Validation Error: {val_error[-1]} | Validation Loss: {val_loss[-1]} | Number of non-binary: {num_not_binary}")

    
    if (val_loss[-1] < min_loss) and (e > 4) or e == (num_epochs - 1):
        print("Saved epoch {}".format(e+1))
        torch.save(model.state_dict(),f"./runs/{timeStarted}/{timeStarted}unet_3c_model_batch{batch}_sheduled_lr{config.l_rate}_epochs{e}.pt")
        min_loss = val_loss[-1]
    torch.save(model.state_dict(),f"./runs/{timeStarted}/{timeStarted}unet_3c_model_batch{batch}_sheduled_lr{config.l_rate}_last.pt")
    wandb.log({'train_err': train_error[-1], 'train_loss': train_loss[-1], 'val_err': val_error[-1], 'val_loss': val_loss[-1]})
    with open(f'./runs/{timeStarted}/training_info.txt', 'a') as f:
        f.write(f'{e} {train_error[-1]} {train_loss[-1]} {val_error[-1]} {val_loss[-1]}\n') # (epoch) (train error) (train loss) (val error) (val loss)
        
    epochs.append(e+1)

# f_train_accuracy = accuracy(unet,trainloader)
# # f_val_accuracy = accuracy(unet,validloader)
# # print("Final Training Error: {} | Final Validation Error: {}".format(f_train_accuracy,f_val_accuracy))
# print("Final Training Error: {}".format(f_train_accuracy))

# ------------------ Plot the training and validation curves -------------------
fig = plt.figure()
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)

ax1.plot(epochs,train_error,label="Training")
ax1.plot(epochs,val_error,label="Validation")
ax1.set_title("Model Error Curves")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Error")
ax1.legend()

ax2.plot(epochs,train_loss,label="Training")
ax2.plot(epochs,val_loss,label="Validation")
ax2.set_title("Model Loss Curves")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()

ax3.plot(epochs,lr_vals,label="Learning Rate")
ax3.set_title("Model LR")
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Learning Rate")
ax3.legend()

fig.tight_layout()
fig.savefig(f'./runs/{timeStarted}/results.png')
plt.show()

# torch.save(unet.state_dict(),"Some Path")

# Loading model weights
# unet.load_state_dict(torch.load("Some Path"))


# In[21]:

'''
if torch.cuda.is_available():
    unet.cuda()
unet.eval()
# unet.prob = 1

orig = cv2.imread("../input/unet-lanes-v3/Dataset 3/Synthetic/inputs/Synthetic_Lane_100.png")
test_im = np.copy(orig)

median = np.median(cv2.cvtColor(test_im,cv2.COLOR_BGR2GRAY))
lower = int(max(0,(1-0.205)*median))
upper = int(min(255,(1+0.205)*median))

test_edges = cv2.Canny(cv2.cvtColor(test_im,cv2.COLOR_BGR2GRAY),lower,upper)
test_edges_inv = cv2.bitwise_not(test_edges)

# grad_x = cv2.Sobel(cv2.cvtColor(test_im,cv2.COLOR_BGR2GRAY), cv2.CV_16S, 1, 0, ksize=1, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
# grad_y = cv2.Sobel(cv2.cvtColor(test_im,cv2.COLOR_BGR2GRAY), cv2.CV_16S, 0, 1, ksize=1, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

# abs_grad_x = cv2.convertScaleAbs(grad_x)
# abs_grad_y = cv2.convertScaleAbs(grad_y)
# grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

# test_im_edges = grad

test_im = np.append(test_im,test_edges.reshape(test_edges.shape[0],test_edges.shape[1],1),axis=2)
test_im = np.append(test_im,test_edges_inv.reshape(test_edges_inv.shape[0],test_edges_inv.shape[1],1),axis=2)

test_im = cv2.resize(test_im,(330,180))
ground_truth = cv2.imread("../input/unet-lanes-v3/Dataset 3/Synthetic/labels/Synthetic_Lane_100_Label.png",cv2.IMREAD_GRAYSCALE)
ground_truth = cv2.resize(ground_truth,(330,180))

# func = transforms.Compose([transforms.ToPILImage(),
#                                 transforms.Resize((180,330)),
#                                 transforms.ToTensor()])
# test_im = test_im/255
# test_img = func(test_im).unsqueeze(0).cuda()
test_img = (torch.Tensor((test_im/255.)).cuda().permute(2,0,1)).reshape(1,5,180,330)

# print((test_img == test_img2).sum().item())
# print(test_img.shape, test_img2.shape)
# print(test_img)
# print(test_img2)

output = unet(test_img)
pred = torch.sigmoid(output)
pred_np = pred.detach().cpu().numpy().squeeze()

pred_mask = np.where(pred_np>0.5,1,0)
# print((pred_mask==ground_truth).sum())
# print((pred_mask!=ground_truth).sum())
# print(pred_mask==ground_truth)
print(pred_np)
# print((pred_np>0.005).sum())
pred_mask = pred_mask*255
plt.figure()
plt.subplot(1,3,1)
plt.imshow(pred_mask.astype('uint8'))
# plt.imshow(grad)
plt.subplot(1,3,2)
plt.imshow(ground_truth.astype('uint8'))
plt.subplot(1,3,3)
plt.imshow(cv2.cvtColor(orig,cv2.COLOR_BGR2RGB))

'''
# In[ ]:


# # Add this part into the custom dataset class and have a check that determines if "carla" or smth is in the 
# # name in order to perform the necessary operations

# input_pth = "../input/unet-lanes-v3/Day Time/inputs/Lane_Input_2800.png"
# label_pth = "../input/unet-lanes-v3/Day Time/labels/Lane_Input_2800_Label.png"

# carla_im = cv2.imread(input_pth)
# carla_lbl = cv2.imread(label_pth)
# hls_im = cv2.cvtColor(carla_im,cv2.COLOR_BGR2HLS)
# print(carla_lbl.shape)

# dashed_mask = np.zeros((carla_lbl.shape[0],carla_lbl.shape[1]))
# line_mask = np.zeros((carla_lbl.shape[0],carla_lbl.shape[1]))
# edges_mask = np.zeros((carla_im.shape[0],carla_im.shape[1]),dtype=np.int)

# gray_im = cv2.cvtColor(carla_im,cv2.COLOR_BGR2GRAY)
# gray_lbl = cv2.cvtColor(carla_lbl,cv2.COLOR_BGR2GRAY)

# # gray_im = cv2.GaussianBlur(gray_im,(5,5),0)

# median = np.median(gray_im)
# lower = int(max(0,(1-0.205)*median))
# upper = int(min(255,(1+0.205)*median))

# edges = cv2.bitwise_not(cv2.Canny(gray_im,lower,upper))
# h_lines = cv2.HoughLinesP(edges,1,np.pi/180,100,50,20)

# # Quadrant division of input image
# width = carla_lbl.shape[1]
# height = carla_lbl.shape[0]

# med1 = np.median(gray_im[:height//2,:width//2])
# med2 = np.median(gray_im[:height//2,width//2:])
# med3 = np.median(gray_im[height//2:,width//2:])
# med4 = np.median(gray_im[height//2:,:width//2])

# l1 = int(max(0,(1-0.33)*med1))
# u1 = int(min(255,(1+0.33)*med1))
# e1 = cv2.Canny(gray_im[:height//2,:width//2],l1,u1)

# l2 = int(max(0,(1-0.33)*med2))
# u2 = int(min(255,(1+0.33)*med2))
# e2 = cv2.Canny(gray_im[:height//2,width//2:],l2,u2)

# l3 = int(max(0,(1-0.33)*med3))
# u3 = int(min(255,(1+0.33)*med3))
# e3 = cv2.Canny(gray_im[height//2:,width//2:],l3,u3)

# l4 = int(max(0,(1-0.33)*med4))
# u4 = int(min(255,(1+0.33)*med4))
# e4 = cv2.Canny(gray_im[height//2:,:width//2],l4,u4)

# # Stitch the edges together
# edges_mask[:height//2,:width//2] = e1
# edges_mask[:height//2,width//2:] = e2
# edges_mask[height//2:,width//2:] = e3
# edges_mask[height//2:,:width//2] = e4

# edges_mask = cv2.bitwise_not(edges_mask)

# print(len(h_lines))

# if len(h_lines) != 0:
#     for l in range(0,len(h_lines)):
#         line = h_lines[l][0]
#         cv2.line(line_mask,(line[0],line[1]),(line[2],line[3]),1,3,cv2.LINE_AA)
#     binary_im = line_mask.astype(np.int64)
#     binary_lbl = np.where(gray_lbl>0,1,0)
# else:
#     mean_value = np.mean(gray_im[(gray_lbl>0).nonzero()])
#     max_value = np.max(gray_im[(gray_lbl>0).nonzero()])
#     min_value = np.min(gray_im[(gray_lbl>0).nonzero()])

#     threshold = int(mean_value - 5)

#     binary_lbl = np.where(gray_lbl>0,1,0)
#     binary_im = np.where(gray_im>=threshold,1,0)

# dashed_mask = cv2.bitwise_and(binary_lbl,binary_im)

# test = np.bitwise_or(np.bitwise_and(gray_lbl,edges_mask),np.bitwise_and(gray_lbl,np.bitwise_not(edges_mask)))
# print(type(test)==np.ndarray)

# plt.figure()
# plt.subplot(1,3,1)
# plt.imshow(cv2.cvtColor(carla_im,cv2.COLOR_BGR2RGB))
# plt.subplot(1,3,2)
# # plt.imshow(255*cv2.cvtColor(carla_lbl,cv2.COLOR_BGR2GRAY))
# plt.imshow(np.bitwise_or(np.bitwise_and(gray_lbl,edges_mask),np.bitwise_and(gray_lbl,np.bitwise_not(edges_mask))))
# plt.subplot(1,3,3)
# plt.imshow(edges_mask)

# set_im = iter(trainloader)
# imgs,lbls = set_im.next()
# # print(imgs[0])
# # print((lbls[0]==1).sum())

