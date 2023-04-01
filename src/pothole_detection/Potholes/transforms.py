import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import transforms, functional
from PIL import Image
import math
import os
import numpy as np
import random
import cv2
from torch.utils.data import ConcatDataset

class CircleDataset(Dataset):
    def __init__(self, image_paths, black_image_paths, transform=None):
        self.image_paths = image_paths
        self.black_image_paths = black_image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # open image and label
        image = Image.open(self.image_paths[idx]).convert('RGB')
        black_image = Image.open(self.black_image_paths[idx]).convert('RGB')

        # apply transform to image and transform label
        if self.transform:
            image, black_image = self.transform([image, black_image])

        # return transformed image and label
        return image, black_image

import glob

# define the folder path and file extension
folder_path = '/Users/leon.lee.21/Desktop/CV-Pipeline/src/pothole_detection/Potholes/processed/data/images/'
black_folder_path = '/Users/leon.lee.21/Desktop/CV-Pipeline/src/pothole_detection/Potholes/processed/data/black/'
file_extension = '*.jpg'

# get a list of file paths
image_paths = glob.glob(f"{folder_path}/{file_extension}")
black_image_paths = glob.glob(f"{black_folder_path}/{file_extension}")

class customTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, images):
        angle = random.uniform(-45, 45)
        image, black_image = images
        rotation = transforms.RandomRotation((angle, angle), resample=Image.BILINEAR)
        h_flip_rand = random.randint(0, 1)
        horizontalFlip = transforms.RandomHorizontalFlip(p=h_flip_rand)
        v_flip_rand = random.randint(0, 1)
        verticalFlip = transforms.RandomVerticalFlip(p=v_flip_rand)

        image = rotation(image)
        black_image = rotation(black_image)

        image = horizontalFlip(image)
        black_image = horizontalFlip(black_image)

        image = verticalFlip(image)
        black_image = verticalFlip(black_image)

        toTensorTransform = transforms.ToTensor()
        image = toTensorTransform(image)
        black_image = toTensorTransform(black_image)
        return image, black_image

transform = customTransform()

# # create the dataset
# circle_dataset = CircleDataset(image_paths, black_image_paths, transform=transform)

circle_dataset = None  # initialize empty dataset
for i in range(5):  # loop over the number of times to run CircleDataset
    dataset = CircleDataset(image_paths, black_image_paths, transform=transform)
    if circle_dataset is None:
        circle_dataset = dataset  # initialize circle_dataset with the first dataset
    else:
        circle_dataset = ConcatDataset([circle_dataset, dataset])  # concatenate datasets

# now circle_dataset contains all the data from running CircleDataset multiple times

# create the dataloader
batch_size = 100000
circle_dataloader = torch.utils.data.DataLoader(circle_dataset, batch_size=batch_size, shuffle=False)


output_dir = 'Potholes/processed/data/t_images'
black_output_dir = 'Potholes/processed/data/t_black'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(black_output_dir, exist_ok=True)

increment = 0
#iterate over the data
for i, (images, black_images) in enumerate(circle_dataloader):
    print(type(images))
    print(f'i = {i}')
    # save the images and labels to files
    for j in range(images.size(0)):
        print(f'j = {j}')
        if j + increment < 10:
            image_path = os.path.join(output_dir, f'{str(i * batch_size).zfill(4)}{increment + j}_T.jpg')
            black_image_path = os.path.join(black_output_dir, f'B_{str(i * batch_size).zfill(4)}{increment + j}_T.jpg')
        elif j + increment < 100:
            image_path = os.path.join(output_dir, f'{str(i * batch_size).zfill(3)}{increment + j}_T.jpg')
            black_image_path = os.path.join(black_output_dir, f'B_{str(i * batch_size).zfill(3)}{increment + j}_T.jpg')
        elif j + increment < 1000:
            image_path = os.path.join(output_dir, f'{str(i * batch_size).zfill(2)}{increment + j}_T.jpg')
            black_image_path = os.path.join(black_output_dir, f'B_{str(i * batch_size).zfill(2)}{increment + j}_T.jpg')
        elif j + increment < 10000:
            image_path = os.path.join(output_dir, f'{str(i * batch_size).zfill(1)}{increment + j}_T.jpg')
            black_image_path = os.path.join(black_output_dir, f'B_{str(i * batch_size).zfill(1)}{increment + j}_T.jpg')
        elif j + increment < 100000:
            image_path = os.path.join(output_dir, f'{str(i * batch_size).zfill(0)}{increment + j}_T.jpg')
            black_image_path = os.path.join(black_output_dir, f'B_{str(i * batch_size).zfill(0)}{increment + j}_T.jpg')
        # save the image file
        transforms.ToPILImage()(images[j]).save(image_path)
        transforms.ToPILImage()(black_images[j]).save(black_image_path)
    # do something with the images and labels
    pass

