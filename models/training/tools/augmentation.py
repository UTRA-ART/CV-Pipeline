'''
Contains data augmentation tools for training models. 
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from functools import partial
from albumentations import (
    Compose, RandomBrightness, JpegCompression, HueSaturationValue, RandomContrast, HorizontalFlip,
    Rotate
)

#-------------------------------------- Get Augmented Set Function --------------------------------------#
def get_aug(dataset, img_height, img_width, transforms):

    #-------------------------------------- Helper Functions --------------------------------------#
    def aug_fn(image, img_height, img_width):
        data = {"image":image}
        aug_data = transforms(**data)
        aug_img = aug_data["image"]
        aug_img = tf.cast(aug_img/255.0, tf.float32)
        aug_img = tf.image.resize(aug_img, size=[img_height, img_width])
        return aug_img

    def process_data(image, label, img_height, img_width):
        aug_img = tf.numpy_function(func=aug_fn, inp=[image, img_height, img_width], Tout=tf.float32)
        return aug_img, label

    def set_shapes(img, label, img_shape=(120,120,3)):
        img.set_shape(img_shape)
        label.set_shape([])
        return img, label

    #-------------------------------------- Main Logic --------------------------------------#
    aug_data = dataset.map(partial(process_data, img_height=img_height, img_width=img_width))
    aug_data = aug_data.map(partial(set_shapes, img_shape=(img_height, img_width,3)))
    return aug_data


#-------------------------------------- Test Code --------------------------------------#
if __name__ == "__main__":

    def view_image(ds):
        image, label = next(iter(ds)) # extract 1 batch from the dataset
        image = image.numpy()
        label = label.numpy()

        fig = plt.figure(figsize=(22, 22))
        for i in range(20):
            ax = fig.add_subplot(4, 5, i+1, xticks=[], yticks=[])
            ax.imshow(image[i])
            ax.set_title(f"Label: {label[i]}")

        plt.show()
        
    # load in the tf_flowers dataset
    data, info= tfds.load(name="tf_flowers", split="train", as_supervised=True, with_info=True)

    # Create Transforms
    transforms = Compose([
                Rotate(limit=40),
                RandomBrightness(limit=0.1),
                JpegCompression(quality_lower=85, quality_upper=100, p=0.5),
                HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                RandomContrast(limit=0.2, p=0.5),
                HorizontalFlip(),
            ])

    # Get augmented set
    aug_data = get_aug(data, 120, 120, transforms)

    view_image(aug_data.batch(32))