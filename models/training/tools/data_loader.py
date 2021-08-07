'''
Handles downloading and importing data for training models
'''

import pathlib
import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np

#------------------------------------ Dataset Loader for Classification  ------------------------------------#
'''
Args: 
  path -  The path to the dataset file that contains a singular file for each class
  validation_split - the decimal percentage of data that is put into the validation set
  batch_size
  img_height
  img_width
  seed - random seed to be used in validation/training split
Returns: 
  Two tensorflow datasets with elements in the form of (images, labels)
  Images is shape (batch_size, img_width, img_height, channels)
  train_ds: training dataset
  val_ds: validation dataset
'''
def classificationDatasetLoader(path, batch_size, img_height, img_width, seed=3, validation_split=0.2):

    path = pathlib.Path(path)

    # Get the training set
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        validation_split=validation_split,
        subset="training", 
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # Get the validaiton set
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        validation_split=validation_split,
        subset="validation", 
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    return train_ds, val_ds




#------------------------------------ Test ------------------------------------#
if __name__ == "__main__":

    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)

    train_ds, val_ds = classificationDatasetLoader(data_dir, 24, 180, 180)

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

    view_image(train_ds)
