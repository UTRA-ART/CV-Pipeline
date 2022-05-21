import os
from pathlib import Path
from random import randint

NUM_TOTAL_IMGS = 4627
NUMF_TRAIN_IMGS = int(0.7 * NUM_TOTAL_IMGS)  # 3239
NUM_VAL_IMGS = int(0.2 * NUM_TOTAL_IMGS)  # 925
NUM_TEST_IMGS = NUM_TOTAL_IMGS - NUMF_TRAIN_IMGS - NUM_VAL_IMGS  # 463

count = 0

for num, type in [
    (NUMF_TRAIN_IMGS, "train"),
    (NUM_VAL_IMGS, "valid"),
    (NUM_TEST_IMGS, "test"),
]:
    for i in range(num):
        files = os.listdir("./dataset/data/")
        print(len(files))
        image_path = Path("./dataset/data/" + files[randint(0, NUM_TOTAL_IMGS - 1)])
        label_path = Path(
            "./dataset/label/" + os.path.splitext(image_path.name)[0] + ".txt"
        )

        out_img = f"./dataset/{type}/images/" + image_path.name
        out_lbl = f"./dataset/{type}/labels/" + label_path.name

        os.rename(image_path, out_img)
        os.rename(label_path, out_lbl)
        NUM_TOTAL_IMGS = NUM_TOTAL_IMGS - 1
