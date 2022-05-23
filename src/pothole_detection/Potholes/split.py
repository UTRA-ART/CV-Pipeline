import os
from pathlib import Path
from random import randint

TRAIN_NUM = 1635
VAL_NUM = 467
TEST_NUM = 233
TOTAL = 2335 - 1

count = 0

for num, type in [(TRAIN_NUM, "train"), (VAL_NUM, "valid"), (TEST_NUM, "test")]:
    for i in range(num):
        files = os.listdir("./dataset/data/")
        print(len(files))
        image_path = Path("./dataset/data/" + files[randint(0, TOTAL)])
        label_path = Path(
            "./dataset/label/" + os.path.splitext(image_path.name)[0] + ".txt"
        )

        out_img = f"./dataset/{type}/images/" + image_path.name
        out_lbl = f"./dataset/{type}/labels/" + label_path.name

        os.rename(image_path, out_img)
        os.rename(label_path, out_lbl)
        TOTAL = TOTAL - 1
