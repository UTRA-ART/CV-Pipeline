from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import random
import os
from pathlib import Path
import math

DATA_PATH = "./data/"
LABEL_PATH = "./processed/label/"
DATA_OUT = "./processed/data/"

# To find asphalt
RADIUS_Y = 50
RADIUS_X = 100

GREY_BOUNDS = (15, 180)

# Check if the chosen points are on pavement


def is_valid(x, y, image):
    pixels = np.array(image)
    rgb = np.zeros(3)

    # check if the mean color is grey
    for i in range(x - RADIUS_X, x + RADIUS_X):
        for j in range(y - RADIUS_Y, y + RADIUS_Y):
            rgb += pixels[j][i]
    mean = rgb / (4 * RADIUS_Y * RADIUS_X)

    # Best rgb values that represent grey:
    if all((GREY_BOUNDS[0] <= num <= GREY_BOUNDS[1]) for num in mean):
        if np.amax(mean) - np.amin(mean) < 30:
            return mean

    return None


# Salt pepper noise


def add_noise(img):
    pixels = np.array(img)
    num = random.randint(30000, 100000)
    for i in range(num):
        x = random.randint(0, img.size[0] - 1)
        y = random.randint(0, img.size[1] - 1)
        pixels[y][x] = random.randint(0, 255)

    return Image.fromarray(pixels)


def add_patches(img, x_rad, y_rad, x1, y1):
    width, height = img.size

    mns = [
        [x1 + random.randint(0, 2 * x_rad), y1 + random.randint(0, 2 * y_rad)],
        [x1 + random.randint(0, 2 * x_rad), y1 + random.randint(0, 2 * y_rad)],
    ]
    sns = [
        min(x_rad, y_rad) * (random.randint(90, 110) / 100.0),
        min(x_rad, y_rad) * (random.randint(90, 110) / 100.0),
    ]
    # print(
    # f'radius1 is {sns[0]} and raius2 is {sns[1]} where xrad is {x_rad} and yrad is {y_rad}')
    k = 2
    g = random.randint(1, 3)
    r = 1.8

    for x in range(width):
        for y in range(height):
            total = 0

            for b in range(k):
                try:
                    val = sns[b] / (
                        (math.sqrt((mns[b][0] - x) ** 2 + (mns[b][1] - y) ** 2)) ** g
                    )
                except ZeroDivisionError:
                    val = r * 2

                total += val

            if total > r:
                # print(x, y)
                img.putpixel((x, y), random.randint(25, 50))

    return img


def draw_ellipse(input_path, max_potholes):
    image = Image.open(input_path)
    count = 0
    max_ = random.randint(2, max_potholes)
    potholes = []
    for i in range(64):
        if count >= max_:
            break
        loc_x = random.randint(0.2 * 1280, 0.8 * 1280)
        loc_y = random.randint((int)(0.7 * 720), (int)(0.9 * 720))
        overlap = False
        for pothole in potholes:
            if abs(pothole[0] - loc_x) < 160 and abs(pothole[1] - loc_y) < 80:
                overlap = True
                break
        if overlap:
            continue
        mean = is_valid(loc_x, loc_y, image)
        if mean is not None:
            mean = np.amax(mean)
            x_rad, y_rad = random.randint(50, RADIUS_X), random.randint(10, 40)
            x_start, x_end, y_start, y_end = (
                loc_x - x_rad,
                loc_x + x_rad,
                loc_y - y_rad,
                loc_y + y_rad,
            )
            # Mask a noisy image in the shape of a pothole
            rgb = min(int(130 + mean), 255)
            # Different type of potholes
            select = random.randint(1, 5)
            background = Image.new("L", image.size, rgb)
            obj_class = 0
            if select == 1:
                background = add_patches(background, x_rad, y_rad, x_start, y_start)
            elif select > 3:
                background = add_noise(background)
                obj_class = 0
            background = background.filter(
                ImageFilter.GaussianBlur(random.randint(50, 125) / 100)
            )
            # Scale down image to get a choppier ellipse
            scale = random.randint(30, 60) / 100
            mask = Image.new(
                "L", ((int)(image.width * scale), (int)(image.height * scale)), 0
            )
            ImageDraw.Draw(mask).ellipse(
                (
                    (int)(x_start * scale),
                    (int)(y_start * scale),
                    (int)(x_end * scale),
                    (int)(y_end * scale),
                ),
                fill=255,
            )
            mask = mask.resize((image.width, image.height))

            mask = mask.filter(ImageFilter.GaussianBlur(random.randint(1, 4)))
            image = Image.composite(background, image, mask)
            potholes.append([loc_x, loc_y, x_rad * 2, y_rad * 2, obj_class])
            count += 1
    if len(potholes) == 0:
        return
    image.save(os.path.join(DATA_OUT, Path(input_path).name))
    height, width, rgb = np.array(image).shape
    # Normalize potholes locations
    potholes = [
        [
            (float)(x[0]) / width,
            (float)(x[1]) / height,
            (float)(x[2]) / width,
            (float)(x[3]) / height,
            x[4],
        ]
        for x in potholes
    ]
    # Save the label
    filename = os.path.splitext(Path(input_path).name)[0] + ".txt"
    file = open(os.path.join(LABEL_PATH, filename), "w")
    for i in range(len(potholes)):
        file.write(
            f"{potholes[i][4]} {potholes[i][0]} {potholes[i][1]}"
            + f" {potholes[i][2]} {potholes[i][3]}\n"
        )
    file.close()


for folder in os.listdir(DATA_PATH):
    for file in os.listdir(DATA_PATH + f"/{folder}"):
        draw_ellipse(DATA_PATH + f"{folder}/{file}", 4)
        print(f"Done: {file}")

    # # Add Gaussian Blur surrounding the pothole
    # blurred_img = image.filter(
    #     ImageFilter.GaussianBlur(random.randint(2, 6)))
    # blur_mask = Image.new('L', image.size, 0)
    # # Extra x and Extra y so that the blur extends into the pothole
    # ex, ey = int((x_end - x_start) * 0.1),\
    #     int((y_end - y_start) * 0.1)
    # ImageDraw.Draw(blur_mask).rectangle(
    #     (x_start - ex, y_start - ey, x_end + ex, y_end + ey), fill=255)
    # ImageDraw.Draw(blur_mask).ellipse(
    #     (x_start, y_start, x_end, y_end), fill=random.randint(20, 80))
    # image = Image.composite(blurred_img, image, blur_mask)
    # Save the positions
