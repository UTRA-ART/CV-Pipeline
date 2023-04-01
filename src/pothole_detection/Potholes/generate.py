from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from cv2 import rotate
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms.functional import rotate
import random
import os
from pathlib import Path
import math
import cv2
import torch

#DATA_PATH = "/Users/leon.lee.21/Desktop/CV-Pipeline/src/pothole_detection/data/"
DATA_PATH = "/Users/leon.lee.21/Desktop/CV-Pipeline/src/pothole_detection/data/"
LABEL_PATH = "/Users/leon.lee.21/Desktop/CV-Pipeline/src/pothole_detection/Potholes/processed/data/labels/"
# DATA_OUT = "/pothole_detection/Potholes/processed/data/"
POINTS_PATH = "/Users/leon.lee.21/Desktop/CV-Pipeline/src/pothole_detection/Potholes/processed/data/points/"
DATA_OUT = "/Users/leon.lee.21/Desktop/CV-Pipeline/src/pothole_detection/Potholes/processed/data/images/"
BLACK_OUT = "/Users/leon.lee.21/Desktop/CV-Pipeline/src/pothole_detection/Potholes/processed/data/black/"

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


#////////////////////////////// ////                 1
def get_poly(p1, p2, width, height, orient=0): #  4  🔲  2
    if(orient==1):                             #     3
        return [(0,0), (width, 0), (width, p2),(0, p1)]
    if(orient==2):
        return [(width,0), (width, height), (p2, height),(p1, 0)]
    if(orient==3):
        return [(0,p1), (width, p2), (width, height),(0, height)]
    if(orient==4):
        return [(0,0), (p1, 0), (p2, height),(0, height)]

def get_rand_poly(width, height, ellipse):
    x_start, y_start, x_end, y_end = ellipse
    orient = random.randint(1,4)
    try:
        if(orient%2==1):                             
            low1 = max((int)((height - (height-y_start)*width/(width-x_start)*0.7)), (int)(height*0.3))
            high1 = min((int)(y_end*width/(width-x_start)*1.3), (int)(height*0.7))
            p1 = random.randint(low1, high1) if (low1 < high1) else random.randint(high1, low1) 
            #h = y_end if (orient==1) else (height-y_end)
            low2 = max((int)((p1 - (p1-x_start)*height/y_end)), height)
            high2 = min((int)((p1 - (p1-x_end)*height/y_end)), height)
            #low2 = max((int)(height - (height-y_start)*width/x_end), 0)
            #high2 = min((int)(y_end*width/x_end), height)
            #print(ellipse, '////', low1, high1, p1, low2, high2)
            p2 = random.randint(low2, high2) if (low2 < high2) else random.randint(high2, low2) 
        else:
            low1 = max((int)((width - (width-x_start)*height/(height-y_start))*0.7), (int)(width*0.3))
            high1 = min((int)(x_end*height/(height-y_start)*1.3), (int)(width*0.7))
            p1 = random.randint(low1, high1) if (low1 < high1) else random.randint(high1, low1) 
            #h = x_end if (orient==1) else (width-x_end)
            low2 = max((int)(p1 - (p1-y_start)*width/x_end), 0)
            high2 = min((int)(p1 - (p1-y_end)*width/x_end), width)
            #low2 = max((int)(width - (width-x_start)*height/y_end), 0)
            #high2 = min((int)(x_end*height/y_end), width)
            #print(ellipse, '////', low1, high1, p1, low2, high2)
            p2 = random.randint(low2, high2) if (low2 < high2) else random.randint(high2, low2) 
        
        return get_poly(p1, p2, width, height, orient)
    except: return [(0,0), (0,0), (0,0), (0,0)]
    
def add_shade(image, poly=None):
    image = image.convert("RGBA")
    shade = Image.new("RGBA", image.size, (0,0,0,0))
    d = ImageDraw.Draw(shade)
    # poly = [(shade_x1, 0), (shade_x2, image.height),(0, image.height), (0, 0)]
    d.polygon(poly, fill=(10,10,10,80), outline=None)
    image = Image.alpha_composite(image, shade)
    return image.convert("RGB")
#/////////////////////////////////////////

# Salt pepper noise


def add_noise(img):
    pixels = np.array(img)
    num = random.randint(10000, 30000)
    for i in range(num):
        x = random.randint(0, img.size[0] - 100)
        y = random.randint(0, img.size[1] - 100)
        pixels[y][x] = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )

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
                pixel = (
                    random.randint(0, 100),
                    random.randint(0, 100),
                    random.randint(0, 100),
                )
                img.putpixel((x, y), random.randint(25, 75))

    return img


def draw_ellipse(input_path, max_potholes):
    image = Image.open(input_path)
    image = image.resize((1280, 720))
    black_image = Image.new('RGB', image.size, color='black')
    #image = ImageEnhance.Brightness(image).enhance(0.25)
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
            if abs(pothole[0] - loc_x) < 500 and abs(pothole[1] - loc_y) < 500:
                overlap = True
                break
        if overlap:
            continue
        image = image.convert('RGB')
        mean = is_valid(loc_x, loc_y, image)
        
        if mean is not None:
            mean = np.amax(mean)
            x_rad = random.randint(120, 140)
            y_rad = random.randint(x_rad-15, x_rad)
            loc_y = random.randint((int)(0.1 * 720), (int)(0.9 * 720))
            #x_rad, y_rad = random.randint(50, RADIUS_X), random.randint(10, 40)
            x_start, x_end, y_start, y_end = (
                loc_x - x_rad,
                loc_x + x_rad,
                loc_y - y_rad,
                loc_y + y_rad,
            )
            # Mask a noisy image in the shape of a pothole
            rgb = random.randint(200, 255)
            rgb = (rgb,) * 3
            # Different type of potholes
            select = random.randint(1, 5)
            background = Image.new("RGB", image.size, rgb)
            obj_class = 0
        # if select == 1:
                #background = add_patches(background, x_rad, y_rad, x_start, y_start)
            if select > 3:
                background = add_noise(background)
                obj_class = 0
            background = background.filter(
                ImageFilter.GaussianBlur(random.randint(50, 125) / 100)
            )
            # Scale down image to get a choppier ellipse
            # scale = random.randint(30, 60) / 100
            # mask = Image.new("L", ((int)(image.width * scale),
            #                  (int)(image.height * scale)), 0)
            # ImageDraw.Draw(mask).ellipse(
            #     ((int)(x_start * scale), (int)(y_start * scale),
            #      (int)(x_end * scale), (int)(y_end * scale)), fill=255)
            # mask = mask.resize((image.width, image.height))

            # mask = mask.filter(ImageFilter.GaussianBlur(random.randint(1, 4)))
            mask = Image.new("L", image.size, 0)

            #///////////////////////////////////////////////////

            # Draw an ellipse on the image
            draw = ImageDraw.Draw(mask)
            #draw.ellipse((x_start, y_start, x_end, y_end), outline='white')

            # Get points on the perimeter of the ellipse
            n_points = 100
            points = []
            for i in range(n_points):
                angle = i * 2 * math.pi / n_points
                x = (x_start + x_end) / 2 + (y_end - y_start) / 2 * math.cos(angle)
                y = (y_start + y_end) / 2 + (y_end - y_start) / 2 * math.sin(angle)
                points.append((x, y))

            # Draw the polygon using the points
            draw.polygon(points, outline='white', fill = 'white')

            draw = ImageDraw.Draw(black_image)

            draw.polygon(points, outline='black', fill = 'white')

            #////////////////////////////////////////////////////

           # ImageDraw.Draw(mask).ellipse((x_start, y_start, x_end, y_end), fill=255)
            image = Image.composite(background, image, mask)
            #/////////////////////////////
            ellipse = (x_start, y_start, x_end, y_end)
            poly = get_rand_poly(image.width, image.height, ellipse)
            image = add_shade(image, poly)
            #///////////////////////////////
            potholes.append([loc_x, loc_y, x_rad * 2, y_rad * 2, obj_class])
            count += 1
    if len(potholes) == 0:
        return
    open
    image.save(
        DATA_OUT + os.path.splitext(Path(input_path).name)[0]  + ".jpg"
    )
    black_image.save(
        BLACK_OUT + os.path.splitext(Path(input_path).name)[0]  + ".jpg"
    )
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
    # filename = os.path.splitext(Path(input_path).name)[0] + "D.txt"
    #filename = folder + file + "D.txt"
    filename = LABEL_PATH + os.path.splitext(Path(input_path).name)[0] + ".txt"
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

