from PIL import Image
import math

img = Image.open(
    "C:\\Users\\ammar\\Documents\\CodingProjects\\ART\\CV-Pipeline\\YOLOv4\\PyTorch_YOLOv4-train\\inference\\Lane_Input_2076.png"
)

width, height = img.size

mns = [[100, 100], [50, 50]]
sns = [100, 20]
k = 2
g = 1
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
            img.putpixel((x, y), (0, 0, 0))

img.save("new.png")
