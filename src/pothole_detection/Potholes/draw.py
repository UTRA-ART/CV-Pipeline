import cv2

# Load the image
#image = cv2.imread("Potholes/transformed/62transform.jpg")
image = cv2.imread("Potholes/processed/data/images/201.jpg")
#image = cv2.imread("Potholes/processed/data/images/62.jpg")

# Define the coordinates
coords = [0.62578125, 0.18888888888888888, 0.2078125, 0.33611111111111114]
# Extract the coordinates and convert them to integers
# y, x, w, h = [int(c * image.shape[i%2]) for i, c in enumerate(coordinates)]

x = coords[0] * 1280
y = coords[1] * 720
w = coords[2] * 1280
h = coords[3] * 720


print((x-(w/2), y-(h/2)))
print((x + (w/2), y + (h/2)))


# Draw the rectangle
cv2.rectangle(image, ((int)(x-(w/2)), (int)(y-(h/2))), ((int)(x + (w/2)), (int)(y + (h/2))), (0, 255, 0), 2)

# Show the image
cv2.imshow("Image with Rectangle", image)
cv2.waitKey(0)