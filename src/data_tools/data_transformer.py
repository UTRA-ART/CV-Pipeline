import re
from cv2 import getRotationMatrix2D, spatialGradient, warpAffine
import numpy as np
import cv2
import math
from PIL import Image
import random
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

# path 
path = "/Users/andy/Documents/Data_Transformer/newdim.jpg"

# output window name
window_name = "Output Frame"

# image dimensions
HEIGHT = 720
WIDTH = 1280
  
def initialize(path):
    '''
    Opens an image from specified file path. 
    
    NOTES
    ----
    image[x,y] --> x is row, y is col.
    '''
    return cv2.imread(path)

def display_image(image, window):
    '''
    Prints input image to window and adds a waitkey.

    INPUTS
    ------
    Image: Input image. Make sure it has dimensions of HEIGHT and WIDTH.
    Window: Name of the output window.
    '''
    cv2.imshow(window, image) 
    cv2.waitKey(0)

def make_black():
    '''
    Creates a black image of dimensions WIDTH x HEIGHT.
    '''
    return np.zeros(shape=[HEIGHT, WIDTH, 3], dtype=np.uint8)

def resize_image(image, hor_factor, ver_factor, centerized=1, scaling_method="factor"):
    '''
    Resizes an image and overlays it onto a black image.

    INPUTS
    ------
    image: Input image of dimensions WIDTH x HEIGHT.
    hor_factor: horizontal scaling factor to be multiplied to the WIDTH of the input.
        float in [0,1]
    ver_factor: vertical scaling factor to be multiplied to the HEIGHT of the input.
        float in [0,1]
    centerized: controls the position of the resulting image
        default makes the resized image overlayed onto the center of a black image. 
        otherwise, the image is overlayed to the top-left-corner.
    scaling_method: controls how the image is resized 
        default makes hor_factor and ver_factor scaling factors. 
        otherwise, hor_factor and ver_factor are the bottom-right coordinates of the resized image.
    
    NOTES
    -----
    Must be appied to labels as well.
    '''
    if (scaling_method == "factor"):
        x = round(WIDTH * hor_factor)
        y = round(HEIGHT * ver_factor)
    else: 
        x = hor_factor
        y = ver_factor
    new_dim = (x,y)
    cpy = np.copy(image)
    resized = cv2.resize(cpy, new_dim, interpolation=cv2.INTER_LINEAR)
    ret = make_black()
    rh, rw, channels = resized.shape

    if (centerized == 1):
        ret[round((HEIGHT - y)/2):(round((HEIGHT - y)/2) + rh),round((WIDTH - x)/2):(round((WIDTH - x)/2) + rw)] = resized
    else:
        ret[0:rh,0:rw] = resized

    return ret

def reflect_image(image, axis):
    '''
    Reflects input image along given axis.
    
    INPUTS
    ------
    image: Input image of dimensions WIDTH x HEIGHT.
    axis: determines which axis the image is flipped along. 
        0: flip over y-axis
        1: flip over x-axis
        -1: flip over both axes
    
    NOTES
    -----
    Must be applied to labels as well.
    '''
    ret = np.copy(image)
    return cv2.flip(ret,axis)
 
def rotate_image(image, deg):
    '''
    Rotates image and resizes if necessary to preserve data from original image.
    
    INPUTS
    ------
    image: Input image of dimension WIDTH x HEIGHT.
    deg: Rotation angle in degrees
        must be between 180 and -180 degrees
    
    NOTES:
    Must be applied to labels as well.
    '''
    
    val = abs(deg)
    if (val > 90):
        val = 180 - val

    hypo = math.sqrt(math.pow(HEIGHT/2, 2) + math.pow(WIDTH/2, 2))
    init_angle = math.atan(HEIGHT/WIDTH)
    factor = HEIGHT/(2*hypo*math.sin(init_angle + math.radians(val)))
    image = resize_image(image, factor, factor)
    
    rows, cols, dim = image.shape
    M = getRotationMatrix2D(center = (round(cols/2), round(rows/2)), angle=deg, scale=1)
    ret = np.copy(image)
    return cv2.warpAffine(ret, M, (int(cols),int(rows)))

def shear_image(image, sh_x, sh_y):
    '''
    Shears input image along x and y axes by specified amounts.

    INPUTS
    ------
    image: Input image of dimension WIDTH x HEIGHT.
    sh_x: shearing factor in x-axis.
        must be less than ~ WIDTH / HEIGHT to stay within image frame
        float in [0,WIDTH / HEIGHT)
    sh_y: shearing factor in y-axis.
        must be less than ~ HEIGHT / WIDTH to stay within image frame
        float in [0,HEIGHT / WIDTH)
    
    NOTES
    -----
    Must be applied to labels as well.
    '''
    M = np.float32([[1, sh_x, 0],
                    [sh_y, 1, 0],
                    [0, 0, 1]])
    inv = np.linalg.inv(M)
    col = np.float32([[1280], [720], [1]])
    res = np.dot(inv,col)

    new_size = resize_image(image, int(math.floor(res[0][0])), int(math.floor(res[1][0])), center=0, method="coordinates")
    display_image(new_size, window_name)
    sheared_img = cv2.warpPerspective(new_size,M,(int(HEIGHT),int(WIDTH)))

    return sheared_img

def apply_gaussian(image, kernel_x=10, kernel_y=10):
    '''
    Applies a gaussian blur to the input image.
    
    INPUTS
    ------
    image: Input image of dimension WIDTH x HEIGHT.
    kernel_x and kernel_y: Determine the blurring. 1 is added to both after they are multipled by 2 to ensure the input into the cv2.GaussianBlur function is odd
        positive integers
        default values for both are 10.
    
    NOTES
    -----
    Does not affect labels.
    '''
    ret = np.copy(image)
    return cv2.GaussianBlur(ret, (2*kernel_x + 1, 2*kernel_y + 1), 0)

def colour(image, bf=1, gf=1, rf=1, mode="RGB"):
    '''
    Modifies colour channels of input image.

    INPUTS
    ------
    image: Input image of dimensions WIDTH x HEIGHT.
    bf: blue-factor to be multiplied to the blue channel of the image.
        bf = 0 will remove all blue from the image
        if bf increases the blue beyond 255, it gets capped at 255
    gf: green-factor to be multiplied to the green channel of the image.
        gf = 0 will remove all green from the image
        if gf increases the green beyond 255, it gets capped at 255  
    rf: red-factor to be multiplied to the red channel of the image.
        rf = 0 will remove all red from the image
        if rf increases the red beyond 255, it gets capped at 255  
    mode: indicates if you want to modify the RGB channels of your image.
        default modifies RGB channels
        otherwise, applies grayscale to the image

    NOTES
    -----
    Does not affect labels.

    '''
    if (mode != "RGB"):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        ret = np.copy(image)
        if (bf <= 1):
            ret[:,:,0] = (ret[:,:,0]*bf).astype(int)
        else:
            blue = np.copy(image)
            blue[:,:,0] = 255
            ret[:,:,0] = ((ret[:,:,0] + 2*blue[:,:,0])/3).astype(int)
        if (gf <= 1):
            ret[:,:,1] = (ret[:,:,1]*gf).astype(int)
        else: 
            green = np.copy(image)
            green[:,:,1] = 255
            ret[:,:,1] = ((ret[:,:,1] + 2*green[:,:,1])/3).astype(int)
        if (rf <= 1):
            ret[:,:,2] = (ret[:,:,2]*rf).astype(int)
        else:
            red = np.copy(image)
            red[:,:,2] = 255
            ret[:,:,2] = ((ret[:,:,2] + 2*red[:,:,2])/3).astype(int)
        
        return ret

def pixel_swap(image, seed, swaps):
    '''
    Randomly swaps pixels of an image to create noise.

    INPUTS
    ------
    image: Input image of dimensions WIDTH x HEIGHT.
    seed: Allows you to repreduce results. If called with the same seed twice, the output will be the same.
        any integer
    swaps: determines the number of times pixels are swapped.
        any integer
    
    NOTES
    -----
    WARNING VERY SLOW
    Must be applied to labels as well.
    '''
    height, width, channels = image.shape
    ret = np.copy(image)
    for num in range(0, swaps):
        random.seed(seed + num)
        a = random.randint(0, height - 1)
        random.seed(seed + num + swaps)
        b = random.randint(0, width - 1)
        random.seed(seed + num + 2*swaps)
        c = random.randint(0, height - 1)
        random.seed(seed + num + 3*swaps)
        d = random.randint(0, width - 1)
        
        x, y, z = ret[a,b]
        ret[a,b] = ret[c,d]
        ret[c,d] = [x,y,z]

    return ret

def apply_mosaic(image, seed, swaps):
    '''
    Randomly swaps sections of an image. Sections are rectangles of random dimensions
    
    INPUTS
    ------
    image: Input image of dimensions WIDTH x HEIGHT.
    seed: Allows you to repreduce results. If called with the same seed twice, the output will be the same.
        any integer
    swaps: determines the number of times pixels are swapped.
        any integer

    NOTES
    -----
    Must be applied to labels as well.
    '''
    height, width, channels = image.shape
    ret = np.copy(image)
    cpy = np.copy(image)

    for k in range(swaps):
        random.seed(seed*k)
        box_height = random.randint(50,200)
        random.seed(seed*k + 50)
        box_width = random.randint(50,200)
        random.seed(seed*k + 100)
        a = random.randint(0, HEIGHT - box_height - 1)
        random.seed(seed*k + 200)
        b = random.randint(0, WIDTH - box_width - 1)
        random.seed(seed*k + 300)
        c = random.randint(0, HEIGHT - box_height - 1)
        random.seed(seed*k + 200)
        d = random.randint(0, WIDTH - box_width - 1)
        random.seed(seed*k + 300)
        
        cpy[a:(a + box_height), b:(b + box_width)] = ret[a:(a + box_height), b:(b + box_width)]
        ret[a:(a + box_height), b:(b + box_width)] = ret[c:(c + box_height), d:(d + box_width)]
        ret[c:(c + box_height), d:(d + box_width)] = cpy[a:(a + box_height), b:(b + box_width)]

    return ret

def uniform_mosaic(image, seed, v_box, h_box):
    '''
    Splits image into a grid of x_box by y_box equal rectangles. Randomly swaps rectangles.
    
    INPUTS
    ------
    image: Input image of dimensions WIDTH x HEIGHT.
    seed: Allows you to repreduce results. If called with the same seed twice, the output will be the same.
        any integer
    v_box: determines the number of boxes along the vertical axis.
        any positive integer less than 721
    h_box: determines the number of boxes along the horizontal axis.
        any positive integer less than 1281

    NOTES
    -----
    Must be applied to labels as well.
    '''
    arr = np.arange(1, v_box * h_box + 1, dtype=int) # middle index is not swapped if odd
    np.random.seed(seed)
    np.random.shuffle(arr)

    rows = int(HEIGHT/v_box)
    cols = int(WIDTH/h_box)

    height, width, channels = image.shape
    ret = np.copy(image)
    cpy = np.copy(image)

    for num in range(int((v_box*h_box)/2)):
        c1 = ((arr[2*num] - 1) % h_box) * cols
        r1 = int((arr[2*num]-1)/h_box) * rows
        c2 = ((arr[2*num + 1] - 1) % h_box) * cols
        r2 = int((arr[2*num + 1]-1)/h_box) * rows
        
        cpy[r1:(r1 + rows), c1:(c1 + cols)] = ret[r1:(r1 + rows), c1:(c1 + cols)]
        ret[r1:(r1 + rows), c1:(c1 + cols)] = ret[r2:(r2 + rows), c2:(c2 + cols)]
        ret[r2:(r2 + rows), c2:(c2 + cols)] = cpy[r1:(r1 + rows), c1:(c1 + cols)]

    return ret

def apply_saltpepper(image, seed, density=10):
    '''
    Randomly turns pixels white, black, or gray to add noise.

    INPUTS
    ------
    image: Input image of dimensions of WIDTH x HEIGHT
    seed: Allows you to repreduce results. If called with the same seed twice, the output will be the same.
        any integer
    density: Controls the amount of black and white pixels added. A greater density lowers the probability that pixels are altered.
        any integer greater than 2
    '''
    ret = np.copy(image)

    np.random.seed(seed)
    arr = np.random.randint(0,density, size=(720, 1280, 1)) # pixels that we want to change, indicated by 1
    ret = np.where(arr == 0, 0, ret)
    ret = np.where(arr == 1, 255, ret)
    random.seed(seed+100)
    val = random.randint(1, 255)
    ret = np.where(arr == 2, val, ret)
    
    return ret

def wave(image, amplitude=100, shift=0, stretch=0.02, axis=1):
    '''
    Creates a wave-like effect on image, based on a sinusoidal curve.

    INPUTS
    ------
    axis: Determines which axis the wave occurs on.
        default is 1, which is the vertical axis
        0 is for the horizontal axis
    amplitude: Amplitude of the sinusoidal curve. 
        Remember that the dimensions of the image are WIDTH x HEIGHT, so amplitude should be much smaller than the corresponding dimension for optimal results
        any float
        default is 100 pixels
    shift: Hotizontal translation of the curve.
        any float
        default is 0
    stretch: Stretches the sinusoid along the horizontal axis.
        The larger the stretch, the lower the period of the sinusoid, which can reduce the quality of the results and make it unrecognizable.
        any float
        default is 0.02
    
    NOTES
    -----
    Must be applied to labels as well.
    '''
    ret = make_black()
    if (axis is 0):
        factor = (WIDTH - 2*amplitude)/WIDTH
        rsz = resize_image(image, factor, factor)

        for i in range(0, HEIGHT):
            num = round(amplitude*math.sin(stretch*(i + math.radians(shift))))

            ret[i, amplitude + num:(WIDTH - amplitude + num)] = rsz[i, amplitude:(WIDTH - amplitude)]
    elif (axis is 1):
        factor = (HEIGHT - 2*amplitude)/HEIGHT
        rsz = resize_image(image, factor, factor)

        for i in range(0, WIDTH):
            num = round(amplitude*math.sin(stretch*(i + math.radians(shift))))

            ret[amplitude + num:(HEIGHT - amplitude + num), i] = rsz[amplitude:(HEIGHT - amplitude), i]
    
    return ret

def generate_mask_shapes(seed, num_shapes):
    '''
    Generates a mask. Draws specified number of shapes that may overlap.
    '''
    ret = make_black()

    for num in range(num_shapes):
        cv2.rectangle()

    cv2.rectangle()
    cv2.circle()
    cv2.ellipse()
    cv2.polygon()

    return ret

def apply_mask(background: np.ndarray, mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    '''
    Apply the mask and image to the background.

    Inputs
    ------
    background: np.ndarray(shape=(H, W, C), dtype=np.uint8) my image (from the camera)
        The background onto which we apply the mask and image.
    mask: np.ndarray(shape=(H', W'[, C']), dtype=np.uint8) black and white image (for shadow) --> use cv draw to draw shapes onto mask
        The mask by which we mask the image. Resized to match the
        background. The mask is in the range [0, 255]. This is so that
        the mask can be immediately displayed as an image; however, it
        adds more complexity to the math.
        Another possibility is to be a float in the range [0, 1].
    image: np.ndarray(shape=(H", W", C"), dtype=np.uint8) mask is grey for shadow
        The image to draw on the background. Resized to match the background.
    '''
    background_shape = (background.shape[1], background.shape[0])
    resize_to_background = lambda x: cv2.resize(x, dsize=background_shape)

    background = background.astype(float)
    mask = resize_to_background(mask.astype(float))    # Resize to background
    image = resize_to_background(image.astype(float))  # Resize to background

    return (((255 - mask) * background + mask * image) / 255).astype(np.uint8)

# def shadow(image, num, seed):
#     for k in range(num):
#         random.seed(seed*k)
#         size = random.randint(50,200)
#         random.seed(seed*k + 100)
#         a = random.randint(0, HEIGHT - size - 1)
#         random.seed(seed*k + 200)
#         b = random.randint(0, WIDTH - size - 1)
#         random.seed(seed*k + 300)
#         shade = random.randint(400, 700)
#         for i in range(size):
#             for j in range(size):
#                 image[a + i, b + j] = image[a + i, b + j] * shade / 1000

#     return image

def generate_mask_round(seed, num_shapes):
    '''
    Generates a mask. Draws specified number of shapes that may overlap.
    '''
    ret = make_black()

    for k in range(int(num_shapes/2)): # for circles
        random.seed(seed*k)
        radius = random.randint(15, 25)
        random.seed(seed*k + 100)
        a = random.randint(radius, HEIGHT - radius - 1) 
        random.seed(seed*k + 200)
        b = random.randint(radius, WIDTH - radius - 1)
        cv2.circle(ret,(a,b), radius, (255,255,255), -1)

    for k in range(num_shapes - int(num_shapes/2)):
        random.seed(seed*k + 300)
        major_axis = random.randint(25,40)
        random.seed(seed*k + 400)
        minor_axis = random.randint(10,25)
        random.seed(seed*k + 500)
        a = random.randint(major_axis, HEIGHT - major_axis - 1) 
        random.seed(seed*k + 600)
        b = random.randint(minor_axis, WIDTH - minor_axis - 1)
        random.seed(seed*k + 700)
        angle = random.randint(0, 360)
        cv2.ellipse(ret, (b,a), (major_axis, minor_axis), angle, 0, 360, (255,255,255), -1)

    return ret

def shadow(image, num, seed, darken):   
    ret = apply_mask(image, apply_gaussian(generate_mask_round(seed, num)), colour(image, bf = darken, gf = darken, rf = darken))

    return ret

def raindrop(image, seed, num=40, kernel_x=50, kernel_y=200):
    '''
    overlay gaussian blur onto groups of pixels?
    shade groups of pixels?
    '''
    return apply_mask(image, apply_gaussian(generate_mask_round(seed, num)), apply_gaussian(image, kernel_x = kernel_x, kernel_y = kernel_y))

def glare_mask(seed):
    ret = make_black()

    # center
    radius = 5
    random.seed(seed + 100)
    a = random.randint(radius + 50, HEIGHT - radius - 51) 
    random.seed(seed + 200)
    b = random.randint(radius + 50, WIDTH - radius - 51)
    cv2.circle(ret,(b,a), radius, (255,255,255), -1)

    pts1 = np.array([[b,a],[0,0],[25, 0]], np.int32)
    pts1 = pts1.reshape((-1,1,2))
    cv2.fillPoly(ret,[pts1],(255,255,255))

    pts1 = np.array([[b,a],[140,0],[170, 0]], np.int32)
    pts1 = pts1.reshape((-1,1,2))
    cv2.fillPoly(ret,[pts1],(255,255,255))

    pts1 = np.array([[b,a],[400,0],[420, 0]], np.int32)
    pts1 = pts1.reshape((-1,1,2))
    cv2.fillPoly(ret,[pts1],(255,255,255))

    pts1 = np.array([[b,a],[1120,0],[1160, 0]], np.int32)
    pts1 = pts1.reshape((-1,1,2))
    cv2.fillPoly(ret,[pts1],(255,255,255))

    pts1 = np.array([[b,a],[WIDTH - 1,0],[WIDTH - 1, 25]], np.int32)
    pts1 = pts1.reshape((-1,1,2))
    cv2.fillPoly(ret,[pts1],(255,255,255))

    pts1 = np.array([[b,a],[WIDTH - 1,300],[WIDTH - 1, 325]], np.int32)
    pts1 = pts1.reshape((-1,1,2))
    cv2.fillPoly(ret,[pts1],(255,255,255))

    pts1 = np.array([[b,a],[WIDTH - 1,400],[WIDTH - 1, 425]], np.int32)
    pts1 = pts1.reshape((-1,1,2))
    cv2.fillPoly(ret,[pts1],(255,255,255))

    pts1 = np.array([[b,a],[WIDTH - 1,650],[WIDTH - 1, 675]], np.int32)
    pts1 = pts1.reshape((-1,1,2))
    cv2.fillPoly(ret,[pts1],(255,255,255))

    pts1 = np.array([[b,a],[900,719],[930, 719]], np.int32)
    pts1 = pts1.reshape((-1,1,2))
    cv2.fillPoly(ret,[pts1],(255,255,255))

    pts1 = np.array([[b,a],[500,719],[530, 719]], np.int32)
    pts1 = pts1.reshape((-1,1,2))
    cv2.fillPoly(ret,[pts1],(255,255,255))

    pts1 = np.array([[b,a],[220,719],[250, 719]], np.int32)
    pts1 = pts1.reshape((-1,1,2))
    cv2.fillPoly(ret,[pts1],(255,255,255))

    pts1 = np.array([[b,a],[0,250],[0,270]], np.int32)
    pts1 = pts1.reshape((-1,1,2))
    cv2.fillPoly(ret,[pts1],(255,255,255))

    pts1 = np.array([[b,a],[0,500],[0,520]], np.int32)
    pts1 = pts1.reshape((-1,1,2))
    cv2.fillPoly(ret,[pts1],(255,255,255))

    pts1 = np.array([[b,a],[0,700],[0,719]], np.int32)
    pts1 = pts1.reshape((-1,1,2))
    cv2.fillPoly(ret,[pts1],(255,255,255))

    pts1 = np.array([[b,a],[750,0],[780, 0]], np.int32)
    pts1 = pts1.reshape((-1,1,2))
    cv2.fillPoly(ret,[pts1],(255,255,255))

    return ret

def lens_glare(image, seed):
    return apply_mask(image, apply_gaussian(glare_mask(seed), kernel_x = 10, kernel_y = 10), np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8))

def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distored_image.reshape(image.shape)

if __name__ == "__main__":
    image = initialize(path)

    #display_image(image, window_name)

    # black = make_black()
    # display_image(black, window_name)

    # resized = resize_image(image, 0.5, 0.25)
    # display_image(resized, window_name)

    # sp1 = apply_saltpepper(image, 0)
    # display_image(sp1, window_name)

    # sp2 = apply_saltpepper(image, 1)
    # display_image(sp2, window_name)

    # sp3 = apply_saltpepper(image, 2)
    # display_image(sp3, window_name)

    # pixel = pixel_swap(image, 100, 100000)
    # display_image(pixel, window_name)

    # ms1 = apply_mosaic(image, 10, 1000)
    # display_image(ms1, window_name)

    # ref0 = reflect(image, 0)
    # display_image(ref0, window_name)

    # ref1 = reflect(image, 1)
    # display_image(ref1, window_name)

    # ref_1 = reflect(image, -1)
    # display_image(ref_1, window_name)

    # gs5 = gaussian(image, 5, 6)
    # display_image(gs5, window_name)

    # gs10 = gaussian(image, 10, 3)
    # display_image(gs10, window_name)

    # gs20 = gaussian(image, 20, 25)
    # display_image(gs20, window_name)

    # gs40 = gaussian(image, 40, 40)
    # display_image(gs40, window_name)

    # rot1 = rotate(image, -60)
    # display_image(rot1, window_name)

    # rot1 = rotate(image, 30)
    # display_image(rot1, window_name)

    # sh1 = shear(image, 0.4, 0.2)
    # display_image(sh1, window_name)

    # sh2 = shear(image, 0.7, 0.5)
    # display_image(sh2, window_name)

    # wv = wave(image)
    # display_image(wv, window_name)

    # ums1 = uniform_mosaic(image, 0, 5, 10)
    # display_image(ums1, window_name)

    # ums2 = uniform_mosaic(image, 1, 16, 128)
    # display_image(ums2, window_name)

    # ums3 = uniform_mosaic(image, 1, 720, 1280)
    # display_image(ums3, window_name)

    # ums3 = uniform_mosaic(image, 0, 9, 10)
    # display_image(ums3, window_name)

    # gray = colour(image, channel="gray")
    # display_image(gray, window_name)
 
    # blue = colour (image, 0,1,0)
    # display_image(blue, window_name)

    # green = colour (image, 1, 0, 1)
    # display_image(green, window_name)

    # red = colour (image, 1 , 1 , 0)
    # display_image(red, window_name)

    # boost_red = colour (image, 1, 1, 2)
    # display_image(boost_red, window_name)

    # sdw1 = shadow(image, 10, 1)
    # display_image(sdw1, window_name)

    # blur = raindrop(image, 1)
    # display_image(blur, window_name)

    # cool = generate_mask_round(2, 20)
    # display_image(cool, window_name)

    # rain = raindrop(image, 10)
    # display_image(rain, window_name)

    shd = shadow(image, 40, 15, 0.5)
    display_image(shd, window_name)

    # glare = lens_glare(image,5)
    # display_image(glare, window_name)

    cv2.destroyAllWindows()



# #Image grayscale

# def im_grayscale(image, scale):
#     return image

# # Load the input image
# image = cv2.imread('C:\\Documents\\full_path\\tomatoes.jpg')
# cv2.imshow('Original', image)
# cv2.waitKey(0)
# # Use the cvtColor() function to grayscale the image
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Grayscale', gray_image)
# cv2.waitKey(0) 
# # Window shown waits for any key pressing event
# cv2.destroyAllWindows()
 
 
# while(1):
#     _, frame = cap.read()
#     # It converts the BGR color space of image to HSV color space
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
     
#     # Threshold of blue in HSV space
#     lower_blue = np.array([60, 35, 140])
#     upper_blue = np.array([180, 255, 255])
 
#     # preparing the mask to overlay
#     mask = cv2.inRange(hsv, lower_blue, upper_blue)
     
#     # The black region in the mask has the value of 0,
#     # so when multiplied with original image removes all non-blue regions
#     result = cv2.bitwise_and(frame, frame, mask = mask)
 
#     cv2.imshow('frame', frame)
#     cv2.imshow('mask', mask)
#     cv2.imshow('result', result)
     
#     cv2.waitKey(0)
 
