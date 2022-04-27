import numpy as np
from scipy.ndimage import binary_fill_holes, gaussian_filter
import cv2
from skimage import filters, measure
import imageio as iio
import matplotlib.pyplot as plt

def detect_xyr(img):
    """
    Taken from https://github.com/linchundan88/Fundus-image-preprocessing/blob/master/fundus_preprocessing.py
    """
    
    # determine the minimum and maximum possible radii
    MIN_REDIUS_RATIO = 0. #cambie
    MAX_REDIUS_RATIO = 1.0
    # get width and height of the image
    width = img.shape[1]
    height = img.shape[0]

    # get the min length of the image
    myMinWidthHeight = min(width, height)
    # and get min and max radii according to the proportion
    myMinRadius = round(myMinWidthHeight * MIN_REDIUS_RATIO)
    myMaxRadius = round(myMinWidthHeight * MAX_REDIUS_RATIO)

    # turn the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # estimate the circles
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist=450, param1=200, param2=0.9,
                               minRadius=myMinRadius,
                               maxRadius=myMaxRadius)

    (x, y, r) = (0, 0, 0)
    found_circle = False

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        if (circles is not None) and (len(circles == 1)):
            x1, y1, r1 = circles[0]
            if x1 > (2 / 5 * width) and x1 < (3 / 5 * width) \
                    and y1 > (2 / 5 * height) and y1 < (3 / 5 * height):
                x, y, r = circles[0]
                found_circle = True


    if not found_circle:
        # sum the R, G, B channels to form a single image
        sum_of_channels = np.asarray(np.sum(img,axis=2), dtype=np.uint8)
        # threshold the image using Otsu
        fov_mask = sum_of_channels > filters.threshold_otsu(sum_of_channels)
        # fill holes in the approximate FOV mask
        fov_mask = np.asarray(binary_fill_holes(fov_mask), dtype=np.uint8)

        # get the coordinates of the bounding box
        coordinates = measure.regionprops(fov_mask)[0].bbox
        # estimate the size of each side
        side_1 = coordinates[2] - coordinates[0]
        side_2 = coordinates[3] - coordinates[1]
        # get the radius
        r = side_2 // 2
        # and the central coordinates
        y = coordinates[0] + (side_1 // 2)
        x = coordinates[1] + (side_2 // 2)

    return found_circle, x, y, r

def get_fov_mask(fundus_picture):
    '''
    Obtains the fov mask of the given fundus picture.
    '''

    fov_mask = np.zeros((fundus_picture.shape[0], fundus_picture.shape[1]), dtype=np.uint8)
    
    # estimate the center (x,y) and the radius (r) of the circle
    _, x, y, r = detect_xyr(fundus_picture)
    

    Y, X = np.ogrid[:fundus_picture.shape[0], :fundus_picture.shape[1]]
    dist_from_center = np.sqrt((X - x)**2 + (Y-y)**2)

    fov_mask[dist_from_center <= r] = 255
                
    return fov_mask, (x,y,r)

def crop_fov(fundus_picture, mask, fov_mask= None):
    '''
    Extract an approximate FOV mask, and crop the picture around it
    '''

    # if the FOV mask is not given, estimate it
    if fov_mask is None:
        fov_mask, (x, y, r) = get_fov_mask(fundus_picture)
    # if the FOV mask is given, estimate the center and the radii
    else:
        # get the coordinates of the bounding box
        coordinates = measure.regionprops(fov_mask)[0].bbox
        # estimate the size of each side
        side_1 = coordinates[2] - coordinates[0]
        side_2 = coordinates[3] - coordinates[1]
        # get the radius
        r = side_2 // 2
        # and the central coordinates
        y = coordinates[0] + (side_1 // 2)
        x = coordinates[1] + (side_2 // 2)

    lim_x_inf = (y - r) if y>=r else 0
    lim_x_sup = (y + r) if (y+r)<fundus_picture.shape[0] else fundus_picture.shape[0]
    lim_y_inf = (x - r) if x>=r else 0
    lim_y_sup = (x + r) if (x+r)<fundus_picture.shape[1] else fundus_picture.shape[1]

    return (fundus_picture[lim_x_inf:lim_x_sup, lim_y_inf:lim_y_sup,:], 
        mask[lim_x_inf:lim_x_sup, lim_y_inf:lim_y_sup])

def crop_fov_2(fundus_picture,mask1, mask2):
    '''
    Extract an approximate FOV mask, and crop the picture around it
    '''

    # if the FOV mask is not given, estimate it
    if fov_mask is None:
        fov_mask, (x, y, r) = get_fov_mask(fundus_picture)
    # if the FOV mask is given, estimate the center and the radii
    else:
        # get the coordinates of the bounding box
        coordinates = measure.regionprops(fov_mask)[0].bbox
        # estimate the size of each side
        side_1 = coordinates[2] - coordinates[0]
        side_2 = coordinates[3] - coordinates[1]
        # get the radius
        r = side_2 // 2
        # and the central coordinates
        y = coordinates[0] + (side_1 // 2)
        x = coordinates[1] + (side_2 // 2)

    lim_x_inf = (y - r) if y>=r else 0
    lim_x_sup = (y + r) if (y+r)<fundus_picture.shape[0] else fundus_picture.shape[0]
    lim_y_inf = (x - r) if x>=r else 0
    lim_y_sup = (x + r) if (x+r)<fundus_picture.shape[1] else fundus_picture.shape[1]

    return (fundus_picture[lim_x_inf:lim_x_sup, lim_y_inf:lim_y_sup,:], 
        mask1[lim_x_inf:lim_x_sup, lim_y_inf:lim_y_sup],
        mask2[lim_x_inf:lim_x_sup, lim_y_inf:lim_y_sup])


# imagen = '/mnt/Almacenamiento/ODOC_segmentation/data/images/IDRID/001.png'
# mask = '/mnt/Almacenamiento/ODOC_segmentation/data/OD1/IDRID/001.png'

# img = iio.imread(imagen)
# mk = iio.imread(mask)

# new_img, new_mask = crop_fov(img,mk)


# fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
# ax0.imshow(img)
# ax1.imshow(new_img)
# ax2.imshow(new_mask)

# plt.show()



