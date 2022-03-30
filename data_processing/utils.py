import numpy as np
from scipy.ndimage import binary_fill_holes, gaussian_filter
import cv2
from skimage import filters, measure
import imageio as iio
import matplotlib.pyplot as plt


def get_fov_mask(fundus_picture):
    '''
    Obtains the fov mask of the given fundus picture.
    '''

    image = np.array(fundus_picture)

    # sum the R, G, B channels to form a single image
    sum_of_channels = np.asarray(np.sum(fundus_picture,axis=2), dtype=np.uint8)
    # threshold the image using Otsu
    fov_mask = sum_of_channels > filters.threshold_otsu(sum_of_channels)
    # fill holes in the approximate FOV mask
    fov_mask = np.asarray(binary_fill_holes(fov_mask), dtype=np.uint8)

    #ver a futuro

    # # find the contour points of the fov mask
    # points,_= cv2.findContours(fov_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

    # # using the contour, obtain the convex hull that involves it
    # hull = cv2.convexHull(np.concatenate(points).squeeze(1))

    # # obtain the hull mask by drawing and filling the convex hull over a black image with the same size
    # hull_mask = cv2.drawContours(np.zeros(image.shape[:-1]), [hull], -1, 1, cv2.FILLED)
    # #hull_mask = cv2.erode(hull_mask, np.ones((11,11), np.uint8))
    return fov_mask

def crop_fov(fundus_picture,mask):
    '''
    Extract an approximate FOV mask, and crop the picture around it
    '''

    #get the fov mask of the picture
    fov_mask = get_fov_mask(fundus_picture)

    # get the coordinate of a bounding box around the fov mask
    coordinates = measure.regionprops(fov_mask)[0].bbox

    # crop the image and return
    return (fundus_picture[coordinates[0]:coordinates[2],coordinates[1]:coordinates[3],:],
        mask[coordinates[0]:coordinates[2],coordinates[1]:coordinates[3]])

def crop_fov_2(fundus_picture,mask1, mask2):
    '''
    Extract an approximate FOV mask, and crop the picture around it
    '''

    #get the fov mask of the picture
    fov_mask = get_fov_mask(fundus_picture)

    # get the coordinate of a bounding box around the fov mask
    coordinates = measure.regionprops(fov_mask)[0].bbox

    # crop the image and return
    return (fundus_picture[coordinates[0]:coordinates[2],coordinates[1]:coordinates[3],:],
        mask1[coordinates[0]:coordinates[2],coordinates[1]:coordinates[3]],
        mask2[coordinates[0]:coordinates[2],coordinates[1]:coordinates[3]])


imagen = '/mnt/Almacenamiento/ODOC_segmentation/data/images/IDRID/001.png'
mask = '/mnt/Almacenamiento/ODOC_segmentation/data/OD1/IDRID/001.png'

img = iio.imread(imagen)
mk = iio.imread(mask)

new_img, new_mask = crop_fov(img,mk)


fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
ax0.imshow(img)
ax1.imshow(new_img)
ax2.imshow(new_mask)

plt.show()



