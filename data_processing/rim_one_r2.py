from cgi import test
from email.mime import image
from re import sub
from unittest import main
import numpy as np
import imageio as iio
import matplotlib.pyplot as plt
import glob
import ntpath
import collections
import scipy.io
from scipy import ndimage
from skimage.morphology import disk, erosion
from skimage.util import compare_images
from skimage import filters

proyect_path = '/mnt/Almacenamiento/ODOC_segmentation'
or_data_path = '/raw_data/'
dst_data_path = '/data/'
dataset = 'RIM_ONE_R2'
sub_dataset = ['/Normal','/Glaucoma and glaucoma suspicious']
sub_dataset_mask = ['/Normal segmentation','/Glaucoma and glaucoma suspicious segmentation']

def generate_mask(paths):
    for p in paths:
        name_base = ntpath.basename(p)
        n1 = name_base.split('.')
        n2 = n1[0].split('-')
        name = n2[0].replace('Im','')

        mask = iio.imread(p)

        iio.imwrite(proyect_path + dst_data_path + 'OD1/' + dataset + '/' + name + '.png', mask)

    return

def generate_images(paths):
    for p in paths:
        name_base = ntpath.basename(p)
        n = name_base.split('.')
        name = n[0].replace('Im','')

        img = iio.imread(p)

        iio.imwrite(proyect_path + dst_data_path + 'images/' + dataset  + '/' +  name + '.png', img)
    return

def main():
    img_paths = []
    OD_paths = []


    for sub_idx in range(len(sub_dataset)):
        for p_img in sorted(glob.glob(proyect_path + or_data_path + dataset + sub_dataset[sub_idx] + '/*.jpg')):
            img_paths.insert(len(img_paths), p_img) #path de las imagenes

        for p_mask in sorted(glob.glob(proyect_path + or_data_path + dataset + sub_dataset_mask[sub_idx] + '/*.bmp')):
            OD_paths.insert(len(OD_paths), p_mask) #path de las mascaras

        generate_images(img_paths)
        generate_mask(OD_paths)

if __name__ == '__main__':
    main()
    



# img_path = '/mnt/Almacenamiento/ODOC_segmentation/data/images/RIM_ONE_R2/001.png'
# OD_path = '/mnt/Almacenamiento/ODOC_segmentation/data/OD1/RIM_ONE_R2/001.png'

# img = iio.imread(img_path)
# OD = iio.imread(OD_path)



# fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
# ax0.imshow(img)
# ax1.imshow(OD)
# ax2.imshow(img[:,:,0] * (OD))

# plt.show()
