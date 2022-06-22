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
import csv

proyect_path = '/mnt/Almacenamiento/ODOC_segmentation'
or_data_path = '/raw_data/'
dst_data_path = '/data/'
dataset = 'RIM_ONE_R3'
sub_dataset = ['/Glaucoma and suspects','/Healthy']

def get_mask(OD_paths, OC_paths, names):
    for idx_p in range(len(OD_paths)):
        OD_image = iio.imread(OD_paths[idx_p])
        OC_image = iio.imread(OC_paths[idx_p])
        row, col = OD_image.shape
        #print(OD_image.shape, OC_image.shape)
        OD_half = OD_image[:,:int(col/2)]
        OC_half = OC_image[:,:int(col/2)]

        iio.imwrite(proyect_path + dst_data_path + 'OD1/' + dataset + '/' + names[idx_p], OD_half)
        iio.imwrite(proyect_path + dst_data_path + 'OC/' + dataset + '/' + names[idx_p], OC_half)



    return

def get_images(paths, names):
    for idx_p in range(len(paths)):
        img = iio.imread(paths[idx_p])
        row, col, depth = img.shape
        #print(row, col, depth)
        img_half = img[:,:int(col/2),:]

        # fig, (ax0, ax1) = plt.subplots(1, 2)
        # ax0.imshow(img)
        # ax1.imshow(img_half)
        # plt.show()

        iio.imwrite(proyect_path+dst_data_path+'images/' + dataset  + '/' +  names[idx_p], img_half) #guardo
    return

def main():
    img_name = 1
    img_paths = []
    final_img_name = []
    OD_paths = []
    OC_paths = []
    dst_filename = '/mnt/Almacenamiento/ODOC_segmentation/data/images/RIM_ONE_R3/labels.csv'
    with open(dst_filename, 'a+') as tags:
        writer = csv.writer(tags)
        writer.writerow(['Name','label'])


    for subD in sub_dataset:
        for p_img in sorted(glob.glob(proyect_path + or_data_path + dataset + subD + '/Stereo Images/*.jpg')):
            img_paths.insert(len(img_paths), p_img) #path de las imagenes
            final_img_name.insert(len(final_img_name),'{0:0=3d}.png'.format(img_name))
            with open(dst_filename, 'a+') as tags:
                writer = csv.writer(tags)
                if subD == '/Glaucoma and suspects':
                    writer.writerow([f"{img_name:03}",'Glaucomatous'])
                elif subD == '/Healthy':
                    writer.writerow([f"{img_name:03}",'Non-glaucomatous'])

            img_name += 1
            


        for p_mask in sorted(glob.glob(proyect_path + or_data_path + dataset + subD + '/Average_masks/*Disc*.png')):
            OD_paths.insert(len(OD_paths), p_mask) #path de las mascaras

        for p_mask in sorted(glob.glob(proyect_path + or_data_path + dataset + subD + '/Average_masks/*Cup*.png')):
            OC_paths.insert(len(OC_paths), p_mask) #path de las mascaras

        get_images(img_paths,final_img_name)
        get_mask(OD_paths,OC_paths,final_img_name)
        print(subD)

        
        #Ver si agregar a los expertos





if __name__ == '__main__':
    main()
    
# img_path = '/mnt/Almacenamiento/ODOC_segmentation/data/images/RIM_ONE_R3/139.png'
# OC_path = '/mnt/Almacenamiento/ODOC_segmentation/data/OC/RIM_ONE_R3/139.png'
# OD_path = '/mnt/Almacenamiento/ODOC_segmentation/data/OD1/RIM_ONE_R3/139.png'


# img = iio.imread(img_path)
# OC = iio.imread(OC_path)
# OD = iio.imread(OD_path)

# print(collections.Counter(OD[800,:])) #255, 0 y 128 para las marcas


# fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
# ax0.imshow(img)
# ax1.imshow(img[:,:,0] * OC)
# ax2.imshow(img[:,:,0] * (OD))

# plt.show()
