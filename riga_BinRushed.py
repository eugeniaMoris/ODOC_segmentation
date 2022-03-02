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
dataset = 'RIGA'
sub_dataset = ['BinRushed1-Corrected','BinRushed2','BinRushed3','BinRushed4']


def generate_mask(path, img):
    ''' generamos la imagen de mascara '''
    print(path)

    mask = iio.imread(path)

    img_arr = np.array(img[:,:,0],dtype=np.uint8)
    mask_arr = np.array(mask[:,:,0],dtype=np.uint8)
    
    resta = compare_images(img, mask, method='diff') #usamos este metodo las las imagenes .jpg
    resta = filters.sobel(resta)
    resta = np.array(resta[:,:,0])
    resta[resta > 0.1] = 1
    resta[resta < 0.1] = 0

    OD = ndimage.binary_fill_holes(resta).astype(int) # obtengo la imagen para OD segmentation

    erosion_img = erosion(OD, disk(20))
    OC_border = resta * erosion_img
    OC = ndimage.binary_fill_holes(OC_border).astype(int)# obtengo la imagen para OC segmentation

    return OD, OC



def get_mask(file_name, final_name, img, sdataset):
    '''
    Genero las mascaras de las imagenes,
    En total hay 6 expertor por lo que se genera la de todos y una mascara de majority voting
    ----
    inputs:
    file_name: nombre origen de la imagen a la que generaremos las mascaras
    final_name: nombre final con el que fue guardada la imagen
    img: imagen sin marcas de expertos
    dataset: dataset de donde se saca la imagen
    '''

    mask_paths = []
    for mask in sorted(glob.glob(proyect_path + or_data_path + dataset + '/BinRushed/' + sdataset + '/' + file_name + '-*' )):
        mask_paths.insert(len(mask_paths), mask)
    
    od_imgs = []
    oc_imgs = []
    for m in mask_paths:
        OD, OC = generate_mask(m,img)
        od_imgs.insert(len(od_imgs),OD)
        oc_imgs.insert(len(oc_imgs),OC)

    OD_add = np.zeros(od_imgs[0].shape)
    OC_add = np.zeros(oc_imgs[0].shape)

    for idx in range(len(od_imgs)):
        OD_add = np.add(OD_add,od_imgs[idx])
        iio.imwrite(proyect_path + dst_data_path + 'OD_extra/' + dataset + '-BinRushed' + '/' + str(idx+1) + '_' + final_name,od_imgs[idx])

        OC_add = np.add(OC_add,oc_imgs[idx])
        iio.imwrite(proyect_path + dst_data_path + 'OC_extra/' + dataset + '-BinRushed' + '/' + str(idx+1) + '_' + final_name,oc_imgs[idx])


    OD_add[OD_add < 3] = 0
    OD_add[OD_add >= 3] = 1

    OC_add[OC_add < 3] = 0
    OC_add[OC_add >= 3] = 1
    
    iio.imwrite(proyect_path + dst_data_path + 'OD1/' + dataset + '-BinRushed' + '/' + final_name,OD_add)
    iio.imwrite(proyect_path + dst_data_path + 'OC/' + dataset + '-BinRushed' + '/' + final_name,OC_add)
    

    return

def get_images(paths, names, sdataset):
    ''' obtengo las imagenes originales y a artir del nombre obtengo los paths de las mascaras y las genero'''
    print('paths in get img' ,paths)
    for p_i in range(len(paths)):
        
        #we get the original image name
        name = ntpath.basename(paths[p_i])
        n = name.split('.')
        old_name = n[0].replace('prime','')
        print(old_name)
        
        img = iio.imread(paths[p_i]) #obtengo imagen
        print('img path: ', paths[p_i])
        #procesar
        iio.imwrite(proyect_path+dst_data_path+'images/' + dataset + '-BinRushed' + '/' +  names[p_i],img) #guardo

        get_mask(old_name, names[p_i], img, sdataset)

    return

def main():
    
    img_name = 1
    #f"{img_name:03}"

    train_img = []
    validation_img = [] #no hay
    test_img = [] #no hay

    img_paths = []
    final_img_name = []
    mask_paths = []

    for data in sub_dataset:

        for img in sorted(glob.glob(proyect_path + or_data_path + dataset + '/BinRushed/' + data + '/*prime.jpg')):
            img_paths.insert(len(img_paths), img)
            final_img_name.insert(len(final_img_name),'{0:0=3d}.png'.format(img_name))
            img_name += 1
        


        #print(img_paths)
        get_images(img_paths, final_img_name, data)

        img_paths = []
        final_img_name = []
        get_mask = []


if __name__ == '__main__':
    main()
    
# img_path = '/mnt/Almacenamiento/ODOC_segmentation/data/images/RIGA/001.png'
# OC_path = '/mnt/Almacenamiento/ODOC_segmentation/data/OC/RIGA/001.png'
# OD_path = '/mnt/Almacenamiento/ODOC_segmentation/data/OD1/RIGA/001.png'


# img = iio.imread(img_path)
# OC = iio.imread(OC_path)
# OD = iio.imread(OD_path)

# print(collections.Counter(OD[800,:])) #255, 0 y 128 para las marcas


# fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
# ax0.imshow(img)
# ax1.imshow(img[:,:,0] * OC)
# ax2.imshow(img[:,:,0] * (OD))

# plt.show()








# path = '/mnt/Almacenamiento/ODOC_segmentation/raw_data/RIGA/MESSIDOR'

# img_mask = iio.imread(path + '/image1-1.tif' )
# img = iio.imread(path + '/image1prime.tif')

# print( img.shape, img_mask.shape)
# resta = img[:,:,0] - img_mask[:,:,0]
# OD = ndimage.binary_fill_holes(resta).astype(int)

# sub_OD = erosion(OD, disk(20))
# OC = resta * sub_OD
# OC_fill = ndimage.binary_fill_holes(OC).astype(int)


# fig, (ax0, ax1,ax2,ax3,ax4) = plt.subplots(1, 5)
# ax0.imshow(img)
# ax1.imshow(OD)
# ax2.imshow(OC_fill)
# ax3.imshow(img[:,:,0] * OD)
# ax4.imshow(img[:,:,0] * OC_fill)


# plt.show()