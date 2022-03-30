import numpy as np
import glob
import ntpath
import imageio as iio
from skimage.draw import polygon
import argparse
from scipy.ndimage import binary_fill_holes, gaussian_filter
import cv2
from skimage import measure, filters
import data_processing.utils as utils

proyect_path = '/mnt/Almacenamiento/ODOC_segmentation'
or_data_path = '/raw_data/'
dst_data_path = '/data/'
dataset= 'DRIONS/'


def get_mask_data(path):
    name = ntpath.basename(path)
    n = name.split('.')
    sub_n = n[0].split('Expert')
    expert, img_n = sub_n[1].split('_')
    print('Expert: ', expert, ' img: ', img_n)

    return expert, img_n

def generate_mask(mask_path, img_size, img_name, expert):
    '''
    A partir del conjunto de pares (x,y) dado por expertos,
    se genera el poligono con la mascara correspondiente al OD
    ------
    Inputs:
    mask_path: ruta de conjunto de pares dados por expertor .txt
    img_size: tamaño de la imagen de fondo de ojo (400,600)
    img_name: nombre del la imagen mascada [1 - 110]
    expert: numero del experto que genero la mascara [1,2]
    
    '''
    mask = np.loadtxt(mask_path,delimiter=',')

    #Repito el primer valor para cerrar el circulo
    new_row = mask[0,:]
    mask = np.vstack([mask, new_row]) 

    img = np.zeros(img_size, dtype=int)
    rr, cc = polygon(mask[:,1], mask[:,0])
    img[rr,cc] = 1
    
    return img

def get_path_mask(name_img, paths):
    experts = []
    for p in paths:
        if name_img in p:
            experts.insert(len(experts), p)

    return experts


def main():
    '''
    Data processing para el dataset DRIONS
    Se generan las mascaras y se agregan las imagenes a /data

    ver si se necesira prepara un resize ya que las imagenes no son cuadradas
    elementos train validation test todavia no definidas
    ----

    ----
    Output:
    Nombre dataset
    lista archivos train
    lista archivos valid
    lista archivos test

    '''

    anotations = []
    imgs_paths = []

    for anot in sorted(glob.glob(proyect_path + or_data_path + dataset + 'experts_anotation/anot*.txt')):
        anotations.insert(len(anotations), anot)

    for img_p in sorted(glob.glob(proyect_path + or_data_path + dataset + 'images/*.jpg')):
        imgs_paths.insert(len(imgs_paths), img_p)

    for i in imgs_paths:
        name = ntpath.basename(i)
        n = name.split('.')
        name= n[0].split('_')

        im = iio.imread(i)
        height, width,_ = im.shape

        mask_paths  = get_path_mask(name[1], anotations)
        mask1 = generate_mask(mask_paths[0],(height, width),name[1], '1')
        mask2 = generate_mask(mask_paths[1],(height, width),name[1], '2')
        
        new_img, n_mask1, n_mask2 = utils.crop_fov_2(im, mask1, mask2)


        iio.imwrite(proyect_path+dst_data_path+'images/' + dataset + name[1] + '.png',new_img)
        iio.imwrite(proyect_path + dst_data_path + 'OD1/'+ dataset + name[1] + '.png', n_mask1)
        iio.imwrite(proyect_path + dst_data_path + 'OD_extra/'+ dataset + name[1] + '.png', n_mask2)
        


    # size = (height, width)
    # for j in anotations:
    #     expert, img_name = get_mask_data(j)
    #     generate_mask(j,size,img_name, expert)


    return 'DRIONS'


# im = iio.imread('/mnt/Almacenamiento/ODOC_segmentation/raw_data/DRIONS/images/image_001.jpg')
# height, width,_ = im.shape
# path = '/mnt/Almacenamiento/ODOC_segmentation/raw_data/DRIONS/experts_anotation/anotExpert1_001.txt'
# expert, img_name = get_mask_data(path)
# size = (height, width)
# mask = generate_mask(path,size,img_name, expert)
    
# final_img = im[:,:,0] * mask
# plt.imshow(final_img)
# plt.show()

if __name__ == '__main__':
    
    main()

    





