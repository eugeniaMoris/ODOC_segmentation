from curses.textpad import rectangle
import numpy as np
import matplotlib.pyplot as plt
#from PIL import Image, ImageDraw
import glob
import ntpath
#import matplotlib.patches as patches
import imageio as iio
from skimage.draw import polygon
from scipy import ndimage


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
    
    iio.imwrite(proyect_path + dst_data_path + 'OD' + expert + '/' + dataset + img_name + '.png', img)

    
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

    for anot in glob.glob(proyect_path + or_data_path + dataset + 'experts_anotation/anot*.txt'):
        anotations.insert(len(anotations), anot)

    for img_p in glob.glob(proyect_path + or_data_path + dataset + 'images/*.jpg'):
        imgs_paths.insert(len(imgs_paths), img_p)

    for i in imgs_paths:
        name = ntpath.basename(i)
        n = name.split('.')
        im = iio.imread(i)
        height, width,_ = im.shape
        iio.imwrite(proyect_path+dst_data_path+'images/' + dataset + n[0] + '.png',im)
        #print(proyect_path + dst_data_path + dataset + 'images/' + n[0] + '.png')

    size = (height, width)
    for j in anotations:
        expert, img_name = get_mask_data(j)
        generate_mask(j,size,img_name, expert)

    #retorna nombre dataset, paths train ,valid, test para agregar al .ini
    #return 'DRIONS', train_names, valid_names, test_names
    return 'DRIONS'



    
def if __name__ == '__main__':
    main()
    




# expert, img_name = get_mask_data(anotations[0])
# print(size)
# generate_mask(anotations[0],size,img_name, expert)


