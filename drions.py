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

def get_mask_data(path):
    name = ntpath.basename(path)
    n = name.split('.')
    sub_n = n[0].split('Expert')
    expert, img_n = sub_n[1].split('_')
    print('Expert: ', expert, ' img: ', img_n)

    return expert, img_n

def generate_mask(mask_path,img_size,img_name, expert):

    mask = np.loadtxt(mask_path,delimiter=',')

    #Repito el primer valor para cerrar el circulo
    new_row = mask[0,:]
    mask = np.vstack([mask, new_row]) 

    # mask_bs = np.zeros(size)

    img = np.zeros(size, dtype=int)
    rr, cc = polygon(mask[:,1], mask[:,0])
    img[rr,cc] = 1
    
    iio.imwrite(proyect_path + dst_data_path + 'OD' + expert + '/' + dataset + img_name + '.png', img)

    

proyect_path = '/mnt/Almacenamiento/ODOC_segmentation'
or_data_path = '/raw_data/'
dst_data_path = '/data/'
dataset= 'DRIONS/'


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
    print(j)
    expert, img_name = get_mask_data(j)
    generate_mask(j,size,img_name, expert)


# expert, img_name = get_mask_data(anotations[0])
# print(size)
# generate_mask(anotations[0],size,img_name, expert)


