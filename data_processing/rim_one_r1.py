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
dataset = 'RIM_ONE_R1'

def generate_mask(masks,name,expert):
    '''
    Se genera la maskara a martir del majority voting entre los 5 expertos
    igual que en el otro dataset se tomara los valores donde 3 o mas expertos entivieron de acuerdo
    el resto se almacena en las mascaras extras por separado
    ------ 
    input:

    masks= lista con las mascara de los 5 expertos, la misma cuenta con valores 0|1
    '''
    OD_add = np.zeros(masks[0].shape)
    for m_idx in range(len(masks)):
        OD_add = np.add(masks,masks[m_idx]) #sumo de a uno las matrices de mascara que se encuentra en la lista
        iio.imwrite(proyect_path + dst_data_path + 'OD_extra/' + dataset + '/' + name + '_' + expert[m_idx] + '.png',masks[m_idx]) #una vez sumada se guarda con le nombre del experto
    
    OD_add[OD_add < 3] = 0
    OD_add[OD_add >= 3] = 255
    iio.imwrite(proyect_path + dst_data_path + 'OD1/' + dataset + '/' + name + '.png',masks[m_idx]) #una vez sumada se guarda con le nombre del experto

    
    return

def get_mask(paths):
    same_img = []
    expert_list=[]
    last_name = ''
    for p in paths:
        base_name = ntpath.basename(p)
        n = base_name.split('.')
        name = n[0].replace('Im','')
        name,expert = name.split('-')

        mask = iio.imread(p)
        mask[mask==255] = 1 #solo para facilitar el majority voting

        # fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
        # ax0.imshow(img)
        # ax1.imshow(mask)
        # ax2.imshow(img[:,:,0] * mask)

        # plt.show()

        if name != last_name: #si el nombre cambio es porque estamos trabajando sobre una nueva imagen
            if same_img != []: #si la lista esta vacia es porque es la primer imagen que se trabaja. Ver de agregar el ultimo caso
                generate_mask(same_img,last_name,expert_list) #genero el majority voting
                same_img = []   #reinicio la de mascaras
                expert_list = []    #reinicio la lista de expertos
            
            last_name = name    #actualizo el nombre actual, tanto para el primer caso como para cuando cambia de imagen a trabajar


        same_img.insert(len(same_img),mask) #agrego el nuevo dato
        expert_list.insert(len(expert_list),expert)

    generate_mask(same_img,last_name,expert_list) #para que se genere la ultima mascara cuando se corte el for



def get_images(paths):
    for p in paths:
        base_name = ntpath.basename(p)
        n = base_name.split('.')
        name = n[0].replace('Im','')
        
        img = iio.imread(p) #obtengo imagen
        #procesar
        iio.imwrite(proyect_path+dst_data_path+'images/' + dataset  + '/' +  name + '.png',img) #guardo



def main():
    img_name = 1
    img_paths = []
    final_img_name = []
    mask_paths = []

    for tr_img in sorted(glob.glob(proyect_path + or_data_path + dataset + '/*/*.bmp')):
        name = ntpath.basename(tr_img)
        n = name.split('.')
        names = n[0]
        if 'exp' in names:
            mask_paths.insert(len(mask_paths), tr_img)
        else:
            img_paths.insert(len(img_paths), tr_img)
    
    get_images(img_paths)
    get_mask(mask_paths)

if __name__ == '__main__':
    main()


# img_path = '/mnt/Almacenamiento/ODOC_segmentation/data/images/RIM_ONE_R1/009.png'
# #OC_path = '/mnt/Almacenamiento/ODOC_segmentation/data/OC/RIGA-Magrabia/001.png'
# OD_path = '/mnt/Almacenamiento/ODOC_segmentation/data/OD1/RIM_ONE_R1/009.png'

# OD_ex1 = '/mnt/Almacenamiento/ODOC_segmentation/data//OD_extra//RIM_ONE_R1/009_exp1.png'
# OD_ex2 = '/mnt/Almacenamiento/ODOC_segmentation/data//OD_extra//RIM_ONE_R1/009_exp2.png'
# OD_ex3 = '/mnt/Almacenamiento/ODOC_segmentation/data//OD_extra//RIM_ONE_R1/009_exp3.png'
# OD_ex4 = '/mnt/Almacenamiento/ODOC_segmentation/data//OD_extra//RIM_ONE_R1/009_exp4.png'
# OD_ex5 = '/mnt/Almacenamiento/ODOC_segmentation/data//OD_extra//RIM_ONE_R1/009_exp5.png'


# img = iio.imread(img_path)
# OD = iio.imread(OD_path)

# OD_ex1 = iio.imread(OD_ex1)
# OD_ex2 = iio.imread(OD_ex2)
# OD_ex3 = iio.imread(OD_ex3)
# OD_ex4 = iio.imread(OD_ex4)
# OD_ex5 = iio.imread(OD_ex5)



#print(collections.Counter(OD[800,:])) #255, 0 y 128 para las marcas


# fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4)
# ax0.imshow(img)
# ax1.imshow(OD)
# ax2.imshow(OD_ex1+OD_ex2+OD_ex2+OD_ex3+OD_ex4+OD_ex5)


# ax3.imshow(img[:,:,0] * (OD))

# plt.show()
    