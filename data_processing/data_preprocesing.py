from configparser import ConfigParser
import configparser
#from operator import getitem
from os import path
from turtle import st
import torch
#from ssl import OP_ENABLE_MIDDLEBOX_COMPAT
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pytorch_lightning as pl
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
#import torch.nn.functional as nnf
from torchvision.io import read_image
#from torchvision.utils import make_grid
import glob
#import os
#import torch
import ntpath
import matplotlib.pyplot as plt
#import cv2

def show(imgs):

    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def list_2_string(input_l):
    '''
    convert the list into a single string separate by ,
    used for generate the split file '''

    if (input_l == None) or (len(input_l) == 0):
        return ''
    else:
        return ','.join(list(input_l))

# def arr_2_string(input_l):
#     'convert the list into a single string separate by ,'
#     if (input_l == None) or (len(input_l) == 0):
#         return ''
#     else:
#         return ','.join(list(input_l))

def config_file_write(dst, filename, dataset):
    ''' 
    Create the config file for the training
    -----------
    input
    dst: the destination were to save the file
    dataset: the name of the dataset to train
    '''

    #CREATE OBJECT
    config_file = configparser.ConfigParser()

    #ADD SECTION
    config_file.add_section("Model")

    #ADD SETTING TO SECTION
    config_file.set("Model", "n_classes", '2')
    config_file.set("Model", "is_deconv", 'False')
    config_file.set("Model", "is_BN", 'True')
    config_file.set("Model", "dropout", '0.0')
    config_file.set("Model", "use-otsu", 'False')
    config_file.set("Model", "filters_encoder", arr_2_string(['64', '128', '256', '512', '1024']))
    config_file.set("Model", "filters_decoder", arr_2_string(['64', '128', '256', '512']))
    config_file.set("Model", "activation", 'relu')
    config_file.set("Model", "pooling", 'max')

    #SAVE CONFIG FILE
    with open(dst + dataset + '_configuration.ini','w') as configfileObj:
        config_file.write(configfileObj)
        configfileObj.flush()
        configfileObj.close()
    
    print("Config file 'configurations.ini' created")


def save_split_file(dst, filename, dataset, train, validation, test):
    '''
    Guardo la separacion entre training, validation y test en archivo .ini
    ----- 
    Input
    dst: path donde guardar el archivo
    filename: nombre del proyecto
    dataset: dataset de donde provienen las imagenes
    train: lista de nombres de las imagenes separadas para training
    validation: lista de nombre de las imagenes separadas para validacion
    test: lista de nombres de las imagenes separadas para el test

    ---
    Output
    split file
    '''

    if '' in train:
        train.remove('')
    if '' in validation:
        validation.remove('')
    if '' in test:
        test.remove('')

    split = ConfigParser()
    split.add_section('split')
    split.set('split','type','holdout')
    split.set('split','training', list_2_string(train))
    split.set('split', 'validation', list_2_string(validation))
    split.set('split','test',list_2_string(test))

    #SAVE THE FILE
    split_file = open(path.join(dst,filename + '_' + dataset + '.ini'),'w')
    split.write(split_file)
    split_file.close()

    #IMPRIMO LA CANTIDAD DE IMAGENES PERTENECIENTES A CADA GRUPO
    print(' - Training: {} images'.format(len(train)))
    print(' - Validation: {} images'.format(len(validation)))
    print(' - Test: {} images'.format(len(test)))

    return split

class DataModuleClass(pl.LightningDataModule):
    '''
    MODULO THE DATASET NECESARIO PARA EL ENTRENAMIENTO  CON PYTORCH LIGHTNING
    INDICO COMO SE LEVANTAN LAS IMAGENES Y GENERO LOS DATALODER TANTO PARA ENTRENAMIENTOS, VALIDACION Y TEST
    '''
    def __init__(self, data_path, split_file, dataset, num_workers=4):
        '''
        INPUTS
        data_path: path donde se encuentran los datos imagen/mascara
        splt_file: archivo split que nos indica la division entre entrenamiento, validacion y test
        dataset: nombre del dataset utilizado
        num_workers: nmero de workers a utilizar
        '''
        super().__init__()#train_transforms, val_transforms, test_transforms, dims
        
        #BATCH_SIZE
        self.batch_size=5
        #transform a utilizar, despues pasados por parametro por ahi
        self.transform = transforms.Compose([transforms.Normalize(mean=0,std=1)])

        self.data_path = data_path 
        self.split_file = split_file         
        self.dataset= dataset
        self.num_workes = num_workers

    
    def prepare_data(self):
        '''Define steps that should be done on only one GPU, like getting data
        Usualmente se utiliza para el proceso de descagar el dataset'''
 
        return

    def setup(self, stage=None):
        '''
        Define steps that shouls be done in every GPU, like splitting data
        applying transforms, etc.
        Usually used to handle the task of loading the data'''

        #OBTENGO EL NOMBRE DE LOS ARCHIVOS PERTENECIENTE A CADA GRUPO
        #YA SEA ENTRENAMIENTO, VALIDACION O TEST
        #transformo el string del .ini en una arreglo de string
        tr_names = np.array((self.split_file['split']['training']).split(','))
        val_names = np.array((self.split_file['split']['validation']).split(','))
        test_names = np.array((self.split_file['split']['test']).split(','))
        
        
        if len(val_names) < 2:
            #EN CASO DE NO TENER VALORES PARA VALIDACION SE SEPARA EL 10% DE LOS
            #VALORES DE ENTRENAMIENTO PARA VALIDAR

            n_val = int(len(tr_names) * 0.1)
            n_train = len(tr_names) - n_val
            val_names = tr_names[n_train:]
            tr_names = tr_names[:n_train]

        #OBTENGO LOS PATHS THE IMAGENES Y MASCARAS
        img_paths = []
        OD_paths = []
        #OC_paths = []

        for path in sorted(glob.glob(self.data_path + '/images/' + self.dataset + '/*.png')):
            img_paths.insert(len(img_paths),path)

        for path in sorted(glob.glob(self.data_path + '/OD1/' + self.dataset + '/*.png')):
            OD_paths.insert(len(OD_paths),path)
                
        # for path in sorted(glob.glob(self.data_path + '/OC/' + self.dataset + '/*.png')):
        #     OC_paths.insert(len(OC_paths),path)


        #CREO QLE DATASET PARA LOS VALORES DE ENTRENAMIENTO VALIDACION Y TEST
        self.train_data = Dataset_proc(img_paths, OD_paths, tr_names)
        self.valid_data = Dataset_proc(img_paths, OD_paths, val_names)
        self.test_data = Dataset_proc(img_paths, OD_paths, test_names)

        return

    def train_dataloader(self):
        '''return Dataloader for Training data here'''
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workes)

    def val_dataloader(self):
        ''' return the DataLoader for Validation Data here'''
        return DataLoader(self.valid_data, batch_size=self.batch_size, num_workers=self.num_workes)

    def test_dataloader(self):
        '''Return DataLoader for Testing Data here'''
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workes)


class Dataset_proc(Dataset):
    def __init__(self,img_path, OD_path, split, augmentacion=None, scale=1):
        '''
        ---------------------------------
        input
        img_path: lista con los paths de las imagenes
        OD_path: lista con los paths de las mascaras de OD
        split: nombre base de las imagenes pertenecientes al dataset
        augmentacion: la augmentacion a aplicarse sobre los datos
        ---------------------------------
        '''

        paths = []
        masks= []

        #OBTENGO LOS PATHS THE LAS IMAGENES QUE PERTENECEN AL GRUPO QUE QUIERO
        for i in range(len(img_path)): 
            base_name = ntpath.basename(img_path[i])
            if base_name in split: 
                paths.insert(len(paths),img_path[i])
                masks.insert(len(masks),OD_path[i])


        self.paths = paths
        self.masks = masks
        self.names = split

        self.augmentation = augmentacion

    
    def __len__(self):
        '''
        Devueve el tamano de datos pertenecientes al dataset
        '''
        return len(self.paths)

    def __getitem__(self, index):

        #LEO LA IMAGEN Y LA MASCARA, DEVUELVE 
        # EN FORMATO TENSOR
        img = read_image(self.paths[index])
        OD_mask = read_image(self.masks[index])

        #DEJO LA MASCARA SOLO CON VALORES 0|1
        OD_mask[OD_mask>0] = 1

        #APLICO LAS AUGMENTACIONES EN CASO DE HABER
        if self.augmentation != None:
            print('aumentacion')

        img = img.float()

        #NORMALIZO LA IMAGEN, 
        #Obtengo media y desvio por cada imagen y normalizo la imagen
        #posteriormente la resizeo a un valor mas chico de 512,512
        mean_r = torch.mean(img[0,:,:])
        mean_g = torch.mean(img[1,:,:])
        mean_b = torch.mean(img[2,:,:])

        std_r = torch.mean(img[0,:,:])
        std_g = torch.mean(img[1,:,:])
        std_b = torch.mean(img[2,:,:])

        img = F.normalize(img, [mean_r, mean_g, mean_b], [std_r,std_g, std_b]) #normalizo la imagen

        #RE-ESCALO LAS IMAGENES A 512X512 
        img = self.scale_img(img, 512, 512)
        mask = self.scale_img(OD_mask, 512, 512)

        mask = mask[0,:,:]
    
        return img.float(), mask.long()

    def scale_img(self,image, width, height):
        ''' 
        retorna la imagen re escalada
        '''
        scaledImg = F.resize(image, (width,height), interpolation=transforms.InterpolationMode.NEAREST)
        return scaledImg




##########################  test  ###############3
        
# split = '/mnt/Almacenamiento/ODOC_segmentation/split/ODOC_segmentation_IDRID.ini'
# config = configparser.ConfigParser()
# config.read(split)
# data = DataModuleClass('/mnt/Almacenamiento/ODOC_segmentation/data',config,'IDRID')
# data.setup()
# print(data)
# data_iter = iter(data)
# x,y,z = data_iter.next()

# fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
# ax0.imshow(x)
# ax1.imshow(y)
# ax2.imshow(z)


# plt.show()

#config_file_write('/mnt/Almacenamiento/ODOC_segmentation/codigo/','','DRISHTI')