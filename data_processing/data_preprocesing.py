from configparser import ConfigParser
import configparser
from distutils.command.config import config
#from operator import getitem
from os import path
from posixpath import split
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
import imageio as iio
from PIL import Image
from skimage.transform import resize
from .utils import crop_fov_limits
from skimage import filters, measure

def getpaths(data_path,tr_names):
    names = []
    path = []
    OD_path = []
    for p in tr_names:
        new_p = data_path + '/images/' + p
        new_od_p = data_path + '/OD1/' + p
        path.append(new_p)
        OD_path.append(new_od_p)
    return path,OD_path

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

def arr_2_string(input_l):
    'convert the list into a single string separate by ,'
    if (input_l == None) or (len(input_l) == 0):
        return ''
    else:
        return ','.join(list(input_l))

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
    config_file.set("Model", "filters_encoder", list_2_string(['64', '128', '256', '512', '1024']))
    config_file.set("Model", "filters_decoder", list_2_string(['64', '128', '256', '512']))
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
    def __init__(self, data_path,  dataset, aumentation, pred_data='total', norm=True, probabilities= None, batch_size= 5,split_file=None, num_workers=8):
        '''
        INPUTS
        data_path: path donde se encuentran los datos imagen/mascara
        dataset: nombre del dataset utilizado
        aumentation: transforms to apply in the datasets
        pred_data : nos dice de que queremos generar los datos de prediccion, los mismos pueden ser train, valid, test, total
        norm : la variable indica si se debe aplicar la normalizacion sobre la imagen o no
        probabilities: probabilities of each transform to be apply. list with the same len than aumentation
        splt_file: archivo split que nos indica la division entre entrenamiento, validacion y test
        num_workers: nmero de workers a utilizar
        '''
        super().__init__()#train_transforms, val_transforms, test_transforms, dims
        
        #BATCH_SIZE
        self.batch_size=batch_size
        #transform a utilizar, despues pasados por parametro por ahi
        self.transform = aumentation
        self.probabilities = probabilities

        self.data_path = data_path 
        self.split_file = split_file         
        self.dataset= dataset
        self.num_workes = num_workers
        self.pred = pred_data
        self.made_norm = norm

    
    def prepare_data(self):
        '''Define steps that should be done on only one GPU, like getting data
        Usualmente se utiliza para el proceso de descagar el dataset'''
 
        return

    def setup(self, stage=None):
        '''
        Define steps that shouls be done in every GPU, like splitting data
        applying transforms, etc.
        Usually used to handle the task of loading the data'''
        tr_names = np.array((self.split_file['split']['training']).split(','))
        val_names = np.array((self.split_file['split']['validation']).split(','))
        test_names = np.array((self.split_file['split']['test']).split(','))
        if self.dataset != 'multi':

            #OBTENGO EL NOMBRE DE LOS ARCHIVOS PERTENECIENTE A CADA GRUPO
            #YA SEA ENTRENAMIENTO, VALIDACION O TEST
            #transformo el string del .ini en una arreglo de string
            
            
            list_val = val_names.tolist()
            if len(val_names) < 2:
                #EN CASO DE NO TENER VALORES PARA VALIDACION SE SEPARA EL 10% DE LOS
                #VALORES DE ENTRENAMIENTO PARA VALIDAR
                print('ENTRO POR NO TENER DATOS DE VALIDACION')

                n_val = int(len(tr_names) * 0.1)
                n_train = len(tr_names) - n_val
                val_names = tr_names[n_train:]
                tr_names = tr_names[:n_train]

        if self.dataset ==  'multi':
            img_paths = []
            OD_paths = []
            paths_tr,od_tr = getpaths(self.data_path,tr_names)
            paths_v,od_v = getpaths(self.data_path,val_names)
            paths_test,od_test = getpaths(self.data_path,test_names)
            #puede mejorarse para que siempre sea asi pero mas adelante se hara

            self.train_data = Dataset_proc(paths_tr, OD_path= od_tr, split= tr_names, dataset = self.dataset, made_norm=self.made_norm, augmentacion=self.transform, probabilities= self.probabilities,multi_data=True)

            self.valid_data = Dataset_proc(paths_v, OD_path= od_v, split= val_names, dataset = self.dataset, made_norm=self.made_norm, augmentacion=None,multi_data=True)
            
            self.test_data = Dataset_proc(paths_test, OD_path= od_test, split= test_names, dataset = self.dataset, made_norm=self.made_norm, augmentacion=None,multi_data=True)

        else:
            #OBTENGO LOS PATHS THE IMAGENES Y MASCARAS
            print('ENTRO ACA')
            print('VALOR DEL DATASET: ', self.dataset)
            img_paths = []
            OD_paths = []

            for path in sorted(glob.glob(self.data_path + '/images/' + self.dataset + '/*.png')):
                img_paths.insert(len(img_paths),path)
            for path in sorted(glob.glob(self.data_path + '/images/' + self.dataset + '/*/*.png')):
                img_paths.insert(len(img_paths),path)
            self.img_paths = img_paths

            try: #EN CASO DE NO POSEER MASCARAS PARA EL DATASET 
                for path in sorted(glob.glob(self.data_path + '/OD1/' + self.dataset + '/*.png')):
                    OD_paths.insert(len(OD_paths),path)
                for path in sorted(glob.glob(self.data_path + '/OD1/' + self.dataset + '/*/*.png')):
                    OD_paths.insert(len(OD_paths),path)
            except:
                print('NO MASK FOR THIS DATASET')
            
                

            if self.split_file != None:
                print('DATOS: train ', len(tr_names), ' validation: ', len(val_names), ' test: ', len(test_names))
                #CREO QLE DATASET PARA LOS VALORES DE ENTRENAMIENTO VALIDACION Y TEST
                self.train_data = Dataset_proc(img_paths,OD_path= OD_paths, split= tr_names, dataset = self.dataset, made_norm=self.made_norm, augmentacion=self.transform, probabilities= self.probabilities)
                self.valid_data = Dataset_proc(img_paths,OD_path= OD_paths,  split= val_names, dataset= self.dataset, made_norm=self.made_norm, augmentacion=None) #EN VALICACION TAMBIRN?
                self.test_data = Dataset_proc(img_paths,OD_path= OD_paths, split= test_names, dataset= self.dataset, made_norm=self.made_norm, augmentacion=None)
            
                if (self.pred == 'train'):
                    self.pred_data = Dataset_proc(img_paths, OD_path= OD_paths, split= tr_names, dataset = self.dataset, made_norm=self.made_norm, augmentacion=None)

                if (self.pred == 'valid'):
                    self.pred_data = Dataset_proc(img_paths, OD_path= OD_paths, split= val_names, dataset = self.dataset, made_norm=self.made_norm, augmentacion=None)
                
                if (self.pred == 'test'):
                    self.pred_data = Dataset_proc(img_paths, OD_path= OD_paths, split= test_names, dataset = self.dataset, made_norm=self.made_norm, augmentacion=None)
            
                if (self.pred == 'total'):
                    self.pred_data = Dataset_proc(self.img_paths, self.dataset, made_norm=self.made_norm, augmentacion=None)
            else:
                print('Como no hay split file, se realiza el predict data con el total de las imagenes')
                self.pred_data = Dataset_proc(self.img_paths, self.dataset, made_norm=self.made_norm, augmentacion=None)
            

    def train_dataloader(self):
        '''return Dataloader for Training data here'''
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workes)

    def val_dataloader(self):
        ''' return the DataLoader for Validation Data here'''
        return DataLoader(self.valid_data, batch_size=self.batch_size, num_workers=self.num_workes)

    def test_dataloader(self):
        '''Return DataLoader for Testing Data here'''

        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workes)

    def predict_dataloader(self):
        ''' return the DataLoader for Prediction Data here'''

        return DataLoader(self.pred_data, batch_size=self.batch_size, num_workers=self.num_workes)

    def get_train_dataloader(self, batch_size):
        return DataLoader(self.train_data, batch_size= batch_size, num_workers=self.num_workes)

    def get_valid_dataloader(self, batch_size):
        return DataLoader(self.valid_data, batch_size= batch_size, num_workers=self.num_workes)
    
    def get_test_dataloader(self, batch_size):
        return DataLoader(self.test_data, batch_size= batch_size, num_workers=self.num_workes)
    
    def get_pred_dataloader(self, batch_size):
        return DataLoader(self.pred_data, batch_size= batch_size, num_workers=self.num_workes)

class Dataset_proc(Dataset):
    def __init__(self,img_path, dataset, made_norm=True, OD_path= None, split= [], augmentacion=None,probabilities=None,rescale='normal', scale=1,multi_data=False):
        '''
        ---------------------------------
        input
        img_path: lista con los paths de las imagenes
        dataset: nos da el nobre del dataset utilizado
        made_norm = nos indica si debemos aplicar la normalizacion sobre las imagenes o no
        OD_path: lista con los paths de las mascaras de OD
        split: nombre base de las imagenes pertenecientes al dataset
        augmentacion: la augmentacion a aplicarse sobre los datos
        probabilities: nos indica las probabilidades de que se aplica cada augmentacion
        rescale: normal/test, para saber que tipo de reescalado se le deben realizar a las imagenes
        ---------------------------------
        '''

        paths = []
        masks= []
        all_names=[]
        self.split = split
        self.dataset = dataset
        self.made_norm = made_norm
        self.transform = augmentacion
        self.probabilities = probabilities
        self.rescale= rescale

        #ARREGLO DE TRANSFORMACIONES 
        #ARREGLO DE PROBABILIDADES
        #SI RANDOM ES MEJOR A PROBABILIDAD SE AGREGA LA TRANSFORMACION AL COMPOSE

        if multi_data == False:
            #OBTENGO LOS PATHS THE LAS IMAGENES QUE PERTENECEN AL GRUPO QUE QUIERO
            for i in range(len(img_path)): 
                base_name = ntpath.basename(img_path[i])
                #print('BASE NAME: ', base_name)
                if (base_name in split)or (('Test/'+base_name) in split): 
                    #print('PATHS:', img_path[i])
                    paths.insert(len(paths),img_path[i])
                    masks.insert(len(masks),OD_path[i])

            #MEJORAR ESTE CODIGO el all names
            self.paths = paths
            self.masks = masks
            self.names = split

            self.augmentation = augmentacion
        else:

            self.paths = img_path
            self.masks = OD_path
            self.names = split  #SE PUEDEN CONSEGUIR SI SE QUIERE

    
    def __len__(self):
        '''
        Devueve el tamano de datos pertenecientes al dataset
        '''
        return len(self.paths)

    def __getitem__(self, index):

        #LEO LA IMAGEN Y LA MASCARA, DEVUELVE 
        # EN FORMATO TENSOR
        #print('index:' , index)
        img = Image.open(self.paths[index])
        mask= self.masks
        shape = img.size
        
        OD_mask = Image.open(self.masks[index])
            
        

        #APLICO LAS AUGMENTACIONES EN CASO DE HABER
        if self.transform:

            #DARLE UNA PROBABILIDAD DE QUE NO TOME TODAS LAS TRANSFOMRACIONES
            #PREGUNTAR QUE SI NO HAY PROBABILIDADES SE APLICA TODOS
            
            probabilities = self.probabilities
            all_transforms = self.transform
            
            #apply an array of probability for the selections of the augmentation to do or not
            for i in range(len(probabilities)):
                rand = torch.rand(1)
                if float(rand) < probabilities[i]:
                    tr = all_transforms[i]
                    img, OD_mask = tr(img,OD_mask)
        else:
            transform = transforms.ToTensor() # YA DEJA LA IMAGEN ENTRE CERO Y UNO
            img = transform(img)
            if mask != []:
                OD_mask = transform(OD_mask)


        if self.rescale == 'normal':
            #RE-ESCALO LAS IMAGENES A 512X512 
            img = self.scale_img(img, 512, 512)

            if mask != []:
                OD_new = self.scale_img(OD_mask, 512, 512)

                OD_new = OD_new[0,:,:]
                OD_new = OD_new.long()
            else:
                img = (img)
                OD_new = []
        else:
            img, OD_mask = self.rescale_test(img, OD_mask)

        if self.made_norm:
            #NORMALIZO LA IMAGEN, 
            #Obtengo media y desvio por cada imagen y normalizo la imagen
            #posteriormente la resizeo a un valor mas chico de 512S,512

            img = img.float()
            #shape = img.size() #Guardo los tamaños originales

            mean_r = torch.mean(img[0,:,:])
            mean_g = torch.mean(img[1,:,:])
            mean_b = torch.mean(img[2,:,:])

            std_r = torch.std(img[0,:,:])+1e-6
            std_g = torch.std(img[1,:,:])+1e-6
            std_b = torch.std(img[2,:,:])+1e-6

            img = F.normalize(img, [mean_r, mean_g, mean_b], [std_r,std_g, std_b]) #normalizo la imagen

    
        return img.float(), OD_new, self.names[index], shape

    def scale_img(self,image, width, height):
        ''' 
        retorna la imagen re escalada
        '''

        scaledImg = F.resize(image, (width,height), interpolation=transforms.InterpolationMode.NEAREST)
        return scaledImg
    
    def rescale_test(fundus_img, mask, target_radio=255):
        '''
        Para los datos de test buscamo reducir la imagen de manera tal que el radio del disco en la imagen de test se asimile al tamano del disco en la imagen de train
        fundu
        inputs: -----------------------------
        fundus_img: la imagen de entrada
        mask: mascara correspondiente a la imagen dada
        target_radio: tamano en el que esperamos que este el radio del disco de la imagen aproximadamente
        '''
        print('ENTRO: ')
        lim_x_inf,lim_x_sup, lim_y_inf,lim_y_sup, radio = crop_fov_limits(fundus_img)
            
        
        width, height,_ = fundus_img.shape    
        scale = target_radio / radio

        new_width = int(scale*width)
        new_height = int(scale*height)
  
        scaledImg = resize(fundus_img, (new_width,new_height))
        
        final_size = max(new_width,new_height)
        square_Img = resize(scaledImg,(final_size,final_size))
        if mask != []:
            y_rescale = resize(mask, (new_width,new_height))
            square_mask = resize(y_rescale,(final_size,final_size))
        else:
            square_mask = None
        
            
        return square_Img,square_mask






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