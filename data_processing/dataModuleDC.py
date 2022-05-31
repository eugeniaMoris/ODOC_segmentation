import glob
import torch
import ntpath
import numpy as np
from PIL import Image
import pytorch_lightning as pl
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader
from .utils import get_OCOD_crop

def getpaths_all(data_path,tr_names):
    names = []
    path = []
    OD_path = []
    OC_path = []
    for p in tr_names:
        new_p = data_path + '/images/' + p
        new_od_p = data_path + '/OD1/' + p
        new_oc = data_path + '/OC/' + p
        path.append(new_p)
        OD_path.append(new_od_p)
        OC_path.append(new_oc)
    return path,OD_path,OC_path

class DataModuleClassDC(pl.LightningDataModule):
    '''
    MODULO THE DATASET NECESARIO PARA EL ENTRENAMIENTO  CON PYTORCH LIGHTNING
    INDICO COMO SE LEVANTAN LAS IMAGENES Y GENERO LOS DATALODER TANTO PARA ENTRENAMIENTOS, VALIDACION Y TEST
    EN ESTE MODULO SE TIENE EN CUENTA LA OBTENCION DE AMBAS MASCARAS DISCO Y COPA Y A SU VEZ EN EL GET ITEM DEL DATASET SE RELIZA UN RECORTE DE LA IMAGEN DADA UNA MASCARA
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

        #OBTENGO EL NOMBRE DE LOS ARCHIVOS PERTENECIENTE A CADA GRUPO
        #YA SEA ENTRENAMIENTO, VALIDACION O TEST
        #transformo el string del .ini en una arreglo de string
        tr_names = np.array((self.split_file['split']['training']).split(','))
        val_names = np.array((self.split_file['split']['validation']).split(','))
        test_names = np.array((self.split_file['split']['test']).split(','))
        
        
        if len(val_names.tolist()) < 1:
            #EN CASO DE NO TENER VALORES PARA VALIDACION SE SEPARA EL 10% DE LOS
            #VALORES DE ENTRENAMIENTO PARA VALIDAR

            n_val = int(len(tr_names) * 0.1)
            n_train = len(tr_names) - n_val
            val_names = tr_names[n_train:]
            tr_names = tr_names[:n_train]
        #print('DATOS: train ', len(tr_names), ' validation: ', len(val_names), ' test: ', len(test_names))

        if self.dataset ==  'multi':
            paths_tr,od_tr,oc_tr = getpaths_all(self.data_path,tr_names)
            paths_v,od_v,oc_v = getpaths_all(self.data_path,val_names)
            paths_test,od_test,oc_test = getpaths_all(self.data_path,test_names)
            #puede mejorarse para que siempre sea asi pero mas adelante se hara
            
            self.train_data = Dataset_DC(paths_tr, OD_path= od_tr,OC_path=oc_tr, split= tr_names, dataset = self.dataset, made_norm=self.made_norm, augmentacion=self.transform, probabilities= self.probabilities,multi_data=True)

            self.valid_data = Dataset_DC(paths_v, OD_path= od_v,OC_path=oc_v, split= val_names, dataset = self.dataset, made_norm=self.made_norm, augmentacion=None,multi_data=True)
            
            self.test_data = Dataset_DC(paths_test, OD_path= od_test,OC_path=oc_test, split= test_names, dataset = self.dataset, made_norm=self.made_norm, augmentacion=None,multi_data=True)
        else:
            #OBTENGO LOS PATHS THE IMAGENES Y MASCARAS
            img_paths = []
            OD_paths = []
            OC_paths = []

            for path in sorted(glob.glob(self.data_path + '/images/' + self.dataset + '/*.png')):
                img_paths.insert(len(img_paths),path)
            for path in sorted(glob.glob(self.data_path + '/images/' + self.dataset + '/*/*.png')):
                img_paths.insert(len(img_paths),path)
            self.img_paths = img_paths

            for path in sorted(glob.glob(self.data_path + '/OD1/' + self.dataset + '/*.png')):
                OD_paths.insert(len(OD_paths),path)
            for path in sorted(glob.glob(self.data_path + '/OD1/' + self.dataset + '/*/*.png')):
                OD_paths.insert(len(OD_paths),path)
            for path in sorted(glob.glob(self.data_path + '/OC/' + self.dataset + '/*.png')):
                OC_paths.insert(len(OC_paths),path)
            for path in sorted(glob.glob(self.data_path + '/OC/' + self.dataset + '/*/*.png')):
                OC_paths.insert(len(OC_paths),path)

            
            #print('len OD_paths: ', len(OD_paths), ' len OC_paths: ', len(OC_paths), ' len img_paths ', len(img_paths))
                    

            #CREO QLE DATASET PARA LOS VALORES DE ENTRENAMIENTO VALIDACION Y TEST
            self.train_data = Dataset_DC(img_paths,OD_path= OD_paths,OC_path=OC_paths, split= tr_names, dataset = self.dataset, made_norm=self.made_norm, augmentacion=self.transform, probabilities= self.probabilities)
            self.valid_data = Dataset_DC(img_paths,OD_path= OD_paths, OC_path=OC_paths, split= val_names, dataset= self.dataset, made_norm=self.made_norm, augmentacion=None) #EN VALICACION TAMBIRN?
            self.test_data = Dataset_DC(img_paths,OD_path= OD_paths, OC_path=OC_paths, split= test_names, dataset= self.dataset, made_norm=self.made_norm, augmentacion=None)
        
            if (self.pred == 'train'):
                self.pred_data = Dataset_DC(img_paths, OD_path= OD_paths, OC_path=OC_paths, split= tr_names, dataset = self.dataset, made_norm=self.made_norm, augmentacion=None)

            if (self.pred == 'valid'):
                self.pred_data = Dataset_DC(img_paths, OD_path= OD_paths, OC_path=OC_paths, split= val_names, dataset = self.dataset, made_norm=self.made_norm, augmentacion=None)
            
            if (self.pred == 'test'):
                self.pred_data = Dataset_DC(img_paths, OD_path= OD_paths, OC_path=OC_paths, split= test_names, dataset = self.dataset, made_norm=self.made_norm, augmentacion=None)
        
            if (self.pred == 'total'):
                self.pred_data = Dataset_DC(self.img_paths, self.dataset, made_norm=self.made_norm, augmentacion=None)

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

class Dataset_DC(Dataset):
    def __init__(self,img_path, dataset, made_norm=True, OD_path= None, OC_path=None, split= [], augmentacion=None,probabilities=None,rescale='normal', scale=1,multi_data=False):
        '''
        ---------------------------------
        input
        img_path: lista con los paths de las imagenes
        dataset: nos da el nobre del dataset utilizado
        made_norm = nos indica si debemos aplicar la normalizacion sobre las imagenes o no
        OD_path: lista con los paths de las mascaras de OD
        OC_path: lista con los paths de las mascaras de OC
        split: nombre base de las imagenes pertenecientes al dataset
        augmentacion: la augmentacion a aplicarse sobre los datos
        probabilities: nos indica las probabilidades de que se aplica cada augmentacion
        rescale: normal/test, para saber que tipo de reescalado se le deben realizar a las imagenes
        ---------------------------------
        '''

        paths = []
        OD_masks= []
        OC_masks= []
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
                    OD_masks.append(OD_path[i])
                    OC_masks.append(OC_path[i])


            #MEJORAR ESTE CODIGO el all names
            self.paths = paths
            self.OD_masks = OD_masks
            self.OC_masks = OC_masks
            self.names = split

            self.augmentation = augmentacion
        else:
            self.paths = img_path
            self.OD_masks = OD_path
            self.OC_masks = OC_path
            self.names = split  #SE PUEDEN CONSEGUIR SI SE QUIERE

            print('len OD_paths: ', len(self.OD_masks), ' len OC_paths: ', len(OC_masks), ' len img_paths ', len(img_path))

    
    def __len__(self):
        '''
        Devueve el tamano de datos pertenecientes al dataset
        '''
        return len(self.paths)

    def __getitem__(self, index):

        #LEO LA IMAGEN Y LA MASCARA, DEVUELVE 
        # EN FORMATO TENSOR


        img = Image.open(self.paths[index])

        shape = img.size
        print( 'íNDEX', index, self.names[index])

        OD_mask = Image.open(self.OD_masks[index])
        print("PATH: ", self.paths[index])
        OC_mask = Image.open(self.OC_masks[index])
            
        

        #APLICO LAS AUGMENTACIONES EN CASO DE HABER
        if self.transform:
            print('ENTRO A TRANSFORMATION')

            #DARLE UNA PROBABILIDAD DE QUE NO TOME TODAS LAS TRANSFOMRACIONES
            #PREGUNTAR QUE SI NO HAY PROBABILIDADES SE APLICA TODOS
            
            probabilities = self.probabilities
            all_transforms = self.transform
            
            #apply an array of probability for the selections of the augmentation to do or not
            for i in range(len(probabilities)):
                rand = torch.rand(1)
                if float(rand) < probabilities[i]:
                    tr = all_transforms[i]
                    #,img, mask1, mask2=None
                    img, OD_mask, OC_mask = tr(img,OD_mask,OC_mask)
                    #print('SHAPE IMG AFTER TR: ', img.size)
        else:
            transform = transforms.ToTensor() # YA DEJA LA IMAGEN ENTRE CERO Y UNO
            img = transform(img)
            if OD_mask:
                OD_mask = transform(OD_mask)
            if OC_mask:
                OC_mask = transform(OC_mask)
        
        #print('FIANLIZO EL AUGMENTACION')


        if self.made_norm:
            #print('ENTRO A NORMALIZATION')

            #NORMALIZO LA IMAGEN, 
            #Obtengo media y desvio por cada imagen y normalizo la imagen
            #posteriormente la resizeo a un valor mas chico de 512S,512

            img = img.float()
            shape = img.size() #Guardo los tamaños originales

            mean_r = torch.mean(img[0,:,:])
            mean_g = torch.mean(img[1,:,:])
            mean_b = torch.mean(img[2,:,:])

            std_r = torch.std(img[0,:,:])+1e-6
            std_g = torch.std(img[1,:,:])+1e-6
            std_b = torch.std(img[2,:,:])+1e-6
            #print('ANTES DE NORMALIZAR: ', img.size())

            img = F.normalize(img, [mean_r, mean_g, mean_b], [std_r,std_g, std_b]) #normalizo la imagen

        #obtengo la zona donde se ubica el disco de la imagen para recortar al borde mas un delta
        #la imagen devuelta es cuadrada teniendo en cuenta el lado mas largo si es mas alto o mas corto
        #print('ENTRO A RECORTE')
        c0, c1, c2, c3, center = get_OCOD_crop(mask= OD_mask[0,:,:], delta= 0.1, descentro=True)
        img = img[:,c0:c2,c1:c3]
        print('IMG SHAPE: ',img.shape)

        #RE-ESCALO LAS IMAGENES A 512X512 
        img = self.scale_img(img, 512, 512)
        OD_mask = OD_mask[:,c0:c2+1,c1:c3+1]
        OD_mask = self.scale_img(image= OD_mask, width= 512, height= 512)
        OD_mask = OD_mask[0,:,:]

        OC_mask = OC_mask[:,c0:c2,c1:c3]
        #,image, width, height
        OC_mask = self.scale_img(image= OC_mask, width= 512, height= 512)
        OC_mask = OC_mask[0,:,:]
            
        mask = OD_mask
        mask = mask  + OC_mask
        #print('values of the mask: ', torch.unique(mask))


        #print('Retorno get item: ',img.float().shape, mask.long().shape, self.names[index], shape)
        return img.float(), mask.long(), self.names[index], shape

    def scale_img(self,image, width, height):
        ''' 
        retorna la imagen re escalada
        '''
        print('IMG SHAPE IN SCALE IMG: ', image.shape)
        scaledImg = F.resize(image, (width,height), interpolation=transforms.InterpolationMode.NEAREST)
        return scaledImg


    