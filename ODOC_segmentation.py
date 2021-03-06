from ctypes import util
import glob
from tabnanny import verbose
from tkinter.messagebox import RETRY
import torch
import numpy as np
import torch.nn as nn
import imageio as iio
from turtle import forward
import csv
import ntpath
from zmq import device
from Models.Unet import Unet
import pytorch_lightning as pl
import torch.nn.functional as F
from argparse import ArgumentParser
from skimage.transform import resize
from configparser import ConfigParser
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import Trainer, seed_everything
from sklearn.metrics import f1_score,accuracy_score, precision_score, recall_score
from data_processing.utils import get_OCOD_crop
import torchvision.transforms as transforms
from skimage import filters, measure
import matplotlib.pyplot as plt
from scipy import ndimage
import os


from PIL import Image

from data_processing.utils import stage1_preprocessing, stage2_preprocessing, Hausdorff_distance
from data_processing.data_preprocesing import DataModuleClass
from data_processing.TestDataModule import TestDataModule



class ODOC_segmentation(pl.LightningModule):
    '''
    Modulo encargado de la segmentacion de imagen de fondo de ojos con la utilizacion de dos modelo
    Primero la imagen pasa por un modelo que dectecta OD
    Esa imagen despues es recortada y pasa por un segundo modelo que clasifica el disco y la copa'''
    def __init__(self, data_path, dataset,data_test, config, loss, model1, model2,path_save, save_test= False, otsu= False):
        '''
        model that pass thoght the two model to make and save the segmentation of the optic cup ans the optic disc
        -----------------------------------
        inputs-
        data_path: path of the data folder
        dataset: dataset used to train de models
        data_test: name of the dataset to segment
        config: config file
        loss: loss used by the models
        model1: model used in the first step to segment the optic disc over the entere image
        model2: model used in the second step o segment the optic disc and the optic cup over a cut image
        path_save: path in which we will save the dice results (in case is test stage) in a txt
        save_test
        otsu: True in case we want to use a otsu in the final segmentation of the optic disc'''
        super(ODOC_segmentation,self).__init__()
        self.loss = loss
        self.model_OD = model1
        self.model_DC = model2
        self.softmax = nn.Softmax(dim=1)
        self.lr = 0.1
        if 'lr' in config['training']:
            self.lr = float(config['training']['lr'])
        self.delta = 0.1
        if 'delta' in config['training']:
            self.delta = float(config['training']['delta'])
        self.data_path= data_path
        self.dataset = dataset
        self.data_test = data_test
        self.path_save = path_save
        self.save_test = save_test
        self.otsu = otsu

    
    def forward(self, inputs, shape, name, stage):
        '''
        Unet1 = segmenta sobre la imagen entera el disco
        Recorto la imagen en funcion de la segmentacion
        Unet2 = segmenta sobre la imagen recortada el disco y la copa'''

        w,h = shape
        x_pr = stage1_preprocessing(inputs.cpu()) #out shape [1,C,512,512]
        x_pr.to(self.device) #envio la imagen a la GPU
        x_pr = x_pr.type(torch.cuda.FloatTensor)

        #print('NAME: ', name)
        outOD = self.model_OD.forward(x_pr) #out shape [1,2,512,512]
        #print('SHAPE OF THE OUTPUT : ', outOD.shape)

        #outOD = torch.argmax(outOD,dim=1)
        outOD = self.softmax(outOD)
        #print('VALUES OF THE OUTPUT : ', torch.unique(outOD))

        c0, c1, c2, c3, sub_x, b_out = stage2_preprocessing(inputs.cpu(),outOD.cpu(),self.delta)
        #SUB IMG [1,3,512,512]

        #print('CUT IMAGE SHAPE',sub_x.shape)
        sub_x.to(self.device) #envio la imagen a la GPU
        sub_x = sub_x.type(torch.cuda.FloatTensor)
        
        out_2 = self.model_DC.forward(sub_x) #envio la imagen recortada segun la segmentacion de la etapa anterior
        #out_2 = torch.argmax(out_2,dim=1)
        out_2 = self.softmax(out_2) #aplico la softmax
        #print('SHAPE OF THE OUTPUT 2: ', out_2.shape)

        
        out_2 = out_2[0,:,:,:].cpu() #[1,3,h,w]
        y_hat = np.zeros((3,h.item(),w.item()))
        y_hat[:,c0:c2,c1:c3] = np.array(out_2)
        y_hat = np.transpose(y_hat, (1,2,0))

        #inputs.cpu()
        inputs = inputs[0,:,:,0].cpu()
        #print(' shape input', inputs.shape, h, w)
        # plt.imshow(np.array(inputs))
        # plt.imshow(y_hat,alpha=0.2)

        # plt.show()

        return y_hat, b_out


        #AGREGO LA DIMENSION BATCH SIZE
        #img= torch.unsqueeze(img, 0)
        #return 'yhat', 'intermediate result'


    def test_step(self, batch, batch_nb):

        tr = transforms.ToTensor()
        x, y, name, shape = batch #x shape [1,H,W,C]
        #print('NAME: ', name[0])
        

        y_hat, intermediate_OD = self.forward(x, shape, name,'test')
        #print('SHAPE Y_HAT: ', y_hat.shape, y.shape)
        y_hat = tr(y_hat)
        #pred = self.softmax(y_hat) #aplico la softmax
        
        #loss = self.loss(y_hat,y) 
        #print('loss: ', loss)
        y_arg = torch.argmax(y_hat,dim=0)
        # y = y.cpu()
        # plt.imshow(y_arg, alpha = 0.4)
        # plt.imshow(y[0,:,:],alpha=0.4)

        # plt.show()
        # print('SHAPE Y_HAT Tensor: ', y_arg.shape, y.shape)


        C_pred = y_arg.detach().clone()
        C_pred[C_pred == 1] = 0
        C_pred[C_pred == 2] = 1

        y_c = y.detach().clone()
        y_c[y_c == 1]=0 
        y_c[y_c == 2]=1 

        if self.otsu == True:
            y_otsu = np.array(y_hat[1,:,:])
            thresh = filters.threshold_otsu(y_otsu) #USAMOS OTSU PARA LA GENERACION DE IMAGENES
            binary = y_otsu > thresh
            D_pred = ndimage.binary_fill_holes(binary).astype(int)
        else:
            D_pred = y_arg.detach().clone() #OTSU
            D_pred[D_pred == 2] = 1

        y_d = y.detach().clone()
        y_d[y_d == 2] = 1

        self.save_metrics(y_c[0,:,:],y_d[0,:,:],C_pred,D_pred,intermediate_OD,name[0])

        dice_C = self.dice_metric(y_c[0,:,:],C_pred)
        dice_D = self.dice_metric(y_d[0,:,:],D_pred)
        dice_Dint = self.dice_metric(y_d,intermediate_OD)

        dice_C= torch.tensor(dice_C)
        dice_D= torch.tensor(dice_D)
        dice_Dint = torch.tensor(dice_Dint)

        dice_prom = (dice_C + dice_D)/2

        #print('Dice c: ', dice_C, 'Dice_d: ', dice_D, 'dice prom: ', dice_prom)

        if self.save_test == True:
            self.save_predictions(name[0],D_pred,C_pred,intermediate_OD)

        self.log('DICE OC', dice_C)
        self.log('DICE OD', dice_D)
        self.log('DICE AVG', dice_prom)
        self.log('DICE int', dice_Dint)

        
        return{'dice_OC':dice_C, 'dice_OD':dice_D, 'dice_prom':dice_prom, 'dice_ODint': dice_Dint}
        

    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        tr = transforms.ToTensor()
        x, y, name, shape = batch #x shape [1,H,W,C]
        print('NAME: ', name[0])
        

        y_hat, intermediate_OD = self.forward(x, shape, name,'test')
        y_hat = tr(y_hat)
        
        y_arg = torch.argmax(y_hat,dim=0)

        C_pred = y_arg.detach().clone()
        C_pred[C_pred == 1] = 0
        C_pred[C_pred == 2] = 1

        if self.otsu == True:
            y_otsu = np.array(y_hat[1,:,:])
            thresh = filters.threshold_otsu(y_otsu) #USAMOS OTSU PARA LA GENERACION DE IMAGENES
            binary = y_otsu > thresh
            D_pred = ndimage.binary_fill_holes(binary).astype(int)
        else:
            D_pred = y_arg.detach().clone() #OTSU
            D_pred[D_pred == 2] = 1

        if self.save_test == True:
            self.save_predictions(name[0],D_pred,C_pred,intermediate_OD)


        

    def save_predictions(self, name, OD_pred, OC_pred, Intermediate_pred):
        if self.dataset == 'REFUGE':
            dst_path = '/mnt/Almacenamiento/ODOC_segmentation/predicted/ModelR/'+self.data_test
        elif self.dataset == 'DRISHTI':
            dst_path = '/mnt/Almacenamiento/ODOC_segmentation/predicted/ModelD/'+self.data_test
        elif self.dataset == 'multi':
            dst_path = '/mnt/Almacenamiento/ODOC_segmentation/predicted/ModelM/'+self.data_test
        else:
            print('EL DATASET NO CORRESPONDE AL DE NINGUN MODELO')

        name = name.replace('Test/','')
        #print('NAME, ', name)
        name_OD = '/OD/'+ name
        name_OC = '/OC/'+ name
        name_OD_int = '/OD_int/'+ name

        #print('SHAPES ', OD_pred.shape, OC_pred.shape, Intermediate_pred.shape)

        #D_pred = torch.transpose(OD_pred, (1,2,0))
        #C_pred = torch.transpose(OC_pred, (1,2,0))
        #Intermediate_pred = torch.transpose(Intermediate_pred, (1,2,0))
        #OD_pred = OD_pred.cpu()
        OC_pred = OC_pred.cpu()
        Intermediate_pred = Intermediate_pred


        OD_pred = np.array(OD_pred*255,dtype=np.uint8)
        OC_pred = np.array(OC_pred*255,dtype=np.uint8)
        Intermediate_pred = np.array(Intermediate_pred*255,dtype=np.uint8)
        #print(Intermediate_pred.shape)

        iio.imsave(dst_path + name_OD, OD_pred)
        iio.imsave(dst_path + name_OC, OC_pred)
        iio.imsave(dst_path + name_OD_int, Intermediate_pred)

    def configure_optimizers(self):
        '''
    Configuration of the opimizer used in the model
    '''
        return torch.optim.Adam(self.parameters(), lr=self.lr) 

    def save_metrics(self, y_OC, y_OD, yhat_OC, yhat_OD, yhat_ODint,name):

        dice_C = self.dice_metric(y_OC,yhat_OC)
        dice_D = self.dice_metric(y_OD,yhat_OD)
        dice_Dint = self.dice_metric(y_OD,yhat_ODint)

        pres_C = self.precision_metric(y_OC,yhat_OC)
        pres_D = self.precision_metric(y_OD,yhat_OD)
        pres_Dint = self.precision_metric(y_OD,yhat_ODint)

        recall_C = self.recall_metric(y_OC,yhat_OC)
        recall_D = self.recall_metric(y_OD,yhat_OD)
        recall_Dint = self.recall_metric(y_OD,yhat_ODint)

        y_OC = y_OC.cpu()
        y_OD = y_OD.cpu()
        yhat_OC = yhat_OC.cpu()
        yhat_OD = yhat_OD.cpu()
        #yhat_ODint.cpu()

        y_OC = np.array(y_OC)
        y_OD = np.array(y_OD)
        yhat_OC = np.array(yhat_OC)
        #yhat_ODint = np.array(yhat_ODint)
        yhat_OD = np.array(yhat_OD)


        hd_OC = Hausdorff_distance(y_OC, yhat_OC)
        hd_Dint = Hausdorff_distance(y_OD,yhat_ODint)
        hd_OD = Hausdorff_distance(y_OD, yhat_OD)


        dice_prom = (dice_C + dice_D)/2

        #print('Dice c: ', dice_C, 'Dice_d: ', dice_D, 'dice prom: ', dice_prom)

        base_name= ntpath.basename(name)
        result_file = self.path_save
        if (result_file != None):
            #GUARDO LOS RESULTADOS DE VALIDACION EN UN ARCHIVO .TXT
            with open(result_file, "a+") as file_object:
                spamwriter = csv.writer(file_object, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                spamwriter.writerow([self.data_test + name , dice_C, dice_D, dice_Dint, dice_prom, pres_C, pres_D, pres_Dint, recall_C,recall_D,recall_Dint, hd_OC, hd_OD, hd_Dint])
                file_object.close()

    def dice_metric(self,gt, pred):
        '''
        Dice Index = 2 * \frac{(A \cap B)}{|A|+|B|}
        '''

        gt = gt.cpu()
        if torch.is_tensor(pred):
            pred = pred.cpu()
        
        return f1_score(gt.flatten(), pred.flatten(), average='macro')

    def precision_metric(self,gt, pred):
        '''
        Dice Index = 2 * \frac{(A \cap B)}{|A|+|B|}
        '''

        gt = gt.cpu()
        if torch.is_tensor(pred):
            pred = pred.cpu()
        
        return precision_score(gt.flatten(), pred.flatten(), average='macro')

    def recall_metric(self,gt, pred):
        '''
        Dice Index = 2 * \frac{(A \cap B)}{|A|+|B|}
        '''

        gt = gt.cpu()
        if torch.is_tensor(pred):
            pred = pred.cpu()
        
        return recall_score(gt.flatten(), pred.flatten(), average='macro')
        


    def get_cut_area(self,out,h,w):
        '''
        With the segmentation of the first model, we obtain the zone of the Optic disk to cut the original image'''
        
             


def main(config_m1, config_m2, hparams):
    #MODELS
    loss= nn.CrossEntropyLoss()
    model1 = Unet(config_m1, loss, model_name= hparams.dataset)
    checkpoint = torch.load(hparams.model1_path)
    model1.load_state_dict(checkpoint['state_dict'])
    #model1.load_from_checkpoint(checkpoint_path= hparams.model1_path, 
    #        config= config_m1,loss=loss,model_name='etapa1')

    model2 = Unet(config_m2,loss,model_name=(hparams.dataset + '_etapa2'))
    checkpoint = torch.load(hparams.model2_path)
    model2.load_state_dict(checkpoint['state_dict'])


    #DATASET
    if hparams.split:
        split_file = hparams.split + '/ODOC_segmentation_' + hparams.dataset_test + '.ini'
        config_split = ConfigParser()
        config_split.read(split_file)
    else: 
        config_split = None

    #data module especifico para test y prediction,
    #this data module retorn the image anda the mask, in case their exist without transforms
    if hparams.pred == False:
        dataMod = TestDataModule(data_path= hparams.data_path,
                    split_file= config_split,
                    dataset= hparams.dataset_test,
                    batch_size=1)
    else:
        dataMod = TestDataModule(data_path= hparams.data_path,
                    split_file= config_split,
                    dataset= hparams.dataset_test,
                    batch_size=1,
                    stage='pred')

    
    #SEGMENTADOR
    segmentator = ODOC_segmentation(hparams.data_path, hparams.dataset,hparams.dataset_test, config_m1, loss, model1, model2,hparams.result_path,save_test=True)
    headers =[ "NOMBRE" , "DICE OC" , "DICE OD" , "DICE ODint" , "DICE AVG" ,  "PRECISION OC" , "PRECISION OD", "PRECISION ODint", "RECALL OC", "RECALL OD", "RECALL ODint", "Hausdorff_distance_OC", "Hausdorff_distance_OD", "Hausdorff_distance_ODint"]

    if (hparams.result_path != None):
            #GUARDO LOS RESULTADOS DE VALIDACION EN UN ARCHIVO .TXT
            file_exists = os.path.isfile('hparams.result_path')
            with open(hparams.result_path, "a+") as file_object:

                writer = csv.writer(file_object, delimiter=',')
                if not file_exists:
                    writer.writerow(headers)
                file_object.close()



    trainer = Trainer(
            auto_lr_find=False,
            auto_scale_batch_size= False,
            max_epochs=int(config_m1['training']['epochs']), 
            accelerator="gpu",
            gpus=1,
            #logger=logger, #logger for the tensorboard
            log_every_n_steps=5,
            fast_dev_run=False) #if True, correra un unico batch, para testear si el modelo anda
    
    if hparams.pred == False:
        out = trainer.test(model=segmentator, datamodule=dataMod, verbose=True)
    else:
        out = trainer.predict(model=segmentator,datamodule=dataMod)




if __name__ == '__main__':
    
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--data_path', default='/mnt/Almacenamiento/ODOC_segmentation/data')
    parser.add_argument('--split', default=None)
    parser.add_argument('--config_m1',type=str, help='full path and file name of config file', default='/mnt/Almacenamiento/ODOC_segmentation/codigo/DRISHTI_configuration.ini')
    parser.add_argument('--config_m2',type=str, help='full path and file name of config file', default='/mnt/Almacenamiento/ODOC_segmentation/codigo/DRISHTI_configuration.ini')
    parser.add_argument('--dataset',required=True, type=str)
    parser.add_argument('--dataset_test',required=True,type=str)
    parser.add_argument('--log_dir', default='lightning_logs')
    parser.add_argument('--model1_path',type= str)
    parser.add_argument('--model2_path',type= str)
    parser.add_argument('--result_path', help='the path to the txt file were the results are saved it', default=None)
    parser.add_argument('--pred', default=False, type=bool)
    hparams = parser.parse_args()

    config_m1 = ConfigParser()
    config_m1.read(hparams.config_m1)
    config_m2 = ConfigParser()
    config_m2.read(hparams.config_m2)
    main(config_m1,config_m2,hparams)
    




