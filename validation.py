import torch
from logging import logMultiprocessing
from math import degrees
import os
from argparse import ArgumentParser
from configparser import ConfigParser
#from posixpath import split
import torch.nn as nn
import torchvision.transforms as transforms
from yaml import compose
from Models.Augmentation import *
import numpy as np
from training import get_augmentation
import ntpath

#from numpy import dtype, lookfor

from Models.Unet import Unet
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
#import tensorboard

from data_processing.data_preprocesing import DataModuleClass
from data_processing.dataModuleDC import DataModuleClassDC

import matplotlib.pyplot as plt

from training import main

def main(config, hparams):
    loss = nn.CrossEntropyLoss()

    con_name = ntpath.basename(hparams.config)
    name = con_name.split('.')
    model = Unet(config, loss, model_name=hparams.dataset)

    split_file = hparams.split + '/ODOC_segmentation_' + hparams.dataset + '.ini'
    config_split = ConfigParser()
    config_split.read(split_file)

    #CREATE THE DATAMODULE CLASS

    transforms, probabilities = get_augmentation(config)
    #probabilities = [0.5, 0.5, 0.5,0.5,0.5,1]
    
    if hparams.etapa == 0:
        dataMod = DataModuleClass(data_path= hparams.data_path,
                split_file= config_split,
                dataset= hparams.dataset, aumentation=transforms,
                pred_data=config['training']['predic_data'],
                norm=bool(config['training']['norm']),
                probabilities=probabilities,
                batch_size=int(config['training']['batch_size']))
    else:
        print('entro aca')
        dataMod = DataModuleClassDC(data_path= hparams.data_path,
                split_file= config_split,
                dataset= hparams.dataset, aumentation=transforms,
                pred_data=config['training']['predic_data'],
                norm=bool(config['training']['norm']),
                probabilities=probabilities,
                batch_size=1)


    seed_everything(42, workers=True)

    trainer = Trainer(
            auto_lr_find=False,
            auto_scale_batch_size= False,
            max_epochs=int(config['training']['epochs']), 
            accelerator="gpu",
            gpus=1,
            #logger=logger, #logger for the tensorboard
            log_every_n_steps=5,
            fast_dev_run=False) #if True, correra un unico batch, para testear si el modelo anda
    
    out = trainer.validate(model=model,datamodule=dataMod,ckpt_path=hparams.model_path,verbose=True)

    #out = trainer.test(model=model, datamodule=dataMod, ckpt_path=hparams.model_path, verbose=True)
    

    result_file = hparams.result_path
    if (result_file != None):
            #GUARDO LOS RESULTADOS DE VALIDACION EN UN ARCHIVO .TXT
            with open(result_file, "a+") as file_object:
                file_object.write(name[0] + ' ' + str(out)+ '\n')
                file_object.close()
    print('OUT: ', out)



if __name__ == '__main__':
    if __name__ == '__main__':
        parser = ArgumentParser(add_help=False)
    parser.add_argument('--data_path', default='/mnt/Almacenamiento/ODOC_segmentation/data')
    parser.add_argument('--split', default='/mnt/Almacenamiento/ODOC_segmentation/split')
    parser.add_argument('--config',type=str, help='full path and file name of config file', default='/mnt/Almacenamiento/ODOC_segmentation/codigo/DRISHTI_configuration.ini')
    parser.add_argument('--dataset',required=True, type=str)
    parser.add_argument('--log_dir', default='lightning_logs')
    parser.add_argument('--model_path',type= str)
    parser.add_argument('--result_path', help='the path to the txt file were the results are saved it', default=None)
    parser.add_argument('--etapa', help='etapa 0 para la deteccion del disco, etapa 1 para la deteccion de disco y copa', type= int)
    #parser.add_argument('--name', default=None, type=str, help='nombre donde queremso que se guarde el experimento')
    hparams = parser.parse_args()

    #READ CONFIG FILE
    config = ConfigParser()
    config.read(hparams.config)
    #parser = Unet.add_model_specific_args(parent_parser)


    main(config,hparams)
    