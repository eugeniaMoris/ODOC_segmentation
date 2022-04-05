from logging import logMultiprocessing
import os
from argparse import ArgumentParser
from configparser import ConfigParser
#from posixpath import split
import torch.nn as nn

#from numpy import dtype, lookfor

from Models.Unet import Unet
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
#import tensorboard

from data_processing.data_preprocesing import DataModuleClass

def main(config,hparams):
    loss = nn.CrossEntropyLoss()
    model = Unet(config, loss)
    
    os.makedirs(hparams.log_dir, exist_ok=True)
    try:
        log_dir = sorted(os.listdir(hparams.log_dir))[-1]
    except IndexError:
        log_dir = os.path.join(hparams.log_dir, 'version_0')
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(log_dir, 'checkpoints'),
        verbose=True,
    )

    split_file = hparams.split + '/ODOC_segmentation_' + hparams.dataset + '.ini'
    config_split = ConfigParser()
    config_split.read(split_file)

    #CREATE THE DATAMODULE CLASS
    dataMod = DataModuleClass(hparams.data_path,
            config_split,
            hparams.dataset)

    seed_everything(42, workers=True)

    #CALLBACKS USED
    early_stop_callback = EarlyStopping(monitor='dice', patience=3, verbose=True, mode='max') #Early stopping
    callback= ModelCheckpoint(monitor='dice', mode='max', #GUARDA EL MEJOR MODELO
            save_last=True, # GUARDA EL ULTIMO MODELO
            save_weights_only=True) #GUARDA SOLAMENTE LOS PESOS DEL MODELO


    logger = TensorBoardLogger('lightning_logs', name='drishti_model') #for the tensor board

    #generate the trainer
    trainer = Trainer(
            callbacks=[callback, early_stop_callback], #add callbacks
            auto_lr_find=True,
            auto_scale_batch_size= False,
            max_epochs=30, 
            accelerator="gpu",
            gpus=1,
            logger=logger, #logger for the tensorboard
            log_every_n_steps=5,
            fast_dev_run=False) #if True, correra un unico batch, para testear si el modelo anda

    #train the model
    trainer.fit(model,dataMod)

if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--data_path', default='/mnt/Almacenamiento/ODOC_segmentation/data')
    parser.add_argument('--split', default='/mnt/Almacenamiento/ODOC_segmentation/split')
    parser.add_argument('--config',type=str, help='full path and file name of config file', default='/mnt/Almacenamiento/ODOC_segmentation/codigo/DRISHTI_configuration.ini')
    parser.add_argument('--dataset',required=True, type=str)
    parser.add_argument('--log_dir', default='lightning_logs')
    hparams = parser.parse_args()

    #READ CONFIG FILE
    config = ConfigParser()
    config.read(hparams.config)
    #parser = Unet.add_model_specific_args(parent_parser)


    main(config,hparams)