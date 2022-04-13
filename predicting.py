import nntplib
from unittest import result
from numpy import dtype


import torch.nn as nn
from configparser import ConfigParser
from argparse import ArgumentParser
from Models.Unet import Unet
from pytorch_lightning import Trainer, seed_everything
from data_processing.data_preprocesing import DataModuleClass


def main(config,hparams):
    loss = nn.CrossEntropyLoss()
    model = Unet(config, loss)
    trainer = Trainer()

    split_file = hparams.split + '/ODOC_segmentation_' + hparams.dataset + '.ini'
    config_split = ConfigParser()
    config_split.read(split_file)

    #CREATE THE DATAMODULE CLASS
    dataMod = DataModuleClass(hparams.data_path,
            config_split,
            hparams.dataset)

    callback_path = hparams.log_dir + '/' + (hparams.dataset).lower() + '_model/version_' + hparams.version + '/checkpoints/'
    if hparams.best_last == 'best':
        callback_path = callback_path + 'epoch*'
    else:
        callback_path = callback_path + 'last.ckpt'

    model_2 = Unet.load_from_checkpoint(callback_path)
    #result = trainer.test(model=model_2, datamodule=dataMod, verbose= True)
    result = trainer.validate(model=model_2,datamodule=dataMod,ckpt_path=callback_path,verbose=True)


    

if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--data_path', default='/mnt/Almacenamiento/ODOC_segmentation/data')
    parser.add_argument('--split', default='/mnt/Almacenamiento/ODOC_segmentation/split')
    parser.add_argument('--config',type=str, help='full path and file name of config file', default='/mnt/Almacenamiento/ODOC_segmentation/codigo/DRISHTI_configuration.ini')
    parser.add_argument('--dataset',required=True, type=str)
    parser.add_argument('--log_dir', default='lightning_logs')
    parser.add_argument('--best_last', help='use the best or the last model', default='best')
    parser.add_argument('--version', help='numero de la version a cargar', default=0, type=str)
    hparams = parser.parse_args()

    #READ CONFIG FILE
    config = ConfigParser()
    config.read(hparams.config)
    #parser = Unet.add_model_specific_args(parent_parser)


    main(config,hparams)
    