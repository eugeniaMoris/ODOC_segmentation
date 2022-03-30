from logging import logMultiprocessing
import os
from argparse import ArgumentParser
from configparser import ConfigParser
#from posixpath import split
import torch.nn as nn

#from numpy import dtype, lookfor

from Models.Unet import Unet
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
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
    # stop_callback = EarlyStopping(
    #     monitor='val_loss',
    #     mode='auto',
    #     patience=5,
    #     verbose=True,
    # )

    split_file = hparams.split + '/ODOC_segmentation_' + hparams.dataset + '.ini'
    config_split = ConfigParser()
    config_split.read(split_file)

    dataMod = DataModuleClass(hparams.data_path,
            config_split,
            hparams.dataset)
    #dataMod.setup()

    # trainer = Trainer(
    #     gpus=1,
    #     checkpoint_callback=checkpoint_callback,
    #     early_stop_callback=stop_callback,
    # )
    seed_everything(42, workers=True)

    logger = TensorBoardLogger('lightning_logs', name='drishti_model')
    trainer = Trainer(auto_lr_find=False,
            auto_scale_batch_size= False,
            max_epochs=50, 
            accelerator="gpu",
            gpus=1,
            logger=logger,
            log_every_n_steps=2,
            fast_dev_run=False) #correra un unico batch, para testear si el modelo anda
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