from logging import logMultiprocessing
import os
from argparse import ArgumentParser
from configparser import ConfigParser
#from posixpath import split
import torch.nn as nn
import torchvision.transforms as transforms
from yaml import compose
from Models.Augmentation import *
import numpy as np

#from numpy import dtype, lookfor

from Models.Unet import Unet
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
#import tensorboard

from data_processing.data_preprocesing import DataModuleClass
import matplotlib.pyplot as plt

def main(config,hparams):
    loss = nn.CrossEntropyLoss()
    model = Unet(config, loss, hparams.dataset)

    
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
    composed= transforms.Compose([Hflip(.5), Vflip(.5), GaussianBlur(),ColorJitter(), RandomAffine(degrees=(0,180),translate=[1,1],scale=(.8,1.2)),ToTensor()])
    transform = [Hflip(.5), Vflip(.5), GaussianBlur(),ColorJitter(), RandomAffine(degrees=(0,180),translate=[1,1],scale=(.8,1.2)),ToTensor()]
    probabilities = [0.5, 0.5, 0.5,0.5,0.5,1]
    
    dataMod = DataModuleClass(data_path= hparams.data_path,
            split_file= config_split,
            dataset= hparams.dataset)

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
            auto_lr_find=False,
            auto_scale_batch_size= False,
            max_epochs=30, 
            accelerator="gpu",
            gpus=1,
            logger=logger, #logger for the tensorboard
            log_every_n_steps=5,
            fast_dev_run=False) #if True, correra un unico batch, para testear si el modelo anda

    #train the model
    trainer.fit(model,dataMod)
    trainer.validate(datamodule=dataMod,ckpt_path='best',verbose=True)
    trainer.predict(model= model,ckpt_path='best', datamodule=dataMod)

# if __name__ == '__main__':
#     parser = ArgumentParser(add_help=False)
#     parser.add_argument('--data_path', default='/mnt/Almacenamiento/ODOC_segmentation/data')
#     parser.add_argument('--split', default='/mnt/Almacenamiento/ODOC_segmentation/split')
#     parser.add_argument('--config',type=str, help='full path and file name of config file', default='/mnt/Almacenamiento/ODOC_segmentation/codigo/DRISHTI_configuration.ini')
#     parser.add_argument('--dataset',required=True, type=str)
#     parser.add_argument('--log_dir', default='lightning_logs')
#     hparams = parser.parse_args()

#     #READ CONFIG FILE
#     config = ConfigParser()
#     config.read(hparams.config)
#     #parser = Unet.add_model_specific_args(parent_parser)


#     main(config,hparams)



# #CREATE THE DATAMODULE CLASS

# sf ='/mnt/Almacenamiento/ODOC_segmentation/split'
# split_file = sf + '/ODOC_segmentation_' + 'DRISHTI' + '.ini'
# config_split = ConfigParser()
# config_split.read(split_file)

# #composed =None
# #composed= transforms.Compose([ToTensor()])
# #composed= transforms.Compose([Hflip(.5),ToTensor()])
# #composed= transforms.Compose([Hflip(.5), Vflip(.5), GaussianBlur(), ToTensor()])
# #composed= transforms.Compose([Hflip(.5), Vflip(.5), RandomAffine(degrees=(0,0.5),translate=(0.2,0.2),scale=(0.8,1.2)),ToTensor()])
# #composed= transforms.Compose([Hflip(.5), Vflip(.5), GaussianBlur(),ColorJitter(), RandomAffine(degrees=(0,10),translate=(1,1),scale=(0,10)),ToTensor()])
# composed= transforms.Compose([Hflip(.5), Vflip(.5), GaussianBlur(),ColorJitter(), RandomAffine(degrees=(0,180),translate=[1,1],scale=(.8,1.2)),ToTensor()])

# transform = [Hflip(.5), Vflip(.5), GaussianBlur(),ColorJitter(), RandomAffine(degrees=(0,180),translate=[1,1],scale=(.8,1.2)),ToTensor()]
# probabilities = [0.5, 0.5, 0.5,0.5,0.5,0]


# dataMod = DataModuleClass(data_path= '/mnt/Almacenamiento/ODOC_segmentation/data',
#         split_file= config_split,
#         dataset= 'DRISHTI',
#         aumentation= transform,
#         probabilities=probabilities)

# dataMod.setup()
# print(dataMod)
# dataset = dataMod.train_dataloader()
# images, labels,name,_ = next(iter(dataset))
# #x,y,_,_ = data_iter.next()

# img1 = images[0,:,:,:].cpu()
# img1 = img1.clip(0,1)
# img1 = np.transpose(img1, (1,2,0))

# img2 = images[1,:,:,:].cpu()
# img2 = img2.clip(0,255)
# img2 = np.transpose(img2, (1,2,0))
# #img2 = img2 * 255

# img3 = images[2,:,:,:].cpu()
# img3 = img3.clip(0,255)
# img3 = np.transpose(img3, (1,2,0))
# #img3 = img3 * 255

# fig, (ax0, ax1,ax2,ax3,ax4,ax5) = plt.subplots(1, 6)
# ax0.imshow(img1)
# ax1.imshow(labels[0,:,:])
# ax2.imshow(img2)
# ax3.imshow(labels[1,:,:])
# ax4.imshow(img3)
# ax5.imshow(labels[2,:,:])
# fig.suptitle(name[0] + name[1] + name[2])


# plt.show()
