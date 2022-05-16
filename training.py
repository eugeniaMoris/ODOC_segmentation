from logging import logMultiprocessing
from math import degrees
import os
from argparse import ArgumentParser
from configparser import ConfigParser
from urllib.request import HTTPPasswordMgr
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

def get_augmentation(config):
    transform = [Hflip(), Vflip(), GaussianBlur(),ColorJitter(), RandomAffine(degrees=(0,180),translate=[1,1],scale=(.8,1.2)),ToTensor()]
    final_transforms = []
    probabilities = []

    flip_p = float(config['augmentation']['flip'])
    if flip_p != 0.0:
        probabilities.insert(len(probabilities),flip_p)
        final_transforms.insert(len(final_transforms), Hflip(probability=float(config['flip']['flip_h'])))

        probabilities.insert(len(probabilities),flip_p)
        final_transforms.insert(len(final_transforms), Vflip(probability=float(config['flip']['flip_v'])))

    blur_p = float(config['augmentation']['blur'])
    if blur_p != 0.0:      
        sigma = config['blur']['sigma']
        
        final_transforms.insert(len(final_transforms), GaussianBlur(sigma=float(sigma)))
        probabilities.insert(len(probabilities), blur_p)

    color_p = float(config['augmentation']['color'])
    if color_p != 0.0:
        brigtness = float(config['color']['brigtness'])
        contrast = float(config['color']['contrast'])
        saturation = float(config['color']['saturation'])
        hue = float(config['color']['hue'])

        final_transforms.insert(len(final_transforms),ColorJitter(brightness=brigtness, contrast=contrast, saturation=saturation, hue=hue))
        probabilities.insert(len(probabilities), color_p)

    affine_p = float(config['augmentation']['affine'])
    if affine_p != 0.0:
        min_d, max_d = config['affine']['degrees'].split(',')
        t1,t2 = config['affine']['translate'].split(',')
        min_s,max_s = config['affine']['scale'].split(',')

        degrees = (int(min_d),int(max_d))
        translate = [int(t1),int(t2)]
        scale = (float(min_s),float(max_s))

        final_transforms.insert(len(final_transforms), RandomAffine(degrees=degrees, translate=translate,scale=scale))
        probabilities.insert(len(probabilities), affine_p)

    final_transforms.insert(len(final_transforms), ToTensor())
    probabilities.insert(len(probabilities),1)

    return final_transforms, probabilities


def main(config,hparams):
    loss = nn.CrossEntropyLoss()

    
    # if hparams.resume != None:
    #     model = Unet(config, loss, hparams.dataset)
    #     model.load_from_checkpoint(hparams.resume,config,loss,hparams.dataset)
    # else:
    model = Unet(config, loss, hparams.dataset)


    
    os.makedirs(hparams.log_dir, exist_ok=True)
    try:
        log_dir = sorted(os.listdir(hparams.log_dir))[-1]
    except IndexError:
        name,_ = (hparams.config).split('.')
        log_dir = os.path.join(hparams.log_dir, 'version_0')
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(log_dir, 'checkpoints'),
        verbose=True,
    )

    split_file = hparams.split + '/ODOC_segmentation_' + hparams.dataset + '.ini'
    config_split = ConfigParser()
    config_split.read(split_file)

    #CREATE THE DATAMODULE CLASS

    transforms, probabilities = get_augmentation(config)
    #probabilities = [0.5, 0.5, 0.5,0.5,0.5,1]
    
    dataMod = DataModuleClass(data_path= hparams.data_path,
            split_file= config_split,
            dataset= hparams.dataset, aumentation=transforms,
            pred_data=config['training']['predic_data'],
            norm=bool(config['training']['norm']),
            probabilities=probabilities,
            batch_size=int(config['training']['batch_size']))

    seed_everything(42, workers=True)

    #CALLBACKS USED
    folder,name = (hparams.config).split('/')
    conf_name,_= name.split('.') 
    early_stop_callback = EarlyStopping(monitor='dice', patience=int(config['training']['patience']), verbose=True, mode='max') #Early stopping
    callback= ModelCheckpoint(monitor='dice', mode='max', #GUARDA EL MEJOR MODELO
            save_last=True, # GUARDA EL ULTIMO MODELO
            filename= conf_name + '-{epoch}',
            save_weights_only=False) #GUARDA SOLAMENTE LOS PESOS DEL MODELO


    logger = TensorBoardLogger('lightning_logs', name=hparams.dataset) #for the tensor board

    #generate the trainer
    if hparams.resume != None:
        resume_path = hparams.resume
    else:
        resume_path = None

    trainer = Trainer(
        callbacks=[callback, early_stop_callback], #add callbacks
        auto_lr_find=False,
        resume_from_checkpoint = resume_path,
        auto_scale_batch_size= False,
        max_epochs=int(config['training']['epochs']), 
        accelerator="gpu",
        gpus=1,
        logger=logger, #logger for the tensorboard
        log_every_n_steps=5,
        fast_dev_run=False) #if True, correra un unico batch, para testear si el modelo anda


    trainer.fit(model,dataMod)#esta opcion y la misma opcion en el trainer solo se pude utilizar si se guarda todo el modelo no solo los weight
    #ver como se puede devolver el estado de los modelos si se almacenan solo los pesos

    #trainer.validate(datamodule=dataMod,ckpt_path='best',verbose=True)
    #trainer.predict(model= model,ckpt_path='best', datamodule=dataMod)

if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--data_path', default='/mnt/Almacenamiento/ODOC_segmentation/data')
    parser.add_argument('--split', default='/mnt/Almacenamiento/ODOC_segmentation/split')
    parser.add_argument('--config',type=str, help='full path and file name of config file', default='/mnt/Almacenamiento/ODOC_segmentation/codigo/DRISHTI_configuration.ini')
    parser.add_argument('--dataset',required=True, type=str)
    parser.add_argument('--log_dir', default='lightning_logs')
    parser.add_argument('--resume', default=None, help='give the checkpoint if we want to continue a training')
    hparams = parser.parse_args()

    #READ CONFIG FILE
    config = ConfigParser()
    config.read(hparams.config)
    #parser = Unet.add_model_specific_args(parent_parser)


    main(config,hparams)



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
