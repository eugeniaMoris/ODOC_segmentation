#from random import random
#from typing import Final
#from warnings import filters
import numpy as np
#import tensorboard
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchvision.utils import save_image
#from torch.utils.data import DataLoader, random_split
from . import utils_model
import pytorch_lightning as pl
from data_processing import *
from sklearn.metrics import f1_score,accuracy_score
#import imageio as iio
from PIL import Image

class UnetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,is_BN,dropout,activation='relu'):
        super(UnetConvBlock,self).__init__()

        #primer conv
        conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=not(is_BN)) #si hay BatchNormalization queda en False
        activation_1 = utils_model.get_activation_fn(activation)
        
        if is_BN:
            self.conv1 = nn.Sequential(conv1, nn.BatchNorm2d(out_channels),activation_1)
        else:
            self.conv1 = nn.Sequential(conv1, activation_1)

        #segundo conv
        conv2 = nn.Conv2d(in_channels= out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=not(is_BN))
        activation_2 = utils_model.get_activation_fn(activation)
        
        if is_BN:
            self.conv2 = nn.Sequential(conv2, nn.BatchNorm2d(out_channels), activation_2)
        else:
            self.conv2 = nn.Sequential(conv2, activation_2)

        if dropout > 0.0:
            self.drop = nn.Dropout(dropout)
        else:
            self.drop = None

    
    def forward(self, x):
        output= self.conv1(x)
        output = self.conv2(output)

        if not(self.drop is None):
            output = self.drop(output)
        
        return output

class UnetUpsampling(nn.Module):
    '''
    constructor del upsampling block
    TransposeConvolution / Upsampling
    Convolutional block
    '''

    def __init__(self, in_channels, out_channels, upsample_size, is_deconv, dropout, is_BN, activation= 'relu',upsampling_type= 'nearest'):
        super(UnetUpsampling, self).__init__()

        if is_deconv: #we need a transposeConv (agrandamos el input)
            #primero un convolution transpose seguido de una convolucion simple
            self.up = nn.ConvTranspose2d(upsample_size,upsample_size,kernel_size=2,stride=2)
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=not(is_BN))
            activation1 = utils_model.get_activation_fn(activation)

            if is_BN:
                self.conv = nn.Sequential(conv, nn.BatchNorm2d(out_channels), activation1)
            else:
                self.conv = nn.Sequential(conv, activation1)
        
        else:
            #first a upersamplig operation and then a convolutional block
            self.up = nn.Upsample(scale_factor=2)
            self.conv = UnetConvBlock(in_channels, out_channels, is_BN, dropout, activation)
        #ver despues porque algunos es con deconv y porque otros no 
        #sera para el ultimo conv?

        self.conv = UnetConvBlock(in_channels, out_channels, is_BN, dropout, activation)

    def forward(self, from_skip_connection, prev_layer):
        '''
        1. upsampling the input from the previous layer
        2. verify the difference between the two tensors and apply padding
        3. concatenate and apply th convolutional block'''

        rescaled_input = self.up(prev_layer)

        offset = rescaled_input.size()[2] - from_skip_connection.size()[2]
        padding= 2 * [offset // 2, offset // 2]
        from_skip_connection = F.pad(from_skip_connection,padding)

        return self.conv(torch.cat([from_skip_connection,rescaled_input],1))

class Unet(pl.LightningModule):
    def __init__(self, config, loss):
        super(Unet, self).__init__()
        self.loss = loss
        self.config=config
        self.softmax = nn.Softmax(dim=1)
        #METRICS
        
        self.name='model'
        self.n_classes = 2
        self.is_deconv = False
        self.is_BN = True
        self.dropout = 0.0
        self.use_otsu = False 

        self.valid_sample = None       

        # change configuration if available in the file
        if 'n_classes' in config['architecture']:
            self.n_classes = int(config['architecture']['n_classes'])
        if 'use-deconvolution' in config['architecture']:
            self.is_deconv = bool(config['architecture']['is_deconv'])
        if 'batch-norm' in config['architecture']:
            self.is_batchnorm = bool(config['architecture']['batch_normalization'])

        self.dropout = [0.0, 0.0, 0.0, 0.0, float(config['architecture']['dropout']), 0.0, 0.0, 0.0, 0.0]
        
        filters_e=config['architecture']['filters_encoder'].split(',')
        print(filters_e)
        filters_encoder=[]
        for f in filters_e:
            filters_encoder.insert(len(filters_encoder),int(f))

        filters_d=config['architecture']['filters_decoder'].split(',')
        filters_decoder=[]
        for f in filters_d:
            filters_decoder.insert(len(filters_decoder),int(f))
        
        
        filters_encoder = np.array(filters_encoder)
        filters_decoder = np.array(filters_decoder)

        activation = config['architecture']['activation']
        self.activation_fn = utils_model.get_activation_fn(activation)
        pooling = config['architecture']['pooling']

        #DOWNSAMPLING
        #dconv = down conv
        self.dconv1 = UnetConvBlock(3, int(filters_encoder[0]), self.is_BN, self.dropout[0], activation)
        self.pool1 = utils_model.get_pooling(pooling, kernel=2)
        self.dconv2 = UnetConvBlock(int(filters_encoder[0]), int(filters_encoder[1]), self.is_BN, self.dropout[1], activation)
        self.pool2 = utils_model.get_pooling(pooling, kernel=2)
        self.dconv3 = UnetConvBlock(int(filters_encoder[1]), int(filters_encoder[2]), self.is_BN, self.dropout[2], activation)
        self.pool3 = utils_model.get_pooling(pooling, kernel=2)
        self.dconv4 = UnetConvBlock(int(filters_encoder[2]), int(filters_encoder[3]), self.is_BN, self.dropout[3], activation)
        self.pool4 = utils_model.get_pooling(pooling, kernel=2)

        #bootleneck layer
        self.bottleneck = UnetConvBlock(int(filters_encoder[3]), int(filters_encoder[4]), self.is_BN, self.dropout[4], activation)

        #UPSAMPLING
        #comienza a concatenarse los skip connection, uconv = up conv
        self.uconv4 = UnetUpsampling(int(filters_encoder[4]) + int(filters_encoder[3]), int(filters_decoder[3]), int(filters_encoder[4]), self.is_deconv, self.dropout[5], self.is_BN, activation)
        self.uconv3 = UnetUpsampling(int(filters_decoder[3]) + int(filters_encoder[2]), int(filters_decoder[2]), int(filters_encoder[3]), self.is_deconv, self.dropout[6], self.is_BN, activation)
        self.uconv2 = UnetUpsampling(int(filters_decoder[2]) + int(filters_encoder[1]), int(filters_decoder[1]), int(filters_encoder[2]), self.is_deconv, self.dropout[7], self.is_BN, activation)
        self.uconv1 = UnetUpsampling(int(filters_decoder[1]) + int(filters_encoder[0]), int(filters_decoder[0]), int(filters_encoder[1]), self.is_deconv, self.dropout[8], self.is_BN, activation)

        #final conv
        self.final = nn.Conv2d(int(filters_decoder[0]), self.n_classes, 1)

    def forward(self, inputs):
        #print('INPUT SIZE: ', inputs.size())
        #print('EPOCH: ',self.current_epoch )
        #batch_size, _, _, _ = inputs.size()

        #x = inputs.view(batch_size, -1)

        #downsampling
        dconv1 = self.dconv1(inputs)
        #dconv1 = self.dconv1(x)
        pool1  = self.pool1( dconv1)
        dconv2 = self.dconv2( pool1)
        pool2  = self.pool2( dconv2)
        dconv3 = self.dconv3( pool2)
        pool3 = self.pool3(dconv3)
        dconv4= self.dconv4(pool3)
        pool4 = self.pool4(dconv4)

        #bottleneck without dropout
        bottleneck = self.bottleneck(pool4)

        #upsampling
        uconv4 = self.uconv4(dconv4, bottleneck)
        uconv3 = self.uconv3(dconv3,uconv4)
        uconv2 = self.uconv2(dconv2,uconv3)
        uconv1 = self.uconv1(dconv1,uconv2)

        final = self.final(uconv1)

        #print('FINAL SHAPE: ', final.size())
        return final

    def training_step(self, batch, batch_nb):
        #if(self.current_epoch==1):
            #sampleImg=torch.rand((1,1,28,28))
            #self.logger.experiment.add_graph(Unet(self.config,self.loss),sampleImg)


        x,y = batch
        #print('TRAINING STE: ', x.size())

        y_hat = self.forward(x)
        #print('Y_hat: ', y_hat.size(), y.size())

        loss = self.loss(y_hat,y) 
        self.logger.experiment.add_scalars("Loss", {'train loss': loss},
                                            self.global_step)
        
        y_arg = torch.argmax(y_hat,dim=1)
        acc = self.acc_metric(y,y_arg)
        self.logger.experiment.add_scalars("Acc", {' train acc' : acc},
                                            self.global_step)
        acc= torch.tensor(acc)
        #print('LOSS TRAINING: ', loss)

        return {'loss':loss, 'acc':acc}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        self.logger.experiment.add_scalars("Avg loss",{'train':avg_loss},
                                            self.current_epoch)

        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()

        self.logger.experiment.add_scalars("Avg acc",{'train' :avg_acc},
                                            self.current_epoch)

    def validation_step(self, batch, batch_nb):

        x, y= batch

        if(self.current_epoch==0):
            self.valid_sample = x,y

        #print('\n VALIDATION STE: ', y.size())
        y_hat= self.forward(x)
        loss = self.loss(y_hat,y)
        #calcular la dice 

        y_arg = torch.argmax(y_hat,dim=1)
        dice = self.dice_metric(y,y_arg)
        acc = self.acc_metric(y,y_arg)

        acc= torch.tensor(acc)
        dice= torch.tensor(dice)

        self.logger.experiment.add_scalars("Loss", {'Val loss' : loss},
                                            self.global_step)

        self.logger.experiment.add_scalar("Valid_DICE", dice,
        self.global_step)

        self.logger.experiment.add_scalars("Acc", {'Val acc' : acc},
        self.global_step)

        #print(f'\n VALIDATION LOSS: {loss} ON EPOCH {self.current_epoch} \n',)
        #self.log('val_loss', loss)
        self.log('dice', dice)



        return {'val_loss': loss, 'val_dice': dice, 'val_acc': acc}

    def validation_epoch_end(self, outputs):

        ''' al terminar la validacion se calcula el promedio de la loss de validacion'''
        #print('OUTPUTS: ', outputs)
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalars("Avg loss",{'valid':avg_loss},
                                            self.current_epoch
                                             )

        avg_dice = torch.stack([x['val_dice'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("valid_avg_dice",avg_dice,
                                            self.current_epoch)

        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        self.logger.experiment.add_scalars("Avg acc",{'valid':avg_acc},
                                            self.current_epoch)
        x, y= self.valid_sample
        pred = self.forward(x)
        pred = self.softmax(pred)
        pred_arg = torch.argmax(pred, dim=1)

        img = self.generate_img(pred_arg[0,:,:], y[0,:,:])

        self.logger.experiment.add_image("Img", x[0,:,:,:] , dataformats="CHW")
        self.logger.experiment.add_image("ground truth", y[0,:,:], dataformats="HW")
        self.logger.experiment.add_image("predict", pred[0,1,:,:], dataformats="HW")
        self.logger.experiment.add_image("predict_arg", pred_arg[0,:,:], dataformats="HW")
        self.logger.experiment.add_image("prediction", img, dataformats="CHW")

        self.log('avg_val_loss', avg_loss)
        print(f'\n VALIDATION AVG LOSS: {avg_dice} ON EPOCH {self.current_epoch} \n',)


        

        return {'avg_val_loss': avg_loss, 'avg_val_dice': avg_dice}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.1) 

    def generate_img(self, pred, true):
        ''' In this method we generate img for TP, FP ans FN
        ------------------
        input:
        pred: img 2D with the prediction
        true: img 2D with the ground truth
        '''
        #print('SHAPES: ', pred.size(), true.size())
        TP = pred * true
        FP = pred - true
        FP[FP == -1] = 0
        FN = true - pred
        FN[FN == -1] = 0

        #print('uniques: ', torch.unique(TP), torch.unique(FP), torch.unique(FN))
        img = self.tensor_to_image(TP,FP,FN)

        return img


    def tensor_to_image(self,green, red, blue):

        green = green*255
        red = red*255
        blue = blue*255

        green = green.cpu()
        red = red.cpu()
        blue = blue.cpu()

        #print('SHAPE ', green.shape, red.shape, blue.shape)

        green = np.array(green, dtype=np.uint8)
        red = np.array(red, dtype=np.uint8)
        blue = np.array(blue, dtype=np.uint8)
        #print('SHAPE arr ', green.shape, red.shape, blue.shape)


        tensor = np.array([red,green,blue])

        #print('SHAPE TENSOR ', tensor.shape)
        #if np.ndim(tensor)>3:
        #    assert tensor.shape[0] == 1
        #    tensor = tensor[0]
        #return Image.fromarray(tensor, 'RGB')
        return tensor

    def dice_metric(self,gt, pred):
        '''
        Dice Index = 2 * \frac{(A \cap B)}{|A|+|B|}
        '''

        gt = gt.cpu()
        pred = pred.cpu()
        return f1_score(gt.flatten(), pred.flatten())

    def acc_metric(self,gt, pred):
        '''
        Dice Index = 2 * \frac{(A \cap B)}{|A|+|B|}
        '''

        gt = gt.cpu()
        pred = pred.cpu()
        return accuracy_score(gt.flatten(), pred.flatten(), normalize= True)

    @staticmethod
    def add_model_specific_args(parent_parser):

        parser = parent_parser.add_argument_group("Unet")
        #parser.add_argument("--data_path", type=str, default="/some/path")
        #parser.add_argument("--data split", type=str, default='/some/path')
        parser.add_argument("--filters_encoder", type=list, default=[64, 128, 256, 512, 1024])
        parser.add_argument("--filters_decoder", type=list, default=[64,128,256,512])
        parser.add_argument("--activation", type=str, default='relu')
        parser.add_argument("--pooling", type= str, default='max')

        parser.add_argument("--n_classes", type=int, default= 2)
        parser.add_argument("--is_deconv", type=bool, default=False)
        parser.add_argument("--is_BN", type=bool, default=True)
        parser.add_argument("--dropout", type=float,default=0.0)
        parser.add_argument("--use-otsu",type=bool, default=False)
        return parser



    