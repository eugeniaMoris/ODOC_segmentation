#from random import random
#from typing import Final
#from warnings import filters
from unicodedata import name
import numpy as np
from sklearn.preprocessing import scale
#import tensorboard
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as FT
from torchvision.utils import save_image
#from torch.utils.data import DataLoader, random_split
from . import utils_model
import pytorch_lightning as pl
from data_processing import *
from sklearn.metrics import f1_score,accuracy_score
import imageio as iio
from PIL import Image
import torchvision.transforms as transforms

class UnetConvBlock(nn.Module):

    '''
    Bloque de la Unet, la misma consiste en un conjunto de 2 capas convolucionales
    '''
    def __init__(self, in_channels, out_channels,is_BN,dropout,activation='relu'):
        '''
        input:
        in_channels: size of the input
        out_channels: size of the output
        is_BN: (bool) True if we want to use Batch Normalization
        dropout: value of dropout to use
        activation: (str) the type of activation fuction to use in the block
        '''
                
        super(UnetConvBlock,self).__init__()

        #FIRST CONV LAYER
        conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=not(is_BN)) #si hay BatchNormalization queda en False
        activation_1 = utils_model.get_activation_fn(activation)
        
        if is_BN:
            self.conv1 = nn.Sequential(conv1, nn.BatchNorm2d(out_channels),activation_1)
        else:
            self.conv1 = nn.Sequential(conv1, activation_1)

        #SECOND CONV LAYER
        conv2 = nn.Conv2d(in_channels= out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=not(is_BN))
        activation_2 = utils_model.get_activation_fn(activation)
        
        if is_BN:
            self.conv2 = nn.Sequential(conv2, nn.BatchNorm2d(out_channels), activation_2)
        else:
            self.conv2 = nn.Sequential(conv2, activation_2)

        #DROPOUT
        if dropout > 0.0:
            self.drop = nn.Dropout(dropout)
        else:
            self.drop = None

    
    def forward(self, x):
        '''
        Foward function of the U-net block
        '''
        output= self.conv1(x)
        output = self.conv2(output)

        if not(self.drop is None):
            output = self.drop(output)
        
        return output

class UnetUpsampling(nn.Module):
    '''
    Construction of the upsampling block
    TransposeConvolution / Upsampling
    Convolutional block
    '''

    def __init__(self, in_channels, out_channels, upsample_size, is_deconv, dropout, is_BN, activation= 'relu',upsampling_type= 'nearest'):
        super(UnetUpsampling, self).__init__()

        if is_deconv: #we need a transposeConv (agrandamos el input)
            #first a convolution transpose following by a simpler convolucion layer
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
    def __init__(self, config, loss, model_name):
        '''
        Unet architecture
        ----------------------
        input
        config: config file with configuration of the model
        loss: loss function to use in the model
        ----------------------'''
        super(Unet, self).__init__()
        self.loss = loss
        self.config=config
        self.softmax = nn.Softmax(dim=1)
        #METRICS
        
        self.name= model_name
        self.n_classes = 2 #we have a mask with 0|1
        self.is_deconv = False
        self.is_BN = True
        self.dropout = 0.0
        self.use_otsu = False 

        self.valid_sample = None     
        self.count = 0  

        # change configuration if available in the file
        if 'n_classes' in config['architecture']:
            self.n_classes = int(config['architecture']['n_classes'])
        if 'use-deconvolution' in config['architecture']:
            self.is_deconv = bool(config['architecture']['is_deconv'])
        if 'batch-norm' in config['architecture']:
            self.is_batchnorm = bool(config['architecture']['batch_normalization'])

        self.dropout = [0.0, 0.0, 0.0, 0.0, float(config['architecture']['dropout']), 0.0, 0.0, 0.0, 0.0]
        
        #de mas grande a mas chico
        filters_e=config['architecture']['filters_encoder'].split(',')
        filters_encoder=[]
        for f in filters_e:
            filters_encoder.insert(len(filters_encoder),int(f))

        # de mas chico a mas grande
        filters_d=config['architecture']['filters_decoder'].split(',')
        filters_decoder=[]
        for f in filters_d:
            filters_decoder.insert(len(filters_decoder),int(f))
        
        filters_encoder = np.array(filters_encoder)
        filters_decoder = np.array(filters_decoder)

        activation = config['architecture']['activation']
        self.activation_fn = utils_model.get_activation_fn(activation)
        pooling = config['architecture']['pooling']

        #ARCHITECTURE

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
        '''
        UNET FORWARD FUNCTION
        '''
        #downsampling
        dconv1 = self.dconv1(inputs)
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

        return final

    def training_step(self, batch, batch_nb):
        '''
        training step inside the training loop
        it is made it for each batch of data'''

        x,y, names, shapes = batch
        y_hat = self.forward(x)

        #calculate the error/loss of the classification
        loss = self.loss(y_hat,y)  

        #SAVE METRICS IN THE TENSORBOARD
        self.logger.experiment.add_scalars("Loss", {'train loss': loss}, self.global_step)
        
        #IN CASE WE WANT TO SAVE ACC
        #y_arg = torch.argmax(y_hat,dim=1)
        #acc = self.acc_metric(y,y_arg)
        #acc= torch.tensor(acc)
        #self.logger.experiment.add_scalars("Acc", {' train acc' : acc}, self.global_step)

        return {'loss':loss}

    def training_epoch_end(self, outputs):
        '''
        Function called at the end of the training loop in one epoch
        outputs: values saved in the return of the training_step'''

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalars("Avg loss",{'train':avg_loss}, self.current_epoch) #graph in tensorboard

    def validation_step(self, batch, batch_nb):
        '''
        Operates on a single batch of data from the validation set. In this step you'd might generate 
        examples or calculate anything of interest like Dice.
        '''

        x, y, names, shapes= batch

        #save the first validation batch for a future prediction
        if(self.current_epoch==0):
            self.valid_sample = x,y, names, shapes

        y_hat= self.forward(x)

        #CALCULATE DICE
        loss = self.loss(y_hat,y)
        
        #CALCULATE DICE
        y_arg = torch.argmax(y_hat,dim=1)
        dice = self.dice_metric(y,y_arg)
        dice= torch.tensor(dice)

        self.logger.experiment.add_scalars("Loss", {'Val loss' : loss}, self.global_step)

        self.logger.experiment.add_scalar("Valid_DICE", dice, self.global_step)

        #SAVE THE LOG FOR THE CALLBACK FOR EARLY STOPPING AND FOR SAVE THE BEST MODEL IN FUNCTION OF THE DICE IN VALIDATION
        self.log('dice', dice) 

        return {'val_loss': loss, 'val_dice': dice}

    def validation_epoch_end(self, outputs):

        ''' al terminar la validacion se calcula el promedio de la loss de validacion'''

        
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_dice = torch.stack([x['val_dice'] for x in outputs]).mean()
        
        #SAVE THE VALUE IN TENSORBOARD
        self.logger.experiment.add_scalars("Avg loss",{'valid':avg_loss}, self.current_epoch)
        self.logger.experiment.add_scalar("valid_avg_dice",avg_dice, self.current_epoch)

        #TAKE THE FIRST VALIDATION BATCH SAVED AND PREDICT THE FIRST SAMPLE
        x, y, names, shapes = self.valid_sample
        pred = self.forward(x)

        #IN THE PREDICTION WITHOUT LOSS CALCULATION WE NEED TO ADD THE SOFTMAX LAYER
        pred = self.softmax(pred) #WE HAD THE PROBABILITIES 
        pred_arg = torch.argmax(pred, dim=1) #BINARY IMAGE OF THE SEGMENTATION

        #WE GENERATE THE IMAGE OF THE SEGMENTATION WITH THE tp, fp AND fn
        img = self.generate_img(pred_arg[0,:,:], y[0,:,:])

        #SAVE THE IMAGES IN TENSORBOARD
        self.logger.experiment.add_image("Img" + names[0], x[0,:,:,:] , dataformats="CHW")
        self.logger.experiment.add_image("ground truth", y[0,:,:], dataformats="HW")
        self.logger.experiment.add_image("predict", pred[0,1,:,:], dataformats="HW")
        self.logger.experiment.add_image("predict_arg", pred_arg[0,:,:], dataformats="HW")
        self.logger.experiment.add_image("prediction", img, dataformats="CHW")

        #IN CASE WE WANT TO SEE DE VALIDATION LOSS FOR EARLY STOPPING
        self.log('avg_val_loss', avg_loss)
     
        return {'avg_val_loss': avg_loss, 'avg_val_dice': avg_dice}

    def predict_step(self, batch, batch_idx):
        '''
        By default, the predict_step() method runs the forward() method. 
        In order to customize this behaviour, simply override the predict_step() method.
        '''
        dst_path = '/mnt/Almacenamiento/ODOC_segmentation/predicted/'
        x,y, names, shapes = batch
        y_hat = self.forward(x)
        pred = self.softmax(y_hat)
        pred_arg = torch.argmax(pred, dim=1) #BINARY IMAGE OF THE SEGMENTATION

        batch_s, c, h, w = x.size()
        for b in range(batch_s):
            name = 'pred_' + names[b] #NAME OF THE FILE
            self.count = self.count + 1


            img = (x[b,:,:,:] * 255).cpu()
            img = np.transpose(img, (1,2,0))
            img = np.clip(img, 0, 255)

            probability= (pred[b,1,:,:] * 255).cpu()
            probability = np.clip(probability, 0, 255)

            deep, h, w = shapes[0][b], shapes[1][b], shapes[2][b]
            deep = int(deep)
            h= int(h)
            w= int(w)
            #print('PROPABILITY SHAPE: ', probability.shape, (h,w))
            #scale = F.interpolate(probability,(h,w), mode='nearest')
            scaledImg = FT.resize(pred[b,:,:,:], (h,w), interpolation=transforms.InterpolationMode.NEAREST)
            #print('SCALED FINAL SHAPE: ', scaledImg.shape)

            #DEVOLVER A SIZE ANTES DE PASAR A BIANRIO

            binary = (torch.argmax(scaledImg, dim=0)).cpu()


            iio.imsave(dst_path + self.name +  '/binary/' + name, binary)
            iio.imsave(dst_path + self.name +  '/img/' + name, img)
            iio.imsave(dst_path + self.name +  '/probability/' + name, probability)

            #iio.imsave(dst_path + 'tp_fp_fn/' + name + '.png', tp_fp_fn)


        return

    def configure_optimizers(self):
        '''
        Configuration of the opimizer used in the model
        '''
        return torch.optim.Adam(self.parameters(), lr=0.1) 

    def generate_img(self, pred, true):
        '''
         In this method we generate img for TP, FP ans FN
        ------------------
        input:
        pred: img 2D with the prediction
        true: img 2D with the ground truth
        ----------------
        output
        a 3D image
        in red dimension we mark the FP
        in green dimension we mark the TP
        in Blue dimension we wark the FN 
        '''

        TP = pred * true #both images got 1
        FP = pred - true #is in prediction but not in true
        FP[FP == -1] = 0
        FN = true - pred #is in true but not in prediction
        FN[FN == -1] = 0

        img = self.tensor_to_image(TP,FP,FN)

        return img


    def tensor_to_image(self,green, red, blue):
        '''
        We want to transform a tensor into a image
        ---------------------
        input
        green: information for the green dimension
        red: information fot the red dimension
        blue: information for the blue dimension
        '''
        green = green*255
        red = red*255
        blue = blue*255

        #as we want to use numpy we need to go into a cpu instead of gpu
        green = green.cpu()
        red = red.cpu()
        blue = blue.cpu()

        green = np.array(green, dtype=np.uint8)
        red = np.array(red, dtype=np.uint8)
        blue = np.array(blue, dtype=np.uint8)

        tensor = np.array([red,green,blue])

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

        '''
        dont used
        but if we want to take parameter for the console we can use this'''

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



    