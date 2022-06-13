from signal import Sigmasks
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np





class ToTensor():
    def __call__(self, img, mask1, mask2=None):
        #print('TO TENSOR')
        transform = transforms.ToTensor()

        if mask2:
            #print('SHAPE IMG IN TRANSFORM: ', transform(img).shape)
            return transform(img), transform(mask1), transform(mask2)
        #print('SDASKMAL')
        return transform(img), transform(mask1)

class MulTransform:
  def __init__(self,factor):
    self.factor = factor

  def __call__(self,img, mask1, mask2=None):
    #print('MULTRANSFORM')

    img *= self.factor
    if mask2:
        return img, mask1, mask2
    return img, mask1, mask2

class Hflip():     
    def __init__(self,probability= 0.5):
        self.probability= probability

    def __call__(self,img, mask1, mask2=None):
        '''
        aplica el flip horizontal tanto a la imagen como a las mascaras, dada una probabilidad
        '''
        #print('HFLIP')

        rand = torch.rand(1)
        if rand > self.probability:
            img = F.hflip(img)
            mask1 = F.hflip(mask1)
            if mask2:
                mask2 = F.hflip(mask2) 

        if mask2:
            return img, mask1,mask2
        else:
            return img, mask1


class Vflip():
    def __init__(self, probability= 0.5):
        self.probability = probability
    
    def __call__(self, img, mask1, mask2= None):
        '''
        aplica el flip vertical tanto a la imagen como a las mascaras, dada una probabilidad
        '''
        #print('HFLIP')

        rand = torch.rand(1)

        if int(rand) > self.probability:
            img = F.vflip(img)
            mask1 = F.vflip(mask1)
            if mask2:
                mask2 = F.vflip(mask2)

        if mask2:
            return img, mask1,mask2
        else:
            return img, mask1
        

class GaussianBlur:
    def __init__(self, sigma=1):
        self.sigma= sigma

    def __call__(self, img, mask1, mask2= None):
        #print('GAUSSIAN')

        k= 2 * int(3*self.sigma) + 1

        blurrer = transforms.GaussianBlur(kernel_size=(k,k),sigma= self.sigma)
        new_input = blurrer(img)

        if mask2:
            return new_input, mask1, mask2
        else:
            return new_input, mask1

class ColorJitter():
    def __init__(self, brightness=(0.0,0.1), contrast=(0.0,0.1), saturation=(0.0,0.1), hue=0):
        #ya poner el valor de random? se quedaria el mismo simepre si lo agrego aca
        #solo guardar rango de valores?
        self.brightness = brightness
        self.contrast = contrast
        self.saturation= saturation
        self.hue = float(hue)
        

    def __call__(self, img, mask1, mask2= None):
        #print('COLLOR')
        
        hue = np.clip(self.hue,-0.5,0.5)
        jitter = transforms.ColorJitter(brightness=float(self.brightness),
            contrast=float(self.contrast), saturation=float(self.saturation), 
            hue= hue)
        #toma un valor randon entre los dos dados
        #jitter = transforms.ColorJitter()

        new_input = jitter(img)
        #new_input = new_input.clip(0,1)
        if mask2:
            return new_input, mask1, mask2
        else:
            return new_input, mask1



class RandomAffine():
    def __init__(self, degrees, translate, scale):
        self.degrees= degrees
        self.translate= translate
        self.scale= scale

    def __call__(self, img, mask1, mask2=None):
        #print('AFFINE')

        #print('SE APLICO RANDOM AFFINE')
        dmin,dmax = self.degrees
        smin, smax = self.scale
        rand_degrees = (dmax-dmin)*torch.rand(1) + dmin
        rand_scale = (smax-smin)*torch.rand(1) + smin

        img= F.affine(img, angle=float(rand_degrees), translate=self.translate,
                            scale=rand_scale, shear=0)
        mask1 = F.affine(mask1, angle=float(rand_degrees), translate=self.translate, 
                            scale=rand_scale, shear=0)
        if mask2:
            mask2 = F.affine(mask2, angle=float(rand_degrees), translate=self.translate, 
                            scale=rand_scale, shear=0)
            return img, mask1, mask2
        else:
            return img, mask1


        
