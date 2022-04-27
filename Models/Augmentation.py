from signal import Sigmasks
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np





class ToTensor():
    def __call__(self, sample):
        inputs, targets = sample
        transform = transforms.ToTensor()
        return transform(inputs), transform(targets)

class MulTransform:
  def __init__(self,factor):
    self.factor = factor

  def __call__(self,sample):
    inputs, target = sample
    inputs *= self.factor
    return inputs, target

class Hflip():     
    def __init__(self,probability= 0.5):
        self.probability= probability

    def __call__(self,sample):

        #print('SE APLICO HFLIP')
        inputs, target = sample

        rand = torch.rand(1)
        if rand > self.probability:
            new_inputs = F.hflip(inputs)
            new_target = F.hflip(target) 
            return new_inputs, new_target
        else:
            return inputs, target

class Vflip():
    def __init__(self, probability= 0.5):
        self.probability = probability
    
    def __call__(self, sample):

        #print('SE APLICO VFLIP')
        inputs, target = sample
        rand = torch.rand(1)

        if int(rand) > self.probability:
            new_inputs = F.vflip(inputs)
            new_target = F.vflip(target)
            return new_inputs, new_target
        else: 
            return inputs, target
        

class GaussianBlur:
    def __init__(self, sigma=1):
        self.sigma= sigma

    def __call__(self, sample):
        #print('SE APLICO GAUSSIAN BLUR')
        inputs, target = sample

        k= 2 * int(3*self.sigma) + 1

        blurrer = transforms.GaussianBlur(kernel_size=(k,k),sigma= self.sigma)
        new_input = blurrer(inputs)
        #new_input = new_input.clip(0,1)
        return new_input, target

class ColorJitter():
    def __init__(self, brightness=(0.0,0.1), contrast=(0.0,0.1), saturation=(0.0,0.1), hue=0):
        #ya poner el valor de random? se quedaria el mismo simepre si lo agrego aca
        #solo guardar rango de valores?
        self.brightness = brightness
        self.contrast = contrast
        self.saturation= saturation
        self.hue = float(hue)
        

    def __call__(self, sample):
        #print('SE APLICO COLOR JITTERING')
        inputs, target = sample
        
        hue = np.clip(self.hue,-0.5,0.5)
        jitter = transforms.ColorJitter(brightness=float(self.brightness),
            contrast=float(self.contrast), saturation=float(self.saturation), 
            hue= hue)
        #toma un valor randon entre los dos dados
        #jitter = transforms.ColorJitter()

        new_input = jitter(inputs)
        #new_input = new_input.clip(0,1)

        return new_input, target



class RandomAffine():
    def __init__(self, degrees, translate, scale):
        self.degrees= degrees
        self.translate= translate
        self.scale= scale

    def __call__(self, sample):
        #print('SE APLICO RANDOM AFFINE')
        inputs, target = sample
        dmin,dmax = self.degrees
        tmin, tmax = self.translate
        smin, smax = self.scale
        rand_degrees = (dmax-dmin)*torch.rand(1) + dmin
        rand_scale = (smax-smin)*torch.rand(1) + smin

        new_input= F.affine(inputs, angle=float(rand_degrees), translate=self.translate,
                            scale=rand_scale, shear=0)
        new_target = F.affine(target, angle=float(rand_degrees), translate=self.translate, 
                            scale=rand_scale, shear=0)
        #new_input = new_input.clip(0,1)
        return new_input, new_target


        
