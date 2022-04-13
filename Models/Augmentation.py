from signal import Sigmasks
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F





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
        inputs, target = sample

        rand = torch.rand(1)
        if rand > self.probability:
            new_inputs = F.hflip(inputs)
            new_target = F.hflip(target) 
            return new_inputs, new_target
        else:
            return inputs, target

class Vflip():
    def __init__(self, probability= 50):
        self.probability = probability
    
    def __call__(self, sample):
        inputs, target = sample
        rand = torch.rand(1)

        if int(rand) > self.probability:
            new_inputs = F.vflip(inputs)
            new_target = F.vflip(target)
            return new_inputs, new_target
        else: 
            return inputs, target
        

class GaussianBlur:
    def __init__(self, kernel_size=(5,9), sigma=(0.1,5)):
        self.kernel_size= kernel_size
        self.sigma= sigma

    def __call__(self, sample):
        inputs, target = sample
        blurrer = transforms.GaussianBlur(self.kernel_size, self.sigma)
        new_input = blurrer(inputs)
        return new_input, target

class ColorJitter():
    def __init__(self, brightness=(0.0,0.1), contrast=(0.0,0.1), saturation=(0.0,0.1), hue=0):
        #ya poner el valor de random? se quedaria el mismo simepre si lo agrego aca
        #solo guardar rango de valores?
        self.brightness_range = brightness
        self.contrast_range = contrast
        self.saturation_range= saturation
        self.hue_range= hue
        

    def __call__(self, sample):
        inputs, target = sample
        #get value random of the range
        bmin,bmax = self.brightness_range
        cmin, cmax = self.contrast_range
        smin, smax = self.saturation_range
        hmin, hmax = 0,0.5

        rand_b = (bmax-bmin)*torch.rand(1) + bmin
        rand_c = (cmax-cmin)*torch.rand(1) + cmin
        rand_s = (smax-smin)*torch.rand(1) + smin
        rand_h = (hmax-hmin)*torch.rand(1) + hmin

        jitter = transforms.ColorJitter(float(rand_b), float(rand_c), float(rand_s), float(rand_h))
        #jitter = transforms.ColorJitter()

        new_input = jitter(inputs)
        return new_input, target



class RandomAffine():
    def __init__(self, degrees, translate, scale):
        self.degrees= degrees
        self.translate= translate
        self.scale= scale

    def __call__(self, sample):
        inputs, target = sample
        dmin,dmax = self.degrees
        tmin, tmax = self.translate
        smin, smax = self.scale
        rand_degrees = (dmax-dmin)*torch.rand(1) + dmin
        rand_scale = (smax-smin)*torch.rand(1) + smin

        new_input= F.affine(inputs, angle=float(rand_degrees), translate=self.translate,
                            scale=rand_scale, shear=.5)
        new_target = F.affine(target, angle=float(rand_degrees), translate=self.translate, 
                            scale=rand_scale, shear=.5)
        return new_input, new_target


        
