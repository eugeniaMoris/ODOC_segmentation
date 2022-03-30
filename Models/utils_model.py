import torch
import torch.nn as nn

def get_activation_fn (activation_name):
    
    if activation_name == 'relu':
        return nn.ReLU(inplace=True)

def get_pooling(name, kernel = 2):
    if name == 'max':
        return nn.MaxPool2d(kernel)
    elif name == 'avg':
        return nn.AvgPool2d(kernel)