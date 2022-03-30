
import nntplib
from turtle import forward
from pip import main
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F

class DobleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DobleConv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias= False),
            nn.BatchNorm2d(out_channels), #cancela el bias por eso se pone en falso
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias= False),
            nn.BatchNorm2d(out_channels), #cancela el bias por eso se pone en falso
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self, in_channels= 3, out_channels= 1, features=[64,128,256,512]):
        super(UNET,self).__init__()

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool= nn.MaxPool2d(kernel_size=2, stride=2)

        #DOWN
        for feature in features:
            self.downs.append(DobleConv(in_channels,feature))
            in_channels = feature

        #UP
        for feature in reversed(features): #reversed porque va creciendo
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)) #agregamos el skip connnection
            self.ups.append(DobleConv(feature*2, feature))
        
        self.bottleneck = DobleConv(features[-1],features[-1]*2) #the part in the bottom f the unet
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self,x):
        skip_connection=[]
        for down in self.downs:
            x= down(x)
            skip_connection.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connection = skip_connection[::-1] #reverse the list

        for idx in range(0,len(self.ups),2): #cada dos porque para hacer up y 2conv
            x = self.ups[idx](x)
            skip_connection = skip_connection[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, skip_connection[2:])

        concat_skip = torch.cat((skip_connection,x), dim=1)
        x=self.ups[idx+1](concat_skip)

        return self.final_conv(x)

def test():
    x= torch.rand((3,1,160,160))
    model= UNET(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape

if __name__ == '__main__':
    test()
    

# import torch.nn as nn
# import pytorch_lightning as pl
# import torch

# class Block(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_ch, out_ch, 3)
#         self.relu  = nn.ReLU()
#         self.conv2 = nn.Conv2d(out_ch, out_ch, 3)
    
#     def forward(self, x):
#         return self.conv2(self.relu(self.conv1(x)))

# class Encoder(nn.Module):
#     def __init__(self, chs=(3,64,128,256,512,1024)):
#         super().__init__()
#         self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
#         self.pool       = nn.MaxPool2d(2)
    
#     def forward(self, x):
#         ftrs = []
#         for block in self.enc_blocks:
#             x = block(x)
#             ftrs.append(x)
#             x = self.pool(x)
#         return ftrs


# class Unet(pl.LightningModule):
#     def __init__(self, hparams):
#         super(Unet,self).__init__()
#         self.hparams = hparams

#         self.n_channels = hparams.n_channels
#         self.n_classes = hparams.n_classes
#         self.bilinear = True


# encoder = Encoder()
# # input image
# x    = torch.randn(1, 3, 572, 572)
# ftrs = encoder(x)
# for ftr in ftrs: print(ftr.shape)