from cgi import test
from email.mime import image
from re import sub
from unittest import main
import numpy as np
import imageio as iio
import matplotlib.pyplot as plt
import glob
import ntpath
import collections
import scipy.io
from scipy import ndimage
from skimage.morphology import disk, erosion
from skimage.util import compare_images
from skimage import filters
from skimage.transform import resize
#la idea es restar las imagenes de r1 y r2 para ver si hay imagenes repetidas

r1_paths = []
r2_paths = []

for p_img in sorted(glob.glob('/mnt/Almacenamiento/ODOC_segmentation/data/images/RIM_ONE_R1/*.png')):
    r1_paths.insert(len(r1_paths), p_img) #path de las imagenes

for p_img in sorted(glob.glob('/mnt/Almacenamiento/ODOC_segmentation/data/images/RIM_ONE_R2/*.png')):
    r2_paths.insert(len(r2_paths), p_img) #path de las imagenes

#levantar datos
#diferencia
#si es menor a x valor si almacena el nombre de ambas
#ver ambas imagenes
for r1_idx in r1_paths:
    for r2_idx in r2_paths:
        imgr1 = iio.imread(r1_idx)
        imgr2 = iio.imread(r2_idx)

        r2 = resize(imgr2[:,:,0],(imgr1[:,:,0]).shape)

        imgr1 = np.array(imgr1[:,:,0],dtype=np.uint8)
        r2 = np.array(r2,dtype=np.uint8)
    
        resta = compare_images(imgr1, r2, method='diff')
        
        #resta = imgr1[:,:,0]- r2
        print(np.sum(resta))

        if np.sum(resta) < 1000:
            plt.imshow(resta)
            plt.show()