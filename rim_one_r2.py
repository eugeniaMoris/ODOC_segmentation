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

img_path = '/mnt/Almacenamiento/ODOC_segmentation/raw_data/RIM_ONE_R2/Glaucoma and glaucoma suspicious/Im256.jpg'
#OC_path = '/mnt/Almacenamiento/ODOC_segmentation/data/OC/RIGA-Magrabia/001.png'
OD_path = '/mnt/Almacenamiento/ODOC_segmentation/raw_data/RIM_ONE_R2/Glaucoma and glaucoma suspicious/Im256-gs.txt'


img = iio.imread(img_path)

mask = np.loadtxt(OD_path,delimiter=',')
print(mask)

#print(collections.Counter(OD[800,:])) #255, 0 y 128 para las marcas


plt.imshow(img)
plt.plot(mask)



plt.show()