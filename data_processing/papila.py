from modulefinder import IMPORT_NAME
import numpy as np
import utils_papila
from glob import glob
import ntpath
import imageio as iio
import csv
import matplotlib.pyplot as plt

root = '/mnt/Almacenamiento/ODOC_segmentation/raw_data/PAPILA/'
img_path = '/mnt/Almacenamiento/ODOC_segmentation/raw_data/PAPILA/FundusImages'
contour_path = '/mnt/Almacenamiento/ODOC_segmentation/raw_data/PAPILA/ExpertsSegmentations/Contours'
dst_path = '/mnt/Almacenamiento/ODOC_segmentation/data/'

imgs_paths = []
od_ex1 = []
oc_ex1 = []

for p in sorted(glob(img_path + '/*.jpg')):
    imgs_paths.append(p)

for p in sorted(glob(contour_path + '/*_disc_exp1.txt')):
    od_ex1.append(p)

for p in sorted(glob(contour_path + '/*_cup_exp1.txt')):
    oc_ex1.append(p)

#print(len(imgs_paths), len(od_ex1), len(oc_ex1))
#print(imgs_paths[41], od_ex1[41], oc_ex1[41])

tag, eyeID, patID = utils_papila.get_diagnosis(root)
#print(tag[41],eyeID[41], patID[41]) #CLASIFICACION, RIGHT/LEFT(1/0), NUM_IMG
file_name = dst_path + 'images/PAPILA/labels.csv'
with open(file_name, 'a+') as tags:
    writer = csv.writer(tags)
    writer.writerow(['Name','label'])

for pos in range(len(imgs_paths)):
    name= ntpath.basename(imgs_paths[pos])
    name = name.replace('RET','')
    name = name.replace('.jpg','')
    img = iio.imread(imgs_paths[pos])
    od_mask = utils_papila.contour_to_mask(ntpath.basename(od_ex1[pos]),img.shape,root)
    oc_mask = utils_papila.contour_to_mask(ntpath.basename(oc_ex1[pos]),img.shape, root)

    od_mask[od_mask==1]=255
    oc_mask[oc_mask==1]=255


    iio.imsave(dst_path + 'images/PAPILA/'+name+'.png',img)
    iio.imsave(dst_path + 'OD1/PAPILA/'+name+'.png',od_mask)
    iio.imsave(dst_path + 'OC/PAPILA/'+name+'.png',oc_mask)
    
    with open(file_name, 'a+') as tags:
        writer = csv.writer(tags)
        if tag[pos] == 0:
            writer.writerow([name,'Non-glaucomatous'])
        elif tag[pos] == 1:
            writer.writerow([name,'Glaucomatous'])
        else:
            writer.writerow([name,'Suspicious'])


    



    #print(name)



