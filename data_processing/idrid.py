import numpy as np
import imageio as iio
import matplotlib.pyplot as plt
import glob
import ntpath
import collections
import data_preprocesing
from utils import crop_fov, crop_fov_2

proyect_path = '/mnt/Almacenamiento/ODOC_segmentation'
or_data_path = '/raw_data/'
dst_data_path = '/data/'
dataset = 'IDRID'

def get_mask(path):
    #for p in paths:
    name = ntpath.basename(path)
    n = name.split('.')
    value = n[0].split('_')
    name = '0' + value[1]

    mask = iio.imread(path)
    mask = mask - 0

    #iio.imwrite(proyect_path + dst_data_path + 'OD1/' + dataset + '/' + name + '.png',mask)
    #print('mask: ', name)
    return mask, name
    

def get_images(path):
    '''
    methodo donde de actualizara la imagen em la forma que se necesite
    '''

    #for p in paths:
    name = ntpath.basename(path)
    n = name.split('.')
    value = n[0].split('_')
    name = '0' + value[1]

    #print('IMAGE PATH: ', p)
    img = iio.imread(path)

    #iio.imwrite(proyect_path+dst_data_path+'images/' + dataset + '/' +  name + '.png',img)
    return img, name

def main():

    train = []
    validation = []
    test = []

    OD_anot = []

    imgs_paths = []

    #img for train
    for tr_img in glob.glob(proyect_path + or_data_path + dataset + '/A. Segmentation/1. Original Images/a. Training Set/IDRiD_*.jpg'):
        name = ntpath.basename(tr_img)
        n = name.split('.')
        name = n[0].split('_')
        
        imgs_paths.insert(len(imgs_paths), tr_img)
        train.insert(len(train), '0' + name[1] + '.png')

    #img for test
    for t_img in glob.glob(proyect_path + or_data_path + dataset + '/A. Segmentation/1. Original Images/b. Testing Set/IDRiD_*.jpg'):
        name = ntpath.basename(t_img)
        n = name.split('.')
        name = n[0].split('_')

        imgs_paths.insert(len(imgs_paths), t_img)
        test.insert(len(test), '0' + name[1] + '.png')

        #OD anotations
    for anot_tr_OD in glob.glob(proyect_path + or_data_path + dataset + '/A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/5. Optic Disc/IDRiD_*_OD.tif'):
        OD_anot.insert(len(OD_anot), anot_tr_OD)

    for anot_t_OD in glob.glob(proyect_path + or_data_path + dataset + '/A. Segmentation/2. All Segmentation Groundtruths/b. Testing Set/5. Optic Disc/IDRiD_*_OD.tif'):
        OD_anot.insert(len(OD_anot), anot_t_OD)

    for path in range(len(imgs_paths)):
        img,n_img = get_images(imgs_paths[path])
        mask, n_mask = get_mask(OD_anot[path])

        if (n_img + '.png') in test:
            iio.imwrite(proyect_path+dst_data_path+'images/' + dataset + '/Test/' +  n_img + '.png',img)
            iio.imwrite(proyect_path + dst_data_path + 'OD1/' + dataset + '/Test/' + n_mask + '.png',mask)
        else:
            new_img, new_mask = crop_fov(img, mask)

            iio.imwrite(proyect_path+dst_data_path+'images/' + dataset + '/' +  n_img + '.png',new_img)
            iio.imwrite(proyect_path + dst_data_path + 'OD1/' + dataset + '/' + n_mask + '.png',new_mask)

    last_test =[]
    for t in test:
        last_test.insert(len(last_test),'Test/'+ t)
    test = last_test
    data_preprocesing.save_split_file('/mnt/Almacenamiento/ODOC_segmentation/split', 'ODOC_segmentation', dataset, train, validation, test)

    return dataset, train, test

if __name__ == '__main__':
    main()



# img_path = '/mnt/Almacenamiento/ODOC_segmentation/data/images/IDRID/015.png'
# OD_path = '/mnt/Almacenamiento/ODOC_segmentation/data/OD1/IDRID/015.png'

# img = iio.imread(img_path)
# od = iio.imread(OD_path)


# fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
# ax0.imshow(img)
# ax1.imshow(img[:,:,0] * od)
# ax2.imshow(od)

# plt.show()