import numpy as np
import imageio as iio
import matplotlib.pyplot as plt
import glob
import ntpath
import collections

proyect_path = '/mnt/Almacenamiento/ODOC_segmentation'
or_data_path = '/raw_data/'
dst_data_path = '/data/'
dataset = 'IDRID'

def get_mask(paths):
    for p in paths:
        name = ntpath.basename(p)
        n = name.split('.')
        value = n[0].split('_')
        name = '0' + value[1]

        mask = iio.imread(p)
        mask = mask - 0

        iio.imwrite(proyect_path + dst_data_path + 'OD1/' + dataset + '/' + name + '.png',mask)
        print('mask: ', name)

def get_images(paths):
    '''
    methodo donde de actualizara la imagen em la forma que se necesite
    '''

    for p in paths:
        name = ntpath.basename(p)
        n = name.split('.')
        value = n[0].split('_')
        name = '0' + value[1]

        #print('IMAGE PATH: ', p)
        img = iio.imread(p)

        iio.imwrite(proyect_path+dst_data_path+'images/' + dataset + '/' +  name + '.png',img)

def main():

    train = []
    test = []

    OD_anot = []

    imgs_paths = []

    #img for train
    for tr_img in glob.glob(proyect_path + or_data_path + dataset + '/A. Segmentation/1. Original Images/a. Training Set/IDRiD_*.jpg'):
        name = ntpath.basename(tr_img)
        n = name.split('.')
        name = n[0].split('_')
        
        imgs_paths.insert(len(imgs_paths), tr_img)
        train.insert(len(train), name[1])

    #img for test
    for t_img in glob.glob(proyect_path + or_data_path + dataset + '/A. Segmentation/1. Original Images/b. Testing Set/IDRiD_*.jpg'):
        name = ntpath.basename(t_img)
        n = name.split('.')
        name = n[0].split('_')

        imgs_paths.insert(len(imgs_paths), t_img)
        test.insert(len(test), name[1])

        #OD anotations
    for anot_tr_OD in glob.glob(proyect_path + or_data_path + dataset + '/A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/5. Optic Disc/IDRiD_*_OD.tif'):
        OD_anot.insert(len(OD_anot), anot_tr_OD)

    for anot_t_OD in glob.glob(proyect_path + or_data_path + dataset + '/A. Segmentation/2. All Segmentation Groundtruths/b. Testing Set/5. Optic Disc/IDRiD_*_OD.tif'):
        OD_anot.insert(len(OD_anot), anot_t_OD)

    get_images(imgs_paths)
    get_mask(OD_anot)

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