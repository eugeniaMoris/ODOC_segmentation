import numpy as np
import imageio as iio
import matplotlib.pyplot as plt
import glob
import ntpath
import collections
import scipy.io
from utils import crop_fov, crop_fov_2

proyect_path = '/mnt/Almacenamiento/ODOC_segmentation'
or_data_path = '/raw_data/'
dst_data_path = '/data/'
dataset = 'ORIGA'

def get_mask(path):

    name = ntpath.basename(path)
    n = name.split('.')
    name = n[0]

    mat = scipy.io.loadmat(path)
    mask = mat['mask']

    mask_1 = mask.copy()
    mask_2 = mask.copy()

    mask_1.astype(np.uint8)
    mask_2.astype(np.uint8)


    mask_1[mask_1 == 2] = 255 # OD
    mask_1[mask_1 == 1] = 255 # OD


    mask_2[mask_2 == 1] = 0 # OC
    mask_2[mask_2 == 2] = 255 # OC

    return mask_1, mask_2

def get_images(path):

    name = ntpath.basename(path)
    n = name.split('.')
    names = n[0]

    img = iio.imread(path)
    return img, names



def main():
    
    anot = []

    imgs_paths = []

    #img for train
    for tr_img in sorted(glob.glob(proyect_path + or_data_path + dataset + '/images/*.jpg')):
        imgs_paths.insert(len(imgs_paths), tr_img)

    #img for test
    for t_img in sorted(glob.glob(proyect_path + or_data_path + dataset + '/manual marking/*.mat')):
        anot.insert(len(anot), t_img)

    for i in range(len(imgs_paths)):
        img,name = get_images(imgs_paths[i])
        OD, OC = get_mask(anot[i])

        #n_img, new_OD, new_OC = crop_fov_2(img, OD, OC)

        iio.imwrite(proyect_path+dst_data_path+'images/' + dataset + '/' +  name + '.png',img)
        iio.imwrite(proyect_path + dst_data_path + 'OD1/' + dataset + '/' + name + '.png', OD)
        iio.imwrite(proyect_path + dst_data_path + 'OC/' + dataset + '/' + name + '.png', OC)

    return dataset

if __name__ == '__main__':
    main()



# img_path = '/mnt/Almacenamiento/ODOC_segmentation/data/images/ORIGA/001.png'
# OC_path = '/mnt/Almacenamiento/ODOC_segmentation/data/OC/ORIGA/001.png'
# OD_path = '/mnt/Almacenamiento/ODOC_segmentation/data/OD1/ORIGA/001.png'


# img = iio.imread(img_path)
# OC = iio.imread(OC_path)
# OD = iio.imread(OD_path)


# fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
# ax0.imshow(img)
# ax1.imshow(img[:,:,0] - OC)
# ax2.imshow(img[:,:,0] - OD)

# plt.show()
