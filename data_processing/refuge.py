from cgi import test
import numpy as np
import imageio as iio
import matplotlib.pyplot as plt
import glob
import ntpath
import collections
from utils import crop_fov, crop_fov_2
import scipy.io
from data_preprocesing import save_split_file



proyect_path = '/mnt/Almacenamiento/ODOC_segmentation'
or_data_path = '/raw_data/'
dst_data_path = '/data/'
dataset = 'REFUGE'

def get_mask(path):
    
    mask = iio.imread(path)
    plt.imshow(mask)
    #print(collections.Counter(mask[1000,:])) #255, 0 y 128 para las marcas

    mask1 = np.copy(mask)
    mask2 = np.copy(mask)

    mask1.astype(np.uint8)
    mask2.astype(np.uint8)

    mask1[mask1 == 0] = 128 # OD
    mask2[mask2 == 128] = 255 # OC

    mask1[mask1 == 255] = 0 # OD
    mask1[mask1 == 128] = 255 # OD

    mask2[mask2 == 0] = 1 # OC
    mask2[mask2 == 255] = 0 # OC
    mask2[mask2 == 1] = 255


    #iio.imwrite(proyect_path + dst_data_path + 'OD1/' + dataset + '/' + names[p_i],mask1)
    #iio.imwrite(proyect_path + dst_data_path + 'OC/' + dataset + '/' + names[p_i],mask2)

    #OD SEGUIDO DE OC
    return mask1, mask2

def get_images(paths,names):
    for p_i in range(len(paths)):
        
        img = iio.imread(paths[p_i])
        #iio.imwrite(proyect_path+dst_data_path+'images/' + dataset + '/' +  names[p_i],img)
        return img


def main():
    img_name = 1
    #f"{img_name:03}"

    train = []
    validation = []
    test = []

    img_paths = []
    final_img_name = []
    mask_paths = []

    #img for train
    for tr_img in sorted(glob.glob(proyect_path + or_data_path + dataset + '/Training400-images/*/*.jpg')):
        img_paths.insert(len(img_paths), tr_img)
        final_img_name.insert(len(final_img_name),'{0:0=3d}.png'.format(img_name))
        train.insert(len(train),'{0:0=3d}.png'.format(img_name))
        img_name += 1
    
    for mask in sorted(glob.glob(proyect_path + or_data_path + dataset + '/Annotation-Training400/Disc_Cup_Masks/*/*.bmp')):
        mask_paths.insert(len(mask_paths), mask)


    #img for validation
    for tr_img in sorted(glob.glob(proyect_path + or_data_path + dataset + '/REFUGE-Validation400/*.jpg')):
        img_paths.insert(len(img_paths), tr_img)
        final_img_name.insert(len(final_img_name),'{0:0=3d}.png'.format(img_name))
        validation.insert(len(validation),'{0:0=3d}.png'.format(img_name))
        img_name += 1
    
    for mask in sorted(glob.glob(proyect_path + or_data_path + dataset + '/REFUGE-Validation400-GT/Disc_Cup_Masks/*.bmp')):
        mask_paths.insert(len(mask_paths), mask)

    #img for test
    for tr_img in sorted(glob.glob(proyect_path + or_data_path + dataset + '/test_dataset/Images/*.jpg')):
        img_paths.insert(len(img_paths), tr_img)
        final_img_name.insert(len(final_img_name),'{0:0=3d}.png'.format(img_name))
        test.insert(len(test),'{0:0=3d}.png'.format(img_name))
        img_name += 1
    
    for mask in sorted(glob.glob(proyect_path + or_data_path + dataset + '/test_dataset/GT/Disc_Cup_Masks/*.bmp')):
        mask_paths.insert(len(mask_paths), mask)

    #print(img_paths[-1], mask_paths[-1])
    #img_paths = img_paths[:10]
    for p_i in range(len(img_paths)):

        img = iio.imread(img_paths[p_i])
        OD, OC = get_mask(mask_paths[p_i])
        name = final_img_name[p_i]

        if name in test:
            iio.imwrite(proyect_path+dst_data_path+'images/' + dataset + '/Test/' +  name, img)
            iio.imwrite(proyect_path + dst_data_path + 'OD1/' + dataset + '/Test/' + name, OD)
            iio.imwrite(proyect_path + dst_data_path + 'OC/' + dataset + '/Test/' + name, OC)
        else:
            n_img, new_OD, new_OC = crop_fov_2(img, OD, OC)

            iio.imwrite(proyect_path+dst_data_path+'images/' + dataset + '/' +  name, n_img)
            iio.imwrite(proyect_path + dst_data_path + 'OD1/' + dataset + '/' + name, new_OD)
            iio.imwrite(proyect_path + dst_data_path + 'OC/' + dataset + '/' + name, new_OC)
        
        #print(dataset, train_img, validation_img, test_img)
    last_test =[]
    for t in test:
        last_test.insert(len(last_test),'Test/'+ t)
    test = last_test
    save_split_file('/mnt/Almacenamiento/ODOC_segmentation/split', 'ODOC_segmentation', dataset, train, validation, test)
    return dataset, train, validation, test



if __name__ == '__main__':
    main()
    
# main()
# img_path = '/mnt/Almacenamiento/ODOC_segmentation/data/images/REFUGE/002.png'
# OC_path = '/mnt/Almacenamiento/ODOC_segmentation/data/OC/REFUGE/002.png'
# OD_path = '/mnt/Almacenamiento/ODOC_segmentation/data/OD1/REFUGE/002.png'


# img = iio.imread(img_path)
# OC = iio.imread(OC_path)
# OD = iio.imread(OD_path)


# fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
# ax0.imshow(img)
# ax1.imshow(img[:,:,0] * OC)
# ax2.imshow(img[:,:,0] * (OD-OC))

# plt.show()