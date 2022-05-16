#DRISHTI-GS1
#se utilizara las AVG boundaries ya que no se posee Softmap the la copa para los datos de validacion

import numpy as np
import glob
import ntpath
import imageio as iio
from skimage.draw import polygon
import matplotlib.pyplot as plt
import data_preprocesing
from utils import crop_fov, crop_fov_2
import cv2
from skimage.transform import resize

proyect_path = '/mnt/Almacenamiento/ODOC_segmentation'
or_data_path = '/raw_data/'
dst_data_path = '/data/'
dataset = 'DRISHTI'


def generate_mask(path):
    
    name = ntpath.basename(path)
    n = name.split('.')
    img_name = n[0].split('_')

    new_img = []
    if img_name[1] != '001':
        im = iio.imread(path)
        img = np.array(im)
        row, col = img.shape
        new_img = np.zeros((row,col),dtype=np.uint8)


        pos = np.where(img == 1)[0]
        for i in range(row):
            for j in range(col):
                value = img[i,j]
                if value == 255:
                    new_img[i,j]= 255
        
        return new_img, img_name[1]


def image_processing(path):

    im = None
    name = ntpath.basename(path)
    n = name.split('.')
    value = n[0].split('_')
    #print('value: ', value)
    if value[1] != '001':
        im = iio.imread(path)
        #iio.imwrite(proyect_path+dst_data_path+'images/' + dataset + '/' + value[1] + '.png',im)

    return im, value[1]

def main():
    train = []
    validation = []
    test = []

    OC_anot = []
    OD_anot = []

    imgs_paths = []

    #img for train
    for tr_img in sorted(glob.glob(proyect_path + or_data_path + dataset + '/Drishti-GS1_files/Training/Images/drishti*.png')):
        base_name = ntpath.basename(tr_img)
        n = base_name.split('.')
        name= n[0].split('_')
        
        imgs_paths.insert(len(imgs_paths), tr_img)
        if name[1] != '001':
            train.insert(len(train), name[1] + '.png')
    

    #img for test
    for t_img in sorted(glob.glob(proyect_path + or_data_path + dataset + '/Drishti-GS1_files/Test/Images/drishti*.png')):
        base_name = ntpath.basename(t_img)
        n = base_name.split('.')
        name= n[0].split('_')
        
        imgs_paths.insert(len(imgs_paths), t_img)

        if name[1] != '001':
            test.insert(len(test), name[1] + '.png')

    

    #OC anotations
    for anot_tr_OC in sorted(glob.glob(proyect_path + or_data_path + dataset + '/Drishti-GS1_files/Training/GT/drishti*/SoftMap/*cupseg*.png')):
        OC_anot.insert(len(OC_anot), anot_tr_OC)

    for anot_t_OC in sorted(glob.glob(proyect_path + or_data_path + dataset + '/Drishti-GS1_files/Test/Test_GT/drishti*/SoftMap/*cupseg*.png')):
        OC_anot.insert(len(OC_anot), anot_t_OC)


    #OD anotations
    for anot_tr_OD in sorted(glob.glob(proyect_path + or_data_path + dataset + '/Drishti-GS1_files/Training/GT/drishti*/SoftMap/*ODseg*.png')):
        OD_anot.insert(len(OD_anot), anot_tr_OD)

    for anot_t_OD in sorted(glob.glob(proyect_path + or_data_path + dataset + '/Drishti-GS1_files/Test/Test_GT/drishti*/SoftMap/*ODseg*.png')):
        OD_anot.insert(len(OD_anot), anot_t_OD)

    
    for proc in range(len(imgs_paths)):

        img, name = image_processing(imgs_paths[proc])

        if name != '001': #esta imagen no tiene ambas mascaras generadas por lo que se descarta
            mask1, n1 = generate_mask(OC_anot[proc])
            # print('TERMINO OC')

            mask2, n2 = generate_mask(OD_anot[proc])
            # print('TERMINO OD')

            if (name + '.png') in test:
                iio.imwrite(proyect_path+dst_data_path+'images/' + dataset + '/Test/' + name + '.png',img)
                iio.imwrite(proyect_path + dst_data_path + 'OC/' + dataset + '/Test/' + name + '.png',mask1)
                iio.imwrite(proyect_path + dst_data_path + 'OD1/' + dataset + '/Test/' + name + '.png',mask2)

            elif (name + '.png') not in test:
                n_img, n_mask1, n_mask2 = crop_fov_2(img, mask1, mask2)

                # n_img = n_img.copy()
                # n_mask1= n_mask1.copy()
                # n_mask2= n_mask2.copy()
                # print('TIPOS: ' ,type(n_img))
                # n_img = resize(n_img,(512, 512,3))
                # n_mask1 = resize(n_mask1, (512,512))
                # n_mask2 = resize(n_mask2, (512,512))
                # #vecinos mas cercanos



                iio.imwrite(proyect_path+dst_data_path+'images/' + dataset + '/' + name + '.png',n_img)
                iio.imwrite(proyect_path + dst_data_path + 'OC/' + dataset + '/' + name + '.png',n_mask1)
                iio.imwrite(proyect_path + dst_data_path + 'OD1/' + dataset + '/' + name + '.png',n_mask2)

            print(f'Imagen {name}, {n1}, {n2}: procesada')



    last_test =[]
    for t in test:
        last_test.insert(len(last_test),'Test/'+ t)
    test = last_test
    data_preprocesing.save_split_file('/mnt/Almacenamiento/ODOC_segmentation/split', 'ODOC_segmentation', dataset, train, validation, test)

    return dataset, train, test


if __name__ == '__main__':
    main()



# img_path = '/mnt/Almacenamiento/ODOC_segmentation/data/images/DRISHTI/015.png'
# cup_path = '/mnt/Almacenamiento/ODOC_segmentation/data/OC/DRISHTI/015.png'
# OD_path = '/mnt/Almacenamiento/ODOC_segmentation/data/OD1/DRISHTI/015.png'

# img = iio.imread(img_path)
# cup = iio.imread(cup_path)
# od = iio.imread(OD_path)


# fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
# ax0.imshow(img)
# ax1.imshow(img[:,:,0] * cup)
# ax2.imshow(img[:,:,0] * od)


# plt.show()
