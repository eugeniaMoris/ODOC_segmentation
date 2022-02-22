#DRISHTI-GS1
#se utilizara las AVG boundaries ya que no se posee Softmap the la copa para los datos de validacion

import numpy as np
import glob
import ntpath
import imageio as iio
from skimage.draw import polygon
import matplotlib.pyplot as plt

proyect_path = '/mnt/Almacenamiento/ODOC_segmentation'
or_data_path = '/raw_data/'
dst_data_path = '/data/'
dataset = 'DRISHTI'

def generate_mask(paths, anot_type):
    for p in paths:
        name = ntpath.basename(p)
        n = name.split('.')
        img_name = n[0].split('_')

        print('Name img: ', img_name[1])
        new_img = []
        if img_name[1] != '001':
            im = iio.imread(p)
            img = np.array(im)
            row, col = img.shape
            new_img = np.zeros((row,col),dtype=np.uint8)


            pos = np.where(img == 1)[0]
            for i in range(row):
                for j in range(col):
                    value = img[i,j]
                    if value == 255:
                        new_img[i,j]= 255
            print(new_img.shape)

            if anot_type == 'OD':
                anot_type = 'OD1'
            iio.imwrite(proyect_path + dst_data_path + anot_type + '/' + dataset + '/' + str(img_name[1]) + '.png',new_img)
            



    return

def image_processing(paths):

    for p in paths:
        name = ntpath.basename(p)
        n = name.split('.')
        value = n[0].split('_')
        #print('value: ', value)
        if value[1] != '001':
            im = iio.imread(p)
            height, width,_ = im.shape
            #iio.imwrite(proyect_path+dst_data_path+'images/' + dataset + '/' + value[1] + '.png',im)

    return (height,width)

def main():
    train = []
    test = []

    OC_anot = []
    OD_anot = []

    imgs_paths = []

    #img for train
    for tr_img in glob.glob(proyect_path + or_data_path + dataset + '/Drishti-GS1_files/Training/Images/drishti*.png'):
        name = ntpath.basename(tr_img)
        n = name.split('.')
        
        imgs_paths.insert(len(imgs_paths), tr_img)
        train.insert(len(train), n[0])
    
    print('TERMINO PATH TRAIN')

    #img for test
    for t_img in glob.glob(proyect_path + or_data_path + dataset + '/Drishti-GS1_files/Test/Images/drishti*.png'):
        name = ntpath.basename(t_img)
        n = name.split('.')
        
        imgs_paths.insert(len(imgs_paths), t_img)
        test.insert(len(test), n[0])

    print('TERMINO PATH TEST')
    

    #OC anotations
    for anot_tr_OC in glob.glob(proyect_path + or_data_path + dataset + '/Drishti-GS1_files/Training/GT/drishti*/SoftMap/*cupseg*.png'):
        OC_anot.insert(len(OC_anot), anot_tr_OC)

    for anot_t_OC in glob.glob(proyect_path + or_data_path + dataset + '/Drishti-GS1_files/Test/Test_GT/drishti*/SoftMap/*cupseg*.png'):
        OC_anot.insert(len(OC_anot), anot_t_OC)

    print('TERMINO PATH OC')

    #OD anotations
    for anot_tr_OD in glob.glob(proyect_path + or_data_path + dataset + '/Drishti-GS1_files/Training/GT/drishti*/SoftMap/*ODseg*.png'):
        OD_anot.insert(len(OD_anot), anot_tr_OD)

    for anot_t_OD in glob.glob(proyect_path + or_data_path + dataset + '/Drishti-GS1_files/Test/Test_GT/drishti*/SoftMap/*ODseg*.png'):
        OD_anot.insert(len(OD_anot), anot_t_OD)

    print('TERMINO PATH OD')
    

    size = image_processing(imgs_paths)
    print('TERMINO PASAR IMAGENES')

    generate_mask(OC_anot, 'OC')
    print('TERMINO OC')

    generate_mask(OD_anot, 'OD')
    print('TERMINO OD')




if __name__ == '__main__':
    main()



# path_mask = '/mnt/Almacenamiento/ODOC_segmentation/raw_data/DRISHTI/Drishti-GS1_files/Training/GT/drishtiGS_002/SoftMap'
# cup = 'drishtiGS_002_cupsegSoftmap.png'
# OD= 'drishtiGS_002_ODsegSoftmap.png'




# fig, (ax0, ax1) = plt.subplots(1, 2)
# ax0.imshow(im)
# ax1.imshow(new_img)

# plt.show()
