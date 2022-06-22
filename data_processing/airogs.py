from glob import glob
from importlib.resources import path
import ntpath
import pandas as pd
import csv
import imageio as io

proyect_path = '/mnt/Almacenamiento/ODOC_segmentation'
or_data_path = '/raw_data/'
dst_data_path = '/data/'
dataset= 'AIROGS/'

def get_img_names():
    fileName =  proyect_path + or_data_path + 'Rotterdam_EyePACS_AIROGS/train_labels.csv'
    names= []
    file_name = proyect_path + dst_data_path + 'images/' + dataset + 'labels.csv'
    with open(file_name, '+a') as tags:
        writer = csv.writer(tags)
        writer.writerow(['Name','label'])
        with open(fileName) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                tag = ''
                name = row[0].replace('TRAIN','') 
                if row[1]== 'NRG':
                    tag = 'Non-glaucomatous'
                    writer.writerow([name,tag])
                    names.append(name)
                elif row[1] == 'RG':
                    tag = 'Glaucomatous'
                    writer.writerow([name,tag])
                    names.append(name)
                else:
                    print(row[1])
                
    return names


names = get_img_names()
base_paths =proyect_path + or_data_path + 'Rotterdam_EyePACS_AIROGS/*/*.jpg'
paths = []
for p in glob(proyect_path + or_data_path + 'Rotterdam_EyePACS_AIROGS/0/*.jpg'):
    paths.append(p)
for p in glob(proyect_path + or_data_path + 'Rotterdam_EyePACS_AIROGS/1/*.jpg'):
    paths.append(p)
for p in glob(proyect_path + or_data_path + 'Rotterdam_EyePACS_AIROGS/2/*.jpg'):
    paths.append(p)
for p in glob(proyect_path + or_data_path + 'Rotterdam_EyePACS_AIROGS/3/*.jpg'):
    paths.append(p)
for p in glob(proyect_path + or_data_path + 'Rotterdam_EyePACS_AIROGS/4/*.jpg'):
    paths.append(p)
for p in glob(proyect_path + or_data_path + 'Rotterdam_EyePACS_AIROGS/5/*.jpg'):
    paths.append(p)
print('Cantidad paths',len(paths))

for p in paths:
    base_name = ntpath.basename(p)
    base_name = base_name.replace('TRAIN','')
    base_name = base_name.replace('.jpg','')


    img_g = io.imread(p)

    io.imsave(proyect_path + dst_data_path + 'images/' + dataset + '/' + base_name + '.png', img_g)




