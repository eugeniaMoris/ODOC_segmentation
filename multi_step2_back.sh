#!/bin/bash

python training.py --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_DC.ini --dataset multi --config config_files/multi/Step2_multi_augm_f-75_p-75.ini --etapa 1 --name multimodel_step2_back

python training.py --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_DC.ini --dataset multi --config config_files/multi/Step2_multi_augm_f-75_p-50.ini --etapa 1 --name multimodel_step2_back

python training.py --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_DC.ini --dataset multi --config config_files/multi/Step2_multi_augm_f-75_p-25.ini --etapa 1 --name multimodel_step2_back

python training.py --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_DC.ini --dataset multi --config config_files/multi/Step2_multi_augm_f-75_p-10.ini --etapa 1 --name multimodel_step2_back

python training.py --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_DC.ini --dataset multi --config config_files/multi/Step2_multi_augm_f-50_p-75.ini --etapa 1 --name multimodel_step2_back

python training.py --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_DC.ini --dataset multi --config config_files/multi/Step2_multi_augm_f-50_p-50.ini --etapa 1 --name multimodel_step2_back












