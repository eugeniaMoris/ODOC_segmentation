#!/bin/bash

python validation.py --config config_files/MULTID_f-10_p-10.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_OD.ini --model_path lightning_logs/MULTID_S1/version_0/checkpoints/MULTID_f-10_p-10-epoch=199.ckpt --result_path new_MULTID_S1.txt --etapa 0
python validation.py --config config_files/MULTID_f-10_p-25.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_OD.ini --model_path lightning_logs/MULTID_S1/version_1/checkpoints/MULTID_f-10_p-25-epoch=94.ckpt --result_path new_MULTID_S1.txt --etapa 0
python validation.py --config config_files/MULTID_f-10_p-50.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_OD.ini --model_path lightning_logs/MULTID_S1/version_2/checkpoints/MULTID_f-10_p-50-epoch=109.ckpt --result_path new_MULTID_S1.txt --etapa 0

python validation.py --config config_files/MULTID_f-25_p-10.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_OD.ini --model_path lightning_logs/MULTID_S1/version_3/checkpoints/MULTID_f-25_p-10-epoch=184.ckpt --result_path new_MULTID_S1.txt --etapa 0
python validation.py --config config_files/MULTID_f-25_p-25.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_OD.ini --model_path lightning_logs/MULTID_S1/version_4/checkpoints/MULTID_f-25_p-25-epoch=199.ckpt --result_path new_MULTID_S1.txt --etapa 0
python validation.py --config config_files/MULTID_f-25_p-50.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_OD.ini --model_path lightning_logs/MULTID_S1/version_5/checkpoints/MULTID_f-25_p-50-epoch=189.ckpt --result_path new_MULTID_S1.txt --etapa 0

python validation.py --config config_files/MULTID_f-50_p-10.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_OD.ini --model_path lightning_logs/MULTID_S1/version_6/checkpoints/MULTID_f-50_p-10-epoch=164.ckpt --result_path new_MULTID_S1.txt --etapa 0
python validation.py --config config_files/MULTID_f-50_p-25.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_OD.ini --model_path lightning_logs/MULTID_S1/version_7/checkpoints/MULTID_f-50_p-25-epoch=179.ckpt --result_path new_MULTID_S1.txt --etapa 0
python validation.py --config config_files/MULTID_f-50_p-50.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_OD.ini --model_path lightning_logs/MULTID_S1/version_8/checkpoints/MULTID_f-50_p-50-epoch=174.ckpt --result_path new_MULTID_S1.txt --etapa 0

