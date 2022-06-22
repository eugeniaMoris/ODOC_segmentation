#!/bin/bash

python validation.py --config config_files/MULTID_STEP2_f-10_p-10.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_DC.ini --model_path lightning_logs/MULTID_S2/version_0/checkpoints/MULTID_STEP2_f-10_p-10-epoch=179.ckpt --result_path MULTI_AUGM_S2.txt --etapa 1

python validation.py --config config_files/MULTID_STEP2_f-10_p-25.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_DC.ini --model_path lightning_logs/MULTID_S2/version_1/checkpoints/MULTID_STEP2_f-10_p-25-epoch=79.ckpt --result_path MULTI_AUGM_S2.txt --etapa 1

python validation.py --config config_files/MULTID_STEP2_f-10_p-50.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_DC.ini --model_path lightning_logs/MULTID_S2/version_2/checkpoints/MULTID_STEP2_f-10_p-50-epoch=89.ckpt --result_path MULTI_AUGM_S2.txt --etapa 1



python validation.py --config config_files/MULTID_STEP2_f-25_p-10.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_DC.ini --model_path lightning_logs/MULTID_S2/version_3/checkpoints/MULTID_STEP2_f-25_p-10-epoch=194.ckpt --result_path MULTI_AUGM_S2.txt --etapa 1

python validation.py --config config_files/MULTID_STEP2_f-25_p-25.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_DC.ini --model_path lightning_logs/MULTID_S2/version_4/checkpoints/MULTID_STEP2_f-25_p-25-epoch=169.ckpt --result_path MULTI_AUGM_S2.txt --etapa 1

python validation.py --config config_files/MULTID_STEP2_f-25_p-50.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_DC.ini --model_path lightning_logs/MULTID_S2/version_5/checkpoints/MULTID_STEP2_f-25_p-50-epoch=154.ckpt --result_path MULTI_AUGM_S2.txt --etapa 1


python validation.py --config config_files/MULTID_STEP2_f-50_p-10.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_DC.ini --model_path lightning_logs/MULTID_S2/version_6/checkpoints/MULTID_STEP2_f-50_p-10-epoch=59.ckpt --result_path MULTI_AUGM_S2.txt --etapa 1

python validation.py --config config_files/MULTID_STEP2_f-50_p-25.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_DC.ini --model_path lightning_logs/MULTID_S2/version_7/checkpoints/MULTID_STEP2_f-50_p-25-epoch=199.ckpt --result_path MULTI_AUGM_S2.txt --etapa 1

python validation.py --config config_files/MULTID_STEP2_f-50_p-50.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_DC.ini --model_path lightning_logs/MULTID_S2/version_8/checkpoints/MULTID_STEP2_f-50_p-50-epoch=139.ckpt --result_path MULTI_AUGM_S2.txt --etapa 1


