#!/bin/bash

python validation.py --config config_files/DRISHTI_augm_f-10_p-10.ini --dataset REFUGE --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/refuge/version_0/checkpoints/DRISHTI_augm_f-10_p-10-epoch=115.ckpt --result_path REFUGE_augm_matrix.txt --etapa 0

python validation.py --config config_files/DRISHTI_augm_f-10_p-25.ini --dataset REFUGE --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/refuge/version_1/checkpoints/DRISHTI_augm_f-10_p-25-epoch=77.ckpt --result_path REFUGE_augm_matrix.txt --etapa 0

python validation.py --config config_files/DRISHTI_augm_f-10_p-50.ini --dataset REFUGE --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/refuge/version_2/checkpoints/DRISHTI_augm_f-10_p-50-epoch=97.ckpt --result_path REFUGE_augm_matrix.txt --etapa 0

python validation.py --config config_files/DRISHTI_augm_f-10_p-75.ini --dataset REFUGE --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/refuge/version_3/checkpoints/DRISHTI_augm_f-10_p-75-epoch=99.ckpt --result_path REFUGE_augm_matrix.txt --etapa 0

python validation.py --config config_files/DRISHTI_augm_f-25_p-10.ini --dataset REFUGE --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/refuge/version_4/checkpoints/DRISHTI_augm_f-25_p-10-epoch=100.ckpt --result_path REFUGE_augm_matrix.txt --etapa 0

python validation.py --config config_files/DRISHTI_augm_f-25_p-25.ini --dataset REFUGE --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/refuge/version_5/checkpoints/DRISHTI_augm_f-25_p-25-epoch=123.ckpt --result_path REFUGE_augm_matrix.txt --etapa 0

python validation.py --config config_files/DRISHTI_augm_f-25_p-50.ini --dataset REFUGE --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/refuge/version_6/checkpoints/DRISHTI_augm_f-25_p-50-epoch=87.ckpt --result_path REFUGE_augm_matrix.txt --etapa 0

python validation.py --config config_files/DRISHTI_augm_f-25_p-75.ini --dataset REFUGE --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/refuge/version_7/checkpoints/DRISHTI_augm_f-25_p-75-epoch=101.ckpt --result_path REFUGE_augm_matrix.txt --etapa 0

python validation.py --config config_files/DRISHTI_augm_f-50_p-10.ini --dataset REFUGE --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/refuge/version_8/checkpoints/DRISHTI_augm_f-50_p-10-epoch=87.ckpt --result_path REFUGE_augm_matrix.txt --etapa 0

python validation.py --config config_files/DRISHTI_augm_f-50_p-25.ini --dataset REFUGE --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/refuge/version_9/checkpoints/DRISHTI_augm_f-50_p-25-epoch=114.ckpt --result_path REFUGE_augm_matrix.txt --etapa 0

python validation.py --config config_files/DRISHTI_augm_f-50_p-50.ini --dataset REFUGE --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/refuge/version_10/checkpoints/DRISHTI_augm_f-50_p-50-epoch=122.ckpt --result_path REFUGE_augm_matrix.txt --etapa 0

python validation.py --config config_files/DRISHTI_augm_f-50_p-75.ini --dataset REFUGE --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/refuge/version_11/checkpoints/DRISHTI_augm_f-50_p-75-epoch=112.ckpt --result_path REFUGE_augm_matrix.txt --etapa 0

python validation.py --config config_files/DRISHTI_augm_f-75_p-10.ini --dataset REFUGE --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/refuge/version_12/checkpoints/DRISHTI_augm_f-75_p-10-epoch=73.ckpt --result_path REFUGE_augm_matrix.txt --etapa 0

python validation.py --config config_files/DRISHTI_augm_f-75_p-25.ini --dataset REFUGE --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/refuge/version_13/checkpoints/DRISHTI_augm_f-75_p-25-epoch=107.ckpt --result_path REFUGE_augm_matrix.txt --etapa 0

python validation.py --config config_files/DRISHTI_augm_f-75_p-50.ini --dataset REFUGE --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/refuge/version_14/checkpoints/DRISHTI_augm_f-75_p-50-epoch=109.ckpt --result_path REFUGE_augm_matrix.txt --etapa 0

python validation.py --config config_files/DRISHTI_augm_f-75_p-75.ini --dataset REFUGE --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/refuge/version_15/checkpoints/DRISHTI_augm_f-75_p-75-epoch=128.ckpt --result_path REFUGE_augm_matrix.txt --etapa 0
