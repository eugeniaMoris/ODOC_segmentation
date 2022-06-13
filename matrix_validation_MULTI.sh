#!/bin/bash

python validation.py --config config_files/multi/step1_multi__augm_f-10_p-10.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_OD.ini --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/multimodel/version_0/checkpoints/step1_multi__augm_f-10_p-10-epoch=110.ckpt --result_path matrix_augm_multiS1.txt --etapa 0

python validation.py --config config_files/multi/step1_multi__augm_f-10_p-25.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_OD.ini --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/multimodel/version_1/checkpoints/step1_multi__augm_f-10_p-25-epoch=66.ckpt --result_path matrix_augm_multiS1.txt --etapa 0

python validation.py --config config_files/multi/step1_multi__augm_f-10_p-50.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_OD.ini --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/multimodel/version_2/checkpoints/step1_multi__augm_f-10_p-50-epoch=107.ckpt --result_path matrix_augm_multiS1.txt --etapa 0

python validation.py --config config_files/multi/step1_multi__augm_f-10_p-75.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_OD.ini --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/multimodel/version_3/checkpoints/step1_multi__augm_f-10_p-75-epoch=118.ckpt --result_path matrix_augm_multiS1.txt --etapa 0

python validation.py --config config_files/multi/step1_multi__augm_f-25_p-10.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_OD.ini --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/multimodel/version_4/checkpoints/step1_multi__augm_f-25_p-10-epoch=83.ckpt --result_path matrix_augm_multiS1.txt --etapa 0

python validation.py --config config_files/multi/step1_multi__augm_f-25_p-25.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_OD.ini --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/multimodel/version_5/checkpoints/step1_multi__augm_f-25_p-25-epoch=125.ckpt --result_path matrix_augm_multiS1.txt --etapa 0

python validation.py --config config_files/multi/step1_multi__augm_f-25_p-50.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_OD.ini --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/multimodel/version_6/checkpoints/step1_multi__augm_f-25_p-50-epoch=64.ckpt --result_path matrix_augm_multiS1.txt --etapa 0

python validation.py --config config_files/multi/step1_multi__augm_f-25_p-75.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_OD.ini --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/multimodel/version_7/checkpoints/step1_multi__augm_f-25_p-75-epoch=128.ckpt --result_path matrix_augm_multiS1.txt --etapa 0

python validation.py --config config_files/multi/step1_multi__augm_f-50_p-10.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_OD.ini --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/multimodel/version_8/checkpoints/step1_multi__augm_f-50_p-10-epoch=125.ckpt --result_path matrix_augm_multiS1.txt --etapa 0

python validation.py --config config_files/multi/step1_multi__augm_f-50_p-25.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_OD.ini --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/multimodel/version_9/checkpoints/step1_multi__augm_f-50_p-25-epoch=127.ckpt --result_path matrix_augm_multiS1.txt --etapa 0

python validation.py --config config_files/multi/step1_multi__augm_f-50_p-50.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_OD.ini --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/multimodel/version_10/checkpoints/step1_multi__augm_f-50_p-50-epoch=118.ckpt --result_path matrix_augm_multiS1.txt --etapa 0

python validation.py --config config_files/multi/step1_multi__augm_f-50_p-75.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_OD.ini --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/multimodel/version_11/checkpoints/step1_multi__augm_f-50_p-75-epoch=118.ckpt --result_path matrix_augm_multiS1.txt --etapa 0

python validation.py --config config_files/multi/step1_multi__augm_f-75_p-10.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_OD.ini --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/multimodel/version_12/checkpoints/step1_multi__augm_f-75_p-10-epoch=38.ckpt --result_path matrix_augm_multiS1.txt --etapa 0

python validation.py --config config_files/multi/step1_multi__augm_f-75_p-25.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_OD.ini --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/multimodel/version_13/checkpoints/step1_multi__augm_f-75_p-25-epoch=125.ckpt --result_path matrix_augm_multiS1.txt --etapa 0

python validation.py --config config_files/multi/step1_multi__augm_f-75_p-50.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_OD.ini --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/multimodel/version_14/checkpoints/step1_multi__augm_f-75_p-50-epoch=125.ckpt --result_path matrix_augm_multiS1.txt --etapa 0

python validation.py --config config_files/multi/step1_multi__augm_f-75_p-75.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_OD.ini --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/multimodel/version_15/checkpoints/step1_multi__augm_f-75_p-75-epoch=121.ckpt --result_path matrix_augm_multiS1.txt --etapa 0
