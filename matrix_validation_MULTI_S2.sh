#!/bin/bash

python validation.py --config config_files/multi/Step2_multi_augm_f-10_p-10.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_DC.ini --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/multimodel_step2/version_0/checkpoints/Step2_multi_augm_f-10_p-10-epoch=51.ckpt --result_path MULTI_AUGM_S2.txt --etapa 1

python validation.py --config config_files/multi/Step2_multi_augm_f-10_p-25.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_DC.ini --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/multimodel_step2/version_1/checkpoints/Step2_multi_augm_f-10_p-25-epoch=115.ckpt --result_path MULTI_AUGM_S2.txt --etapa 1

python validation.py --config config_files/multi/Step2_multi_augm_f-10_p-50.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_DC.ini --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/multimodel_step2/version_2/checkpoints/Step2_multi_augm_f-10_p-50-epoch=76.ckpt --result_path MULTI_AUGM_S2.txt --etapa 1

python validation.py --config config_files/multi/Step2_multi_augm_f-10_p-75.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_DC.ini --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/multimodel_step2/version_3/checkpoints/Step2_multi_augm_f-10_p-75-epoch=62.ckpt --result_path MULTI_AUGM_S2.txt --etapa 1

python validation.py --config config_files/multi/Step2_multi_augm_f-25_p-10.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_DC.ini --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/multimodel_step2/version_4/checkpoints/Step2_multi_augm_f-25_p-10-epoch=51.ckpt --result_path MULTI_AUGM_S2.txt --etapa 1

python validation.py --config config_files/multi/Step2_multi_augm_f-25_p-25.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_DC.ini --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/multimodel_step2/version_5/checkpoints/Step2_multi_augm_f-25_p-25-epoch=103.ckpt --result_path MULTI_AUGM_S2.txt --etapa 1

python validation.py --config config_files/multi/Step2_multi_augm_f-25_p-50.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_DC.ini --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/multimodel_step2/version_6/checkpoints/Step2_multi_augm_f-25_p-50-epoch=85.ckpt --result_path MULTI_AUGM_S2.txt --etapa 1

python validation.py --config config_files/multi/Step2_multi_augm_f-25_p-75.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_DC.ini --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/multimodel_step2/version_7/checkpoints/Step2_multi_augm_f-25_p-75-epoch=103.ckpt --result_path MULTI_AUGM_S2.txt --etapa 1

#python validation.py --config config_files/multi/Step2_multi_augm_f-50_p-10.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_DC.ini --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/multimodel_step2/version_8/checkpoints/DC_REFUGE_augm_f-50_p-10-epoch=11.ckpt --result_path MULTI_AUGM_S2.txt --etapa 1

python validation.py --config config_files/multi/Step2_multi_augm_f-50_p-25.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_DC.ini --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/multimodel_step2/version_8/checkpoints/Step2_multi_augm_f-50_p-25-epoch=111.ckpt --result_path MULTI_AUGM_S2.txt --etapa 1

python validation.py --config config_files/multi/Step2_multi_augm_f-50_p-50.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_DC.ini --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/multimodel_step2/version_9/checkpoints/Step2_multi_augm_f-50_p-50-epoch=98.ckpt --result_path MULTI_AUGM_S2.txt --etapa 1

python validation.py --config config_files/multi/Step2_multi_augm_f-50_p-75.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_DC.ini --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/multimodel_step2/version_10/checkpoints/Step2_multi_augm_f-50_p-75-epoch=107.ckpt --result_path MULTI_AUGM_S2.txt --etapa 1

#python validation.py --config config_files/multi/Step2_multi_augm_f-75_p-10.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_DC.ini --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/multimodel_step2/version_12/checkpoints/DC_DRISHTI_augm_f-75_p-10-epoch=107.ckpt --result_path drishti_etapa2.txt --etapa 1

#python validation.py --config config_files/multi/Step2_multi_augm_f-75_p-25.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_DC.ini --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/multimodel_step2/version_13/checkpoints/DC_DRISHTI_augm_f-75_p-25-epoch=117.ckpt --result_path drishti_etapa2.txt --etapa 1

#python validation.py --config config_files/multi/Step2_multi_augm_f-75_p-50.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_DC.ini --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/multimodel_step2/version_14/checkpoints/DC_DRISHTI_augm_f-75_p-50-epoch=127.ckpt --result_path drishti_etapa2.txt --etapa 1

#python validation.py --config config_files/multi/Step2_multi_augm_f-75_p-75.ini --dataset multi --split /mnt/Almacenamiento/ODOC_segmentation/split/multi-dataset_DC.ini --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/multimodel_step2/version_15heckpoints/DC_DRISHTI_augm_f-75_p-75-epoch=110.ckpt --result_path drishti_etapa2.txt --etapa 1
