#!/bin/bash

python validation.py --config config_files/REFUGE/DC_REFUGE_augm_f-10_p-10.ini --dataset REFUGE --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/refuge_step2/version_0/checkpoints/DC_REFUGE_augm_f-10_p-10-epoch=95.ckpt --result_path REFUGE_AUGM_S2.txt --etapa 1

python validation.py --config config_files/REFUGE/DC_REFUGE_augm_f-10_p-25.ini --dataset REFUGE --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/refuge_step2/version_1/checkpoints/DC_REFUGE_augm_f-10_p-25-epoch=129.ckpt --result_path REFUGE_AUGM_S2.txt --etapa 1

python validation.py --config config_files/REFUGE/DC_REFUGE_augm_f-10_p-50.ini --dataset REFUGE --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/refuge_step2/version_2/checkpoints/DC_REFUGE_augm_f-10_p-50-epoch=109.ckpt --result_path REFUGE_AUGM_S2.txt --etapa 1

python validation.py --config config_files/REFUGE/DC_REFUGE_augm_f-10_p-75.ini --dataset REFUGE --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/refuge_step2/version_3/checkpoints/DC_REFUGE_augm_f-10_p-75-epoch=80.ckpt --result_path REFUGE_AUGM_S2.txt --etapa 1

python validation.py --config config_files/REFUGE/DC_REFUGE_augm_f-25_p-10.ini --dataset REFUGE --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/refuge_step2/version_4/checkpoints/DC_REFUGE_augm_f-25_p-10-epoch=110.ckpt --result_path REFUGE_AUGM_S2.txt --etapa 1

python validation.py --config config_files/REFUGE/DC_REFUGE_augm_f-25_p-25.ini --dataset REFUGE --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/refuge_step2/version_5/checkpoints/DC_REFUGE_augm_f-25_p-25-epoch=126.ckpt --result_path REFUGE_AUGM_S2.txt --etapa 1

python validation.py --config config_files/REFUGE/DC_REFUGE_augm_f-25_p-50.ini --dataset REFUGE --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/refuge_step2/version_6/checkpoints/DC_REFUGE_augm_f-25_p-50-epoch=110.ckpt --result_path REFUGE_AUGM_S2.txt --etapa 1

python validation.py --config config_files/REFUGE/DC_REFUGE_augm_f-25_p-75.ini --dataset REFUGE --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/refuge_step2/version_7/checkpoints/DC_REFUGE_augm_f-25_p-75-epoch=89.ckpt --result_path REFUGE_AUGM_S2.txt --etapa 1

python validation.py --config config_files/REFUGE/DC_REFUGE_augm_f-50_p-10.ini --dataset REFUGE --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/refuge_step2/version_8/checkpoints/DC_REFUGE_augm_f-50_p-10-epoch=11.ckpt --result_path REFUGE_AUGM_S2.txt --etapa 1

python validation.py --config config_files/REFUGE/DC_REFUGE_augm_f-50_p-25.ini --dataset REFUGE --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/refuge_step2/version_9/checkpoints/DC_REFUGE_augm_f-50_p-25-epoch=124.ckpt --result_path REFUGE_AUGM_S2.txt --etapa 1

#python validation.py --config config_files/REFUGE/DC_REFUGE_augm_f-50_p-50.ini --dataset REFUGE --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/refuge_step2/version_10/checkpoints/DC_DRISHTI_augm_f-50_p-50-epoch=116.ckpt --result_path drishti_etapa2.txt --etapa 1

python validation.py --config config_files/REFUGE/DC_REFUGE_augm_f-50_p-75.ini --dataset REFUGE --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/refuge_step2/version_11/checkpoints/DC_REFUGE_augm_f-50_p-75-epoch=123.ckpt --result_path REFUGE_AUGM_S2.txt --etapa 1

#python validation.py --config config_files/REFUGE/DC_REFUGE_augm_f-75_p-10.ini --dataset REFUGE --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/refuge_step2/version_12/checkpoints/DC_DRISHTI_augm_f-75_p-10-epoch=107.ckpt --result_path drishti_etapa2.txt --etapa 1

#python validation.py --config config_files/REFUGE/DC_REFUGE_augm_f-75_p-25.ini --dataset REFUGE --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/refuge_step2/version_13/checkpoints/DC_DRISHTI_augm_f-75_p-25-epoch=117.ckpt --result_path drishti_etapa2.txt --etapa 1

#python validation.py --config config_files/REFUGE/DC_REFUGE_augm_f-75_p-50.ini --dataset REFUGE --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/refuge_step2/version_14/checkpoints/DC_DRISHTI_augm_f-75_p-50-epoch=127.ckpt --result_path drishti_etapa2.txt --etapa 1

#python validation.py --config config_files/REFUGE/DC_REFUGE_augm_f-75_p-75.ini --dataset REFUGE --model_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/refuge_step2/version_15/checkpoints/DC_DRISHTI_augm_f-75_p-75-epoch=110.ckpt --result_path drishti_etapa2.txt --etapa 1
