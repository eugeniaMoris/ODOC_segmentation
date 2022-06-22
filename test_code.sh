#!/bin/bash

python ODOC_segmentation.py --dataset DRISHTI --dataset_test DRISHTI --config_m1 config_files/DRISHTI_augm_f-75_p-25.ini --config_m2 config_files/DC_DRISHTI_augm_f-75_p-10.ini --model1_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/drishti_model/matrix/version_19/checkpoints/DRISHTI_augm_f-75_p-25-epoch\=109.ckpt --model2_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/drishti_etapa2/version_12/checkpoints/DC_DRISHTI_augm_f-75_p-10-epoch\=107.ckpt --result_path DRISHTI_SEGMENTATION.txt

python ODOC_segmentation.py --dataset DRISHTI --dataset_test REFUGE --config_m1 config_files/DRISHTI_augm_f-75_p-25.ini --config_m2 config_files/DC_DRISHTI_augm_f-75_p-10.ini --model1_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/drishti_model/matrix/version_19/checkpoints/DRISHTI_augm_f-75_p-25-epoch\=109.ckpt --model2_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/drishti_etapa2/version_12/checkpoints/DC_DRISHTI_augm_f-75_p-10-epoch\=107.ckpt --result_path REFUGE_SEGMENTATION.txt

python ODOC_segmentation.py --dataset DRISHTI --dataset_test ORIGA --config_m1 config_files/DRISHTI_augm_f-75_p-25.ini --config_m2 config_files/DC_DRISHTI_augm_f-75_p-10.ini --model1_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/drishti_model/matrix/version_19/checkpoints/DRISHTI_augm_f-75_p-25-epoch\=109.ckpt --model2_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/drishti_etapa2/version_12/checkpoints/DC_DRISHTI_augm_f-75_p-10-epoch\=107.ckpt --result_path ORIGA_SEGMENTATION.txt

python ODOC_segmentation.py --dataset DRISHTI --dataset_test RIM_ONE_R3 --config_m1 config_files/DRISHTI_augm_f-75_p-25.ini --config_m2 config_files/DC_DRISHTI_augm_f-75_p-10.ini --model1_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/drishti_model/matrix/version_19/checkpoints/DRISHTI_augm_f-75_p-25-epoch\=109.ckpt --model2_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/drishti_etapa2/version_12/checkpoints/DC_DRISHTI_augm_f-75_p-10-epoch\=107.ckpt --result_path RIM_ONE_R3_SEGMENTATION.txt


#multidataset
python ODOC_segmentation.py --dataset multi --dataset_test DRISHTI --config_m1 config_files/multi/step1_multi__augm_f-25_p-25.ini --config_m2 config_files/multi/Step2_multi_augm_f-50_p-50.ini --model1_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/multimodel/version_5/checkpoints/step1_multi__augm_f-25_p-25-epoch=125.ckpt --model2_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/multimodel_step2/version_9/checkpoints/Step2_multi_augm_f-50_p-50-epoch=98.ckpt --result_path DRISHTI_SEGMENTATION_ModelM.txt

python ODOC_segmentation.py --dataset multi --dataset_test AIROGS --config_m1 config_files/old/multi/step1_multi__augm_f-25_p-25.ini --config_m2 config_files/old/multi/Step2_multi_augm_f-50_p-50.ini --model1_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/multimodel/version_5/checkpoints/step1_multi__augm_f-25_p-25-epoch=125.ckpt --model2_path /mnt/Almacenamiento/ODOC_segmentation/lightning_logs/multimodel_step2/version_5/checkpoints/Step2_multi_augm_f-25_p-25-epoch=103.ckpt --result_path AIROGS_SEGMENTATION_ModelM.txt --pred True


#drishti model corregido
python ODOC_segmentation.py --dataset DRISHTI --dataset_test DRISHTI --split /mnt/Almacenamiento/ODOC_segmentation/split --config_m1 config_files/DRISHTI_f-50_p-50_400.ini --config_m2 config_files/DRISHTI_STEP2_f-10_p-25.ini --model1_path lightning_logs/DRISHTI_STEP1_400/version_8/checkpoints/DRISHTI_f-50_p-50_400-epoch=199.ckpt --model2_path /mnt/Almacenamiento/ODOC_segmentation/codigo/lightning_logs/DRISHTI_STEP2/version_1/checkpoints/DRISHTI_STEP2_f-10_p-25-epoch=199.ckpt --result_path MODELD_DRISHTI_SEGM.csv

#refuge model corregido
python ODOC_segmentation.py --dataset REFUGE --dataset_test DRISHTI --split /mnt/Almacenamiento/ODOC_segmentation/split --config_m1 config_files/REFUGE_f-50_p-25.ini --config_m2 config_files/REFUGE_STEP2_f-50_p-10.ini --model1_path lightning_logs/REFUGE_S1/version_7/checkpoints/REFUGE_f-50_p-25-epoch=159.ckpt --model2_path /mnt/Almacenamiento/ODOC_segmentation/codigo/lightning_logs/REFUGE_S2/version_6/checkpoints/REFUGE_STEP2_f-50_p-10-epoch=94.ckpt --result_path MODELR_DRISHTI_SEGM.csv

#multimodel model corregido
python ODOC_segmentation.py --dataset multi --dataset_test DRISHTI --split /mnt/Almacenamiento/ODOC_segmentation/split --config_m1 config_files/MULTID_f-25_p-50.ini --config_m2 config_files/MULTID_STEP2_f-50_p-25.ini --model1_path lightning_logs/MULTID_S1/version_5/checkpoints/MULTID_f-25_p-50-epoch=189.ckpt --model2_path lightning_logs/MULTID_S2/version_7/checkpoints/MULTID_STEP2_f-50_p-25-epoch=199.ckpt --result_path MODELM_DRISHTI_SEGM.csv
