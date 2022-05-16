#!/bin/bash

python validation.py --config config_files/DRISHTI_augm_f-10_p-10.ini --dataset DRISHTI --model_path lightning_logs/drishti_model/matrix/version_6/checkpoints/DRISHTI_augm_f-10_p-10-epoch=122.ckpt --result_path result_matrix.txt

python validation.py --config config_files/DRISHTI_augm_f-10_p-25.ini --dataset DRISHTI --model_path lightning_logs/drishti_model/matrix/version_7/checkpoints/DRISHTI_augm_f-10_p-25-epoch=108.ckpt --result_path result_matrix.txt

python validation.py --config config_files/DRISHTI_augm_f-10_p-50.ini --dataset DRISHTI --model_path lightning_logs/drishti_model/matrix/version_8/checkpoints/DRISHTI_augm_f-10_p-50-epoch=124.ckpt --result_path result_matrix.txt

python validation.py --config config_files/DRISHTI_augm_f-10_p-75.ini --dataset DRISHTI --model_path lightning_logs/drishti_model/matrix/version_9/checkpoints/DRISHTI_augm_f-10_p-75-epoch=123.ckpt --result_path result_matrix.txt

python validation.py --config config_files/DRISHTI_augm_f-25_p-10.ini --dataset DRISHTI --model_path lightning_logs/drishti_model/matrix/version_10/checkpoints/DRISHTI_augm_f-25_p-10-epoch=73.ckpt --result_path result_matrix.txt

python validation.py --config config_files/DRISHTI_augm_f-25_p-25.ini --dataset DRISHTI --model_path lightning_logs/drishti_model/matrix/version_11/checkpoints/DRISHTI_augm_f-25_p-25-epoch=115.ckpt --result_path result_matrix.txt

python validation.py --config config_files/DRISHTI_augm_f-25_p-50.ini --dataset DRISHTI --model_path lightning_logs/drishti_model/matrix/version_12/checkpoints/DRISHTI_augm_f-25_p-50-epoch=115.ckpt --result_path result_matrix.txt

python validation.py --config config_files/DRISHTI_augm_f-25_p-75.ini --dataset DRISHTI --model_path lightning_logs/drishti_model/matrix/version_13/checkpoints/DRISHTI_augm_f-25_p-75-epoch=118.ckpt --result_path result_matrix.txt

python validation.py --config config_files/DRISHTI_augm_f-50_p-10.ini --dataset DRISHTI --model_path lightning_logs/drishti_model/matrix/version_14/checkpoints/DRISHTI_augm_f-50_p-10-epoch=121.ckpt --result_path result_matrix.txt

python validation.py --config config_files/DRISHTI_augm_f-50_p-25.ini --dataset DRISHTI --model_path lightning_logs/drishti_model/matrix/version_15/checkpoints/DRISHTI_augm_f-50_p-25-epoch=128.ckpt --result_path result_matrix.txt

python validation.py --config config_files/DRISHTI_augm_f-50_p-50.ini --dataset DRISHTI --model_path lightning_logs/drishti_model/matrix/version_16/checkpoints/DRISHTI_augm_f-50_p-50-epoch=117.ckpt --result_path result_matrix.txt

python validation.py --config config_files/DRISHTI_augm_f-50_p-75.ini --dataset DRISHTI --model_path lightning_logs/drishti_model/matrix/version_17/checkpoints/DRISHTI_augm_f-50_p-75-epoch=112.ckpt --result_path result_matrix.txt

python validation.py --config config_files/DRISHTI_augm_f-75_p-10.ini --dataset DRISHTI --model_path lightning_logs/drishti_model/matrix/version_18/checkpoints/DRISHTI_augm_f-75_p-10-epoch=76.ckpt --result_path result_matrix.txt

python validation.py --config config_files/DRISHTI_augm_f-75_p-25.ini --dataset DRISHTI --model_path lightning_logs/drishti_model/matrix/version_19/checkpoints/DRISHTI_augm_f-75_p-25-epoch=109.ckpt --result_path result_matrix.txt

python validation.py --config config_files/DRISHTI_augm_f-75_p-50.ini --dataset DRISHTI --model_path lightning_logs/drishti_model/matrix/version_20/checkpoints/DRISHTI_augm_f-75_p-50-epoch=119.ckpt --result_path result_matrix.txt

python validation.py --config config_files/DRISHTI_augm_f-75_p-75.ini --dataset DRISHTI --model_path lightning_logs/drishti_model/matrix/version_21/checkpoints/DRISHTI_augm_f-75_p-75-epoch=117.ckpt --result_path result_matrix.txt
