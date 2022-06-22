#!/bin/bash

python validation.py --config config_files/REFUGE_f-10_p-10.ini --dataset REFUGE --model_path lightning_logs/REFUGE_S1/version_0/checkpoints/REFUGE_f-10_p-10-epoch=94.ckpt --result_path new_REFUGE_S1.txt --etapa 0
python validation.py --config config_files/REFUGE_f-10_p-25.ini --dataset REFUGE --model_path lightning_logs/REFUGE_S1/version_1/checkpoints/REFUGE_f-10_p-25-epoch=69.ckpt --result_path new_REFUGE_S1.txt --etapa 0
python validation.py --config config_files/REFUGE_f-10_p-50.ini --dataset REFUGE --model_path lightning_logs/REFUGE_S1/version_2/checkpoints/REFUGE_f-10_p-50-epoch=134.ckpt --result_path new_REFUGE_S1.txt --etapa 0

python validation.py --config config_files/REFUGE_f-25_p-10.ini --dataset REFUGE --model_path lightning_logs/REFUGE_S1/version_3/checkpoints/REFUGE_f-25_p-10-epoch=189.ckpt --result_path new_REFUGE_S1.txt --etapa 0
python validation.py --config config_files/REFUGE_f-25_p-25.ini --dataset REFUGE --model_path lightning_logs/REFUGE_S1/version_4/checkpoints/REFUGE_f-25_p-25-epoch=74.ckpt --result_path new_REFUGE_S1.txt --etapa 0
python validation.py --config config_files/REFUGE_f-25_p-50.ini --dataset REFUGE --model_path lightning_logs/REFUGE_S1/version_5/checkpoints/REFUGE_f-25_p-50-epoch=174.ckpt --result_path new_REFUGE_S1.txt --etapa 0

python validation.py --config config_files/REFUGE_f-50_p-10.ini --dataset REFUGE --model_path lightning_logs/REFUGE_S1/version_6/checkpoints/REFUGE_f-50_p-10-epoch=164.ckpt --result_path new_REFUGE_S1.txt --etapa 0
python validation.py --config config_files/REFUGE_f-50_p-25.ini --dataset REFUGE --model_path lightning_logs/REFUGE_S1/version_7/checkpoints/REFUGE_f-50_p-25-epoch=159.ckpt --result_path new_REFUGE_S1.txt --etapa 0
python validation.py --config config_files/REFUGE_f-50_p-50.ini --dataset REFUGE --model_path lightning_logs/REFUGE_S1/version_8/checkpoints/REFUGE_f-50_p-50-epoch=139.ckpt --result_path new_REFUGE_S1.txt --etapa 0

