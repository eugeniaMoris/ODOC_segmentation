#!/bin/bash

python validation.py --config config_files/REFUGE_STEP2_f-10_p-10.ini --dataset REFUGE --model_path lightning_logs/REFUGE_S2/version_0/checkpoints/REFUGE_STEP2_f-10_p-10-epoch=39.ckpt --result_path new_REFUGE_S2.txt --etapa 1
python validation.py --config config_files/REFUGE_STEP2_f-10_p-25.ini --dataset REFUGE --model_path lightning_logs/REFUGE_S2/version_1/checkpoints/REFUGE_STEP2_f-10_p-25-epoch=79.ckpt --result_path new_REFUGE_S2.txt --etapa 1
python validation.py --config config_files/REFUGE_STEP2_f-10_p-50.ini --dataset REFUGE --model_path lightning_logs/REFUGE_S2/version_2/checkpoints/REFUGE_STEP2_f-10_p-50-epoch=109.ckpt --result_path new_REFUGE_S2.txt --etapa 1


python validation.py --config config_files/REFUGE_STEP2_f-25_p-10.ini --dataset REFUGE --model_path lightning_logs/REFUGE_S2/version_3/checkpoints/REFUGE_STEP2_f-25_p-10-epoch=69.ckpt --result_path new_REFUGE_S2.txt --etapa 1
python validation.py --config config_files/REFUGE_STEP2_f-25_p-25.ini --dataset REFUGE --model_path lightning_logs/REFUGE_S2/version_4/checkpoints/REFUGE_STEP2_f-25_p-25-epoch=189.ckpt --result_path new_REFUGE_S2.txt --etapa 1
python validation.py --config config_files/REFUGE_STEP2_f-25_p-50.ini --dataset REFUGE --model_path lightning_logs/REFUGE_S2/version_5/checkpoints/REFUGE_STEP2_f-25_p-50-epoch=109.ckpt --result_path new_REFUGE_S2.txt --etapa 1

python validation.py --config config_files/REFUGE_STEP2_f-50_p-10.ini --dataset REFUGE --model_path lightning_logs/REFUGE_S2/version_6/checkpoints/REFUGE_STEP2_f-50_p-10-epoch=94.ckpt --result_path new_REFUGE_S2.txt --etapa 1
python validation.py --config config_files/REFUGE_STEP2_f-50_p-25.ini --dataset REFUGE --model_path lightning_logs/REFUGE_S2/version_7/checkpoints/REFUGE_STEP2_f-50_p-25-epoch=169.ckpt --result_path new_REFUGE_S2.txt --etapa 1
python validation.py --config config_files/REFUGE_STEP2_f-50_p-50.ini --dataset REFUGE --model_path lightning_logs/REFUGE_S2/version_8/checkpoints/REFUGE_STEP2_f-50_p-50-epoch=169.ckpt --result_path new_REFUGE_S2.txt --etapa 1

