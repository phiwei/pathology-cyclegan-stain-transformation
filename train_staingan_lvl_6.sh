#!/bin/bash
python main.py --run_name staingan6 --output_dir /mnt/hdd8tb/phiwei/staingan/ --param_file_path /home/phiwei/repos/pathology-cyclegan-stain-transformation/config/config_cyclegan_mod.yml --albumentations_path /home/phiwei/repos/pathology-cyclegan-stain-transformation/config/config_albumentations.yml --data_file_path /mnt/ssd2tb/phiwei/acrobat_train_low_res/6/
sleep 5m
python main.py --run_name staingan5 --output_dir /mnt/hdd8tb/phiwei/staingan/ --param_file_path /home/phiwei/repos/pathology-cyclegan-stain-transformation/config/config_cyclegan_mod.yml --albumentations_path /home/phiwei/repos/pathology-cyclegan-stain-transformation/config/config_albumentations.yml --data_file_path /mnt/ssd2tb/phiwei/acrobat_train_low_res/5/
