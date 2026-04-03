#!/bin/bash

#SBATCH --account=ai
#SBATCH --job-name=oemeksiz24
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --time=24:0:0
#SBATCH --mem=192G
#SBATCH -c 4 
#SBATCH --output=output/conformer/job_%j/%j.log  
#SBATCH --error=output/conformer/job_%j/%j.err

# python conformer.py --eeg_data_path ~/cs_conformer/data/eeg_clustered/ --label_data_path ~/cs_conformer/data/eeg_original/ --eeg_time_points 80000 --use_sigmoid --sigmoid_scale 11.0  --use_projection --seed 715 --use_time_reverse_augmentation
python conformer.py --eeg_data_path ~/cs_conformer/data/eeg_clustered/  --label_data_path ~/cs_conformer/data/eeg_original/ --eeg_time_points 80000 --use_sigmoid --sigmoid_scale 11.0 --use_projection --seed 715 --use_time_shift_augmentation