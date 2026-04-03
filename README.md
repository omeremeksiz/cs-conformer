# CS-Conformer: EEG-Based Cybersickness Detection

Implementation code for the paper "A Convolutional Transformer Model for EEG-Driven Cybersickness Detection in VR Experience" (SIU 2025).

## Project Structure
```
cs_conformer/
├── conformer.py          # Main model implementation
├── conformer.sh          # Training script
├── labels/              
│   ├── data_split_clustered.xlsx
│   ├── data_split_original.xlsx
│   ├── labels_clustered/     
│   ├── labels_original/      
│   ├── ssq_clustered.xlsx
│   └── ssq_original.xlsx
├── output/              # Model outputs and results
└── preprocessing/       # Data preprocessing scripts
    ├── convert_label.m
    ├── group_split_clustered.py
    ├── group_split_original.py
    ├── label_clustering.ipynb
    └── subject_merger.m
```

## Requirements
Install dependencies using:
```bash
pip install -r requirements.txt
```

Required packages:
```
torch
numpy
scipy
matplotlib
sklearn
einops
```

## Dataset
This implementation uses the [VRSA-FR Dataset](https://www.ivylab.kaist.ac.kr/database/360-vr-vrsa-fr). Please download and place the data in appropriate directories as per the project structure.

## Usage

### Direct Execution
```bash
python conformer.py \
    --eeg_data_path /path/to/eeg_clustered/ \
    --label_data_path /path/to/eeg_original/ \
    --eeg_time_points 80000 \
    --use_sigmoid \
    --sigmoid_scale 11.0 \
    --use_projection \
    --seed 715
```

### Using SLURM
```bash
sbatch conformer.sh
```

## Citation
```bibtex
@inproceedings{emeksiz2025conformer,
    title={A Convolutional Transformer Model for EEG-Driven Cybersickness Detection in VR Experience},
    author={Emeksiz, Ömer Sabri and Erzin, Engin and Yemez, Yücel and Sezgin, Tevfik Metin},
    booktitle={33rd Signal Processing and Communications Applications Conference (SIU)},
    year={2025}
}
``` 