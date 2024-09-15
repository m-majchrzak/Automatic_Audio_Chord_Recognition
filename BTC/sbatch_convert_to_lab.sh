#!/bin/bash 
#SBATCH --job-name=convert_csv_to_lab
#SBATCH --account=mandziuk-lab
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --mem=1GB
#SBATCH --output=logs/run_%j_output.txt
#SBATCH --error=logs/run_%j_error.txt

echo "Activating conda"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate BTC
cd utils

echo ""
echo "Converting csv files to lab"
echo ""

python convert_csv_to_lab.py --root_path='../../data' --dataset_path='Winterreise' --csv_dir_path='ann_audio_chord' --lab_dir_path='lab_majmin' --audio_dir_path='audio_wav' --voca_type='majmin' --audio_type='wav'
python convert_csv_to_lab.py --root_path='../../data' --dataset_path='Winterreise' --csv_dir_path='ann_audio_chord' --lab_dir_path='lab_extended' --audio_dir_path='audio_wav' --voca_type='extended' --audio_type='wav'

# echo ""
# echo "Converting arff files to lab"
# echo ""

# python convert_arff_to_lab.py --root_path='../../data' --dataset_path='AAM'