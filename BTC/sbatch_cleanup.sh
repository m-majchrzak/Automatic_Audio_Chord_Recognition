#!/bin/bash 
#SBATCH --job-name=convert_arff_to_lab
#SBATCH --account=mandziuk-lab
#SBATCH --ntasks=3
#SBATCH --time=00:10:00
#SBATCH --mem=1GB
#SBATCH --output=logs/run_%j_output.txt
#SBATCH --error=logs/run_%j_error.txt

echo "Activating conda"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate BTC

echo ""
echo "Converting arff files to lab"
echo ""
cd utils
srun --exclusive --ntasks=1 python cleanup.py --folder_path='../../data/AAM/predictions/lab_without_training'
srun --exclusive --ntasks=1 python cleanup.py --folder_path='../../data/Winterreise/predictions/lab_test'
srun --exclusive --ntasks=1 python cleanup.py --folder_path='../../data/Winterreise/predictions/lab_test_voca'