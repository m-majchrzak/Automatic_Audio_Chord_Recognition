#!/bin/bash 
#SBATCH --job-name=train
#SBATCH --account=mandziuk-lab
#SBATCH --gpus-per-node=1 
#SBATCH --time=10:00
#SBATCH --mem=1GB
#SBATCH --output=logs/run_%j_output.txt
#SBATCH --error=logs/run_%j_error.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=01142127@pw.edu.pl
#SBATCH --partition=short,long,experimental


echo "Activating conda"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate BTC

# echo "Mean and std"
python mean_and_std.py --kfold=0