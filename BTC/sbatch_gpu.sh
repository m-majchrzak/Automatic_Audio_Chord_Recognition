#!/bin/bash 
#SBATCH --job-name=train
#SBATCH --account=mandziuk-lab
#SBATCH --gpus-per-node=1 
#SBATCH --time=00:01:00
#SBATCH --mem=1GB
#SBATCH --output=logs/run_%j_output.txt
#SBATCH --error=logs/run_%j_error.txt
#SBATCG --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=01142127@pw.edu.pl

echo "Activating conda"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate BTC

echo ""
echo "Testing"
echo ""
python gpu_test.py