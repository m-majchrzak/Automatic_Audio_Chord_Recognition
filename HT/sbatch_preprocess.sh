#!/bin/bash 
#SBATCH --job-name=preprocess
#SBATCH --account=mandziuk-lab
#SBATCH --time=1:00:00
#SBATCH --mem=100GB
#SBATCH --output=logs/run_%j_output.txt
#SBATCH --error=logs/run_%j_error.txt
# #SBATCG --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=01142127@pw.edu.pl
#SBATCH --partition=long,short,experimental


echo "Activating conda"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate BTC

echo ""
echo "Preprocessing"
echo ""

python Preprocessing.py



