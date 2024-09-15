#!/bin/bash 
#SBATCH --job-name=multithreading-test
#SBATCH --account=mandziuk-lab
#SBATCH --mem=300GB
#SBATCH --cpus-per-task=20
#SBATCH --output=logs/run_%j_output.txt
#SBATCH --error=logs/run_%j_error.txt
#SBATCG --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=martyna.majchrzak.stud@pw.edu.pl
#SBATCH --time=1:00:00

echo "Activating conda"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate BTC

echo ""
echo "Creating dataset"
echo ""
python multithreading_test.py