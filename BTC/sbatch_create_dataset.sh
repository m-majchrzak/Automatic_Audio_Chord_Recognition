#!/bin/bash 
#SBATCH --job-name=create-dataset
#SBATCH --account=mandziuk-lab
#SBATCH --mem=1GB
#SBATCH --output=logs/run_%j_output.txt
#SBATCH --error=logs/run_%j_error.txt
#SBATCG --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=martyna.majchrzak.stud@pw.edu.pl
#SBATCH --time=1:00:00
#SBATCH --partition=long,short,experimental

echo "Activating conda"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate BTC

echo ""
echo "Creating dataset"
echo ""
python create_audio_dataset.py