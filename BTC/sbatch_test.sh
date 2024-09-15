#!/bin/bash 
#SBATCH --job-name=test_aam
#SBATCH --account=mandziuk-lab
#SBATCH --ntasks=3
#SBATCH --time=5:00:00
#SBATCH --mem=40GB
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
echo "Winterreise"
srun --exclusive --ntasks=1 python test.py --audio_dir '../data/Winterreise/audio_wav' --save_dir '../data/Winterreise/predictions/lab_without_training' --voca False
echo "Winterreise voca"
srun --exclusive --ntasks=1 python test.py --audio_dir '../data/Winterreise/audio_wav' --save_dir '../data/Winterreise/predictions/lab_without_training_voca' --voca True
echo "AAM"
srun --exclusive --ntasks=1 python test.py --audio_dir '../data/AAM/audio-mixes-mp3' --save_dir '../data/AAM/predictions/lab_without_training' --voca False
echo "Finished"


