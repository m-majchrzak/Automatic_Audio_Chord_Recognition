#!/bin/bash 
#SBATCH --job-name=metrics
#SBATCH --account=mandziuk-lab
#SBATCH --ntasks=2
#SBATCH --time=3:00:00
#SBATCH --mem=1GB
#SBATCH --output=logs/run_%j_output.txt
#SBATCH --error=logs/run_%j_error.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=01142127@pw.edu.pl


echo "Activating conda"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate BTC

echo ""
echo "Calculating metrics"
echo ""

# echo "Winterreise"
srun --exclusive --ntasks=1 python calculate_metrics.py --gt_dir="../data/Winterreise/lab_majmin" --est_dir="../data/Winterreise/predictions/lab_without_training" --audio_dir="../data/Winterreise/audio_wav" --voca=False --audio_type='wav'

# echo "Winterreise voca"
# srun --exclusive --ntasks=1 python calculate_metrics.py --gt_dir="../data/Winterreise/lab_extended" --est_dir="../data/Winterreise/predictions/lab_without_training_voca" --audio_dir="../data/Winterreise/audio_wav" --voca=True --audio_type='wav'

echo "AAM"
srun --exclusive --ntasks=1 python calculate_metrics.py --gt_dir="../data/AAM/lab" --est_dir="../data/AAM/predictions/lab_without_training" --audio_dir="../data/AAM/audio-mixes-mp3" --voca=False --audio_type='flac'
#python calculate_metrics.py --gt_dir="../data/AAM_test/lab" --est_dir="../data/AAM_test/predictions/lab_without_training" --audio_dir="../data/AAM_test/audio-mixes-mp3" --voca=False --audio_type='flac'

echo "Finished"