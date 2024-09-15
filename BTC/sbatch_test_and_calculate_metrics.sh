#!/bin/bash 
#SBATCH --job-name=test_aam
#SBATCH --account=mandziuk-lab
# #SBATCH --ntasks=3
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
echo "Winterreise"
echo ""

# echo "Testing"
# python test.py --audio_dir '../data/Winterreise/audio_wav' --save_dir '../data/Winterreise/predictions/lab_1_4' --model_file='assets/checkpoint/idx_1_4/idx_1_064.pth.tar'

# echo "Calculating metrics"
# python calculate_metrics.py --gt_dir="../data/Winterreise/lab_majmin" --est_dir="../data/Winterreise/predictions/lab_1_4" --audio_dir="../data/Winterreise/audio_wav" --audio_type='wav'

echo "Testing"
python test.py --audio_dir='../data/Winterreise/audio_wav_test3' --save_dir='../data/Winterreise/predictions/lab_3_0' --model_file='assets/checkpoint/idx_3_0/idx_3_035.pth.tar'

echo "Calculating metrics"
python calculate_metrics.py --gt_dir="../data/Winterreise/lab_majmin" --est_dir="../data/Winterreise/predictions/lab_3_0" --audio_dir='../data/Winterreise/audio_wav_test3' --audio_type='wav'

# echo ""
# echo "AAM"
# echo ""

# echo "Testing"
# python test.py --audio_dir '../data/AAM/audio-mixes-mp3' --save_dir '../data/AAM/predictions/lab_2_0' --model_file='assets/checkpoint/idx_2_0/idx_2_069.pth.tar'
# echo "Calculating metrics"
# python calculate_metrics.py --gt_dir="../data/AAM/lab" --est_dir='../data/AAM/predictions/lab_2_0' --audio_dir="../data/AAM/audio-mixes-mp3" --audio_type='flac'

echo ""
echo "AAM"
echo ""

echo "Testing"
python test.py --audio_dir='../data/AAM/audio-mixes-mp3-test3' --save_dir='../data/AAM/predictions/lab_3_0' --model_file='assets/checkpoint/idx_3_0/idx_3_035.pth.tar'
echo "Calculating metrics"
python calculate_metrics.py --gt_dir="../data/AAM/lab" --est_dir='../data/AAM/predictions/lab_3_0' --audio_dir='../data/AAM/audio-mixes-mp3-test3' --audio_type='flac'

echo "Finished"