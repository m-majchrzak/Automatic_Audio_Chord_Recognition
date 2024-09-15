#!/bin/bash 
#SBATCH --job-name=train
#SBATCH --account=mandziuk-lab
#SBATCH --gpus-per-node=1 
#SBATCH --mem=10GB
#SBATCH --output=logs/run_%j_output.txt
#SBATCH --error=logs/run_%j_error.txt
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=01142127@pw.edu.pl
#SBATCH --partition=long,short,experimental


echo "Activating conda"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate BTC

echo ""
echo "Validating"
echo ""

### 0. PRETRAINED BTC ###
#python validate.py --index=0 --kfold=0  --use_pretrained=True 
# python validate.py --index=0 --kfold=1  --use_pretrained=True
python validate.py --index=0 --kfold=2  --use_pretrained=True
python validate.py --index=0 --kfold=3  --use_pretrained=True
python validate.py --index=0 --kfold=4  --use_pretrained=True
python validate.py --index=0 --kfold=5  --use_pretrained=True


### 1. AAM ###
# python validate.py --dataset=aam --index=1 --kfold=0 --use_pretrained=False --restore_epoch=48
# python validate.py --dataset=aam --index=1 --kfold=1 --use_pretrained=False --restore_epoch=65
# python validate.py --dataset=aam --index=1 --kfold=2 --use_pretrained=False --restore_epoch=52
# python validate.py --dataset=aam --index=1 --kfold=3 --use_pretrained=False --restore_epoch=47
# python validate.py --dataset=aam --index=1 --kfold=4 --use_pretrained=False --restore_epoch=64
# python validate.py --dataset=aam --index=1 --kfold=5 --use_pretrained=False --restore_epoch=54

### 2. WINTERREISE ###
# python validate.py --dataset=winterreise --index=2 --kfold=0 --use_pretrained=False --restore_epoch=36
# python validate.py --dataset=winterreise --index=2 --kfold=1 --use_pretrained=False --restore_epoch=21
# python validate.py --dataset=winterreise --index=2 --kfold=2 --use_pretrained=False --restore_epoch=18
# python validate.py --dataset=winterreise --index=2 --kfold=3 --use_pretrained=False --restore_epoch=18
# python validate.py --dataset=winterreise --index=2 --kfold=4 --use_pretrained=False --restore_epoch=56
# python validate.py --dataset=winterreise --index=2 --kfold=5 --use_pretrained=False --restore_epoch=20


### 3. AAM + WINTERREISE ###
# python validate.py --index=3 --kfold=0 --dataset="aam_winterreise" --dataset1="aam" --dataset1_subset=192 --dataset1_multiplier=1 --dataset2="winterreise" --dataset2_subset=48 --dataset2_multiplier=4 --use_pretrained=False --restore_epoch=19
# python validate.py --index=3 --kfold=1 --dataset="aam_winterreise" --dataset1="aam" --dataset1_subset=192 --dataset1_multiplier=1 --dataset2="winterreise" --dataset2_subset=48 --dataset2_multiplier=4 --use_pretrained=False --restore_epoch=35
# python validate.py --index=3 --kfold=2 --dataset="aam_winterreise" --dataset1="aam" --dataset1_subset=192 --dataset1_multiplier=1 --dataset2="winterreise" --dataset2_subset=48 --dataset2_multiplier=4 --use_pretrained=False --restore_epoch=29
# python validate.py --index=3 --kfold=3 --dataset="aam_winterreise" --dataset1="aam" --dataset1_subset=192 --dataset1_multiplier=1 --dataset2="winterreise" --dataset2_subset=48 --dataset2_multiplier=4 --use_pretrained=False --restore_epoch=15
# python validate.py --index=3 --kfold=4 --dataset="aam_winterreise" --dataset1="aam" --dataset1_subset=192 --dataset1_multiplier=1 --dataset2="winterreise" --dataset2_subset=48 --dataset2_multiplier=4 --use_pretrained=False --restore_epoch=37
# python validate.py --index=3 --kfold=5 --dataset="aam_winterreise" --dataset1="aam" --dataset1_subset=192 --dataset1_multiplier=1 --dataset2="winterreise" --dataset2_subset=48 --dataset2_multiplier=4 --use_pretrained=False --restore_epoch=27

### FINETUNING PRETRAINED ###

### 5. WINTERREISE ###
# python validate.py --index=5 --kfold=0 --dataset="winterreise" --use_pretrained=True --restore_epoch=19
# python validate.py --index=5 --kfold=1 --dataset="winterreise" --use_pretrained=True --restore_epoch=32
# python validate.py --index=5 --kfold=2 --dataset="winterreise" --use_pretrained=True --restore_epoch=11
# python validate.py --index=5 --kfold=3 --dataset="winterreise" --use_pretrained=True --restore_epoch=24
# python validate.py --index=5 --kfold=4 --dataset="winterreise" --use_pretrained=True --restore_epoch=3
# python validate.py --index=5 --kfold=5 --dataset="winterreise" --use_pretrained=True --restore_epoch=16
