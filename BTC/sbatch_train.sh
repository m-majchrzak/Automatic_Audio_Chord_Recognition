#!/bin/bash 
#SBATCH --job-name=train
#SBATCH --account=mandziuk-lab
#SBATCH --gpus-per-node=1 
#SBATCH --time=5-00:00:00
#SBATCH --mem=100GB
#SBATCH --output=logs/run_%j_output.txt
#SBATCH --error=logs/run_%j_error.txt
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=01142127@pw.edu.pl
#SBATCH --partition=long,short,experimental


echo "Activating conda"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate BTC

echo ""
echo "Training"
echo ""


### 1. AAM ###
# python train.py --dataset=aam --index=1 --kfold=0 --use_pretrained=False
# python train.py --dataset=aam --index=1 --kfold=1 --use_pretrained=False
# python train.py --dataset=aam --index=1 --kfold=2 --use_pretrained=False
# python train.py --dataset=aam --index=1 --kfold=3 --use_pretrained=False
# python train.py --dataset=aam --index=1 --kfold=4 --use_pretrained=False
# python train.py --dataset=aam --index=1 --kfold=5 --use_pretrained=False
# python train.py --dataset=aam --index=1 --kfold=5 --use_pretrained=False --restore_epoch=41

### 2. WINTERREISE ###
# python train.py --dataset=winterreise --index=2 --kfold=0 --use_pretrained=False
# python train.py --dataset=winterreise --index=2 --kfold=1 --use_pretrained=False
# python train.py --dataset=winterreise --index=2 --kfold=2 --use_pretrained=False
# python train.py --dataset=winterreise --index=2 --kfold=3 --use_pretrained=False
# python train.py --dataset=winterreise --index=2 --kfold=4 --use_pretrained=False
# python train.py --dataset=winterreise --index=2 --kfold=5 --use_pretrained=False


### 3. AAM + WINTERREISE ###
# python train.py --index=3 --kfold=0 --dataset="aam_winterreise" --dataset1="aam" --dataset1_subset=192 --dataset1_multiplier=1 --dataset2="winterreise" --dataset2_subset=48 --dataset2_multiplier=4 --use_pretrained=False --restore_epoch=30
# python train.py --index=3 --kfold=1 --dataset="aam_winterreise" --dataset1="aam" --dataset1_subset=192 --dataset1_multiplier=1 --dataset2="winterreise" --dataset2_subset=48 --dataset2_multiplier=4 --use_pretrained=False
# python train.py --index=3 --kfold=2 --dataset="aam_winterreise" --dataset1="aam" --dataset1_subset=192 --dataset1_multiplier=1 --dataset2="winterreise" --dataset2_subset=48 --dataset2_multiplier=4 --use_pretrained=False
# python train.py --index=3 --kfold=3 --dataset="aam_winterreise" --dataset1="aam" --dataset1_subset=192 --dataset1_multiplier=1 --dataset2="winterreise" --dataset2_subset=48 --dataset2_multiplier=4 --use_pretrained=False
# python train.py --index=3 --kfold=4 --dataset="aam_winterreise" --dataset1="aam" --dataset1_subset=192 --dataset1_multiplier=1 --dataset2="winterreise" --dataset2_subset=48 --dataset2_multiplier=4 --use_pretrained=False

### FINETUNING PRETRAINED ###

### 4. AAM ###
# python train.py --index=4 --kfold=0 --dataset="aam" --use_pretrained=True 
# python train.py --index=4 --kfold=1 --dataset="aam" --use_pretrained=True
# python train.py --index=4 --kfold=2 --dataset="aam" --use_pretrained=True
# python train.py --index=4 --kfold=3 --dataset="aam" --use_pretrained=True
# python train.py --index=4 --kfold=4 --dataset="aam" --use_pretrained=True
# python train.py --index=4 --kfold=5 --dataset="aam" --use_pretrained=True

# python train.py --index=4 --kfold=0 --dataset="aam" --use_pretrained=True --restore_epoch=15
# python train.py --index=4 --kfold=1 --dataset="aam" --use_pretrained=True --restore_epoch=18
# python train.py --index=4 --kfold=2 --dataset="aam" --use_pretrained=True --restore_epoch=16
# python train.py --index=4 --kfold=3 --dataset="aam" --use_pretrained=True --restore_epoch=18
# python train.py --index=4 --kfold=4 --dataset="aam" --use_pretrained=True --restore_epoch=15
# python train.py --index=4 --kfold=5 --dataset="aam" --use_pretrained=True --restore_epoch=16

# python train.py --index=4 --kfold=0 --dataset="aam" --use_pretrained=True --restore_epoch=51
# python train.py --index=4 --kfold=1 --dataset="aam" --use_pretrained=True --restore_epoch=52
# python train.py --index=4 --kfold=3 --dataset="aam" --use_pretrained=True --restore_epoch=50
# python train.py --index=4 --kfold=4 --dataset="aam" --use_pretrained=True --restore_epoch=43
# python train.py --index=4 --kfold=5 --dataset="aam" --use_pretrained=True --restore_epoch=42

# python train.py --index=4 --kfold=0 --dataset="aam" --use_pretrained=True --restore_epoch=73
# python train.py --index=4 --kfold=3 --dataset="aam" --use_pretrained=True --restore_epoch=65
#python train.py --index=4 --kfold=5 --dataset="aam" --use_pretrained=True --restore_epoch=63

### 5. WINTERREISE ###
# python train.py --index=5 --kfold=0 --dataset="winterreise" --use_pretrained=True
# python train.py --index=5 --kfold=1 --dataset="winterreise" --use_pretrained=True
# python train.py --index=5 --kfold=2 --dataset="winterreise" --use_pretrained=True
# python train.py --index=5 --kfold=3 --dataset="winterreise" --use_pretrained=True
# python train.py --index=5 --kfold=4 --dataset="winterreise" --use_pretrained=True
# python train.py --index=5 --kfold=5 --dataset="winterreise" --use_pretrained=True

### 6. AAM + WINTERREISE ###
# python train.py --index=6 --kfold=0 --dataset="aam_winterreise" --dataset1="aam" --dataset1_subset=192 --dataset1_multiplier=1 --dataset2="winterreise" --dataset2_subset=48 --dataset2_multiplier=4 --use_pretrained=True
# python train.py --index=6 --kfold=1 --dataset="aam_winterreise" --dataset1="aam" --dataset1_subset=192 --dataset1_multiplier=1 --dataset2="winterreise" --dataset2_subset=48 --dataset2_multiplier=4 --use_pretrained=True
# python train.py --index=6 --kfold=2 --dataset="aam_winterreise" --dataset1="aam" --dataset1_subset=192 --dataset1_multiplier=1 --dataset2="winterreise" --dataset2_subset=48 --dataset2_multiplier=4 --use_pretrained=True
# python train.py --index=6 --kfold=3 --dataset="aam_winterreise" --dataset1="aam" --dataset1_subset=192 --dataset1_multiplier=1 --dataset2="winterreise" --dataset2_subset=48 --dataset2_multiplier=4 --use_pretrained=True
# python train.py --index=6 --kfold=4 --dataset="aam_winterreise" --dataset1="aam" --dataset1_subset=192 --dataset1_multiplier=1 --dataset2="winterreise" --dataset2_subset=48 --dataset2_multiplier=4 --use_pretrained=True
# python train.py --index=6 --kfold=5 --dataset="aam_winterreise" --dataset1="aam" --dataset1_subset=192 --dataset1_multiplier=1 --dataset2="winterreise" --dataset2_subset=48 --dataset2_multiplier=4 --use_pretrained=True

