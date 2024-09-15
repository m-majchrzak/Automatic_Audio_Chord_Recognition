#!/bin/bash 
#SBATCH --job-name=train_ht
#SBATCH --account=mandziuk-lab
#SBATCH --gpus-per-node=1 
#SBATCH --time=1-00:00:00
#SBATCH --mem=200GB
#SBATCH --output=logs/run_%j_output.txt
#SBATCH --error=logs/run_%j_error.txt
# #SBATCG --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=01142127@pw.edu.pl
#SBATCH --partition=long,short,experimental


echo "Activating conda"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate BTC

echo ""
echo "Training"
echo ""

### 7. Billboard" ###
#python train.py --dataset=billboard --index=7 --kfold=0 
# python train.py --dataset=billboard --index=7 --kfold=1 
# python train.py --dataset=billboard --index=7 --kfold=2 
# python train.py --dataset=billboard --index=7 --kfold=3 
# python train.py --dataset=billboard --index=7 --kfold=4 
#python train.py --dataset=billboard --index=7 --kfold=5 


### 8. AAM ###
# python train.py --dataset=aam --index=8 --kfold=0 
# python train.py --dataset=aam --index=8 --kfold=1 
# python train.py --dataset=aam --index=8 --kfold=2 
# python train.py --dataset=aam --index=8 --kfold=3 
# python train.py --dataset=aam --index=8 --kfold=4 
# python train.py --dataset=aam --index=8 --kfold=5 
# python train.py --dataset=aam --index=8 --kfold=5 

### 9. WINTERREISE ###
# python train.py --dataset=winterreise --index=9 --kfold=0 
# python train.py --dataset=winterreise --index=9 --kfold=1 
# python train.py --dataset=winterreise --index=9 --kfold=2 
# python train.py --dataset=winterreise --index=9 --kfold=3 
# python train.py --dataset=winterreise --index=9 --kfold=4 
# python train.py --dataset=winterreise --index=9 --kfold=5 

### 10. BILLBOARD_AAM ###

# python train.py --dataset=billboard_aam --index=10 --kfold=0 
# python train.py --dataset=billboard_aam --index=10 --kfold=1 
# python train.py --dataset=billboard_aam --index=10 --kfold=2 
# python train.py --dataset=billboard_aam --index=10 --kfold=3 
# python train.py --dataset=billboard_aam --index=10 --kfold=4 
# python train.py --dataset=billboard_aam --index=10 --kfold=5 

### 11. BILLBOARD_WINTERREISE ###

#python train.py --dataset=billboard_winterreise --index=11 --kfold=0 
# python train.py --dataset=billboard_winterreise --index=11 --kfold=1 
#python train.py --dataset=billboard_winterreise --index=11 --kfold=2 
# python train.py --dataset=billboard_winterreise --index=11 --kfold=3 
#python train.py --dataset=billboard_winterreise --index=11 --kfold=4 
#python train.py --dataset=billboard_winterreise --index=11 --kfold=5 

### 12. AAM_WINTERREISE ###

#python train.py --dataset=aam_winterreise --index=12 --kfold=0 
#python train.py --dataset=aam_winterreise --index=12 --kfold=1 
#python train.py --dataset=aam_winterreise --index=12 --kfold=2 
#python train.py --dataset=aam_winterreise --index=12 --kfold=3 
#python train.py --dataset=aam_winterreise --index=12 --kfold=4 
#python train.py --dataset=aam_winterreise --index=12 --kfold=5 

### 13. BILLBOARD_AAM_WINTERREISE ###
#python train.py --dataset=billboard_aam_winterreise --index=13 --kfold=0 
#python train.py --dataset=billboard_aam_winterreise --index=13 --kfold=1 
#python train.py --dataset=billboard_aam_winterreise --index=13 --kfold=2 
#python train.py --dataset=billboard_aam_winterreise --index=13 --kfold=3 
#python train.py --dataset=billboard_aam_winterreise --index=13 --kfold=4 
#python train.py --dataset=billboard_aam_winterreise --index=13 --kfold=5 


