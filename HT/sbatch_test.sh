#!/bin/sh 
#SBATCH --job-name=test
#SBATCH --account=mandziuk-lab
#SBATCH --gpus-per-node=1 
#SBATCH --mem=50GB                     # Job memory request
# #SBATCH --time=3:00:00               # Time limit hrs:min:sec
#SBATCH --output=logs/run_%j_output.txt
#SBATCH --error=logs/run_%j_error.txt

# conda init
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate BTC
# pip freeze

echo "Activating conda"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate BTC

### Billboard" ###
python test.py --index=7 --kfold=0 --last_best_epoch=27
# python test.py  --index=7 --kfold=1 --last_best_epoch=17
# python test.py  --index=7 --kfold=2 --last_best_epoch=24
# python test.py  --index=7 --kfold=3 --last_best_epoch=18
# python test.py  --index=7 --kfold=4 --last_best_epoch=15
# python test.py  --index=7 --kfold=5 --last_best_epoch=27


# ### AAM ###
# python test.py   --index=8 --kfold=0 --last_best_epoch=20
# python test.py   --index=8 --kfold=1 --last_best_epoch=55
# python test.py   --index=8 --kfold=2 --last_best_epoch=26
# python test.py   --index=8 --kfold=3 --last_best_epoch=5
# python test.py   --index=8 --kfold=4 --last_best_epoch=40
# python test.py   --index=8 --kfold=5 --last_best_epoch=16

# ### WINTERREISE ###
# python test.py   --index=9 --kfold=0 --last_best_epoch=40
# python test.py   --index=9 --kfold=1 --last_best_epoch=16
# python test.py   --index=9 --kfold=2 --last_best_epoch=22
# python test.py   --index=9 --kfold=3 --last_best_epoch=57
# python test.py   --index=9 --kfold=4 --last_best_epoch=40
# python test.py   --index=9 --kfold=5 --last_best_epoch=28