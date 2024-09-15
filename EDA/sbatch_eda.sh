#!/bin/sh 
#SBATCH --job-name=eda
#SBATCH --account=mandziuk-lab
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --mem=5gb                     # Job memory request
#SBATCH --time=1:00:00               # Time limit hrs:min:sec
#SBATCH --output=logs/run_%j_output.txt
#SBATCH --error=logs/run_%j_error.txt
#SBATCH --partition=long,short,experimental

# conda init
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate BTC
# pip freeze

echo "Activating conda"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate BTC

python EDA.py
