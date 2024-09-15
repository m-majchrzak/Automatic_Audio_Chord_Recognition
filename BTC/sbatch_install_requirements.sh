#!/bin/sh 
#SBATCH --job-name=install_requirements
#SBATCH --account=mandziuk-lab
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --mem=5gb                     # Job memory request
#SBATCH --time=00:05:00               # Time limit hrs:min:sec
#SBATCH --output=logs/run_%j_output.txt
#SBATCH --error=logs/run_%j_error.txt

# conda init
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate BTC
# pip freeze

echo "Activating conda"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate BTC

echo "Installing requirements"
python -m pip install -r requirements.txt