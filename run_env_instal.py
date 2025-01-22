#!/bin/bash
#SBATCH --job-name=download_kaggle_dataset
#SBATCH --output=download_output.log
#SBATCH --error=download_error.log
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

# Load Python module (if needed) or activate your environment
# source activate myenv  # Uncomment if using a virtual environment
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
eval "$(./bin/micromamba shell hook -s posix)"

# channels
micromamba config append channels conda-forge
micromamba config append channels pytorch
micromamba config append channels nvidia
micromamba config append channels pyg
micromamba config append channels dglteam


micromamba activate graphlearning

pip install kaggle

# Set Kaggle credentials directory
#export KAGGLE_CONFIG_DIR=$HOME/.kaggle

# Run the Python script
python download_kaggle_dataset.py

