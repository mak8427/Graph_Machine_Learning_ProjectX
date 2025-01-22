#!/usr/bin/env bash

# Slurm directive for GPU allocation
slurm_directive="--time=0-01:00:00"  # 1 hour, no specific GPU needed for this task

# General definitions
wrapper="run/wrapper.sb"
main="python download_kaggle_dataset.py"

# Submit the job
eval sbatch ${slurm_directive} -J kaggle-dataset-download ${wrapper} ${main}

