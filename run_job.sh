#!/usr/bin/env bash

# SLURM directives for time and GPU type
slurm_directive="--time=01:00:00" # Adjust as needed

# General parameters
main="python edge_features.py"

# Wrapper script
wrapper="wrapper.sb"

# Submit the job
eval sbatch ${slurm_directive} ${wrapper} ${main}

