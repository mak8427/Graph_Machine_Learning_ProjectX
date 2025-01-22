curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
eval "$(./bin/micromamba shell hook -s posix)"

# channels
micromamba config append channels conda-forge
micromamba config append channels pytorch
micromamba config append channels nvidia
micromamba config append channels pyg
micromamba config append channels dglteam

micromamba activate graphlearning
