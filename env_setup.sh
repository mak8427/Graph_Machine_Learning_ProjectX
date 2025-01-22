# install micromamba
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
eval "$(./bin/micromamba shell hook -s posix)"

# channels
micromamba config append channels conda-forge 
micromamba config append channels pytorch
micromamba config append channels nvidia
micromamba config append channels pyg
micromamba config append channels dglteam

micromamba create -n graphlearning --yes python=3.11
micromamba activate graphlearning

micromamba install --quiet --yes \
    'ipyparallel=8.8.0' \
    'notebook=7.2.1' \
    'jupyterhub=4.1.5' \
    'jupyterlab=4.2.3' \
    scikit-learn \
    pandas \
    matplotlib \
    igraph \
    ipyvolume \
    ipython_genutils \
    optuna \
    dgl \
    nvitop \
    ogb \
    wandb \
    tabulate \
    plotly \
    nbterm

micromamba install --yes pytorch pytorch-cuda=12.1 pyg=*=*cu* -c pyg -c pytorch -c nvidia
pip install kaleido
