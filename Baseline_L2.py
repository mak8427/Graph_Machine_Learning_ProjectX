import torch
import torch_geometric as pyg
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn as nn
from torch.nn import Dropout, Linear
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, BatchNorm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from einops import rearrange
import numpy as np
import os
from collections import Counter
import networkx as nx
import matplotlib.pyplot as plt

from hik.data.scene import Scene
from hik.vis.pose import get_meta


# Example: define data_location
data_location = "data"

# Choose your dataset (options: "A", "B", "C", or "D")
dataset = "A"

# Number of frames in each sequence segment
sample_length = 30

# Load Scene object
scene = Scene.load_from_paths(
    dataset,
    f"{data_location}/poses/",
    f"{data_location}/scenes/",
    f"{data_location}/body_models/",
)

# Generate splits
splits = scene.get_splits(
    length=sample_length,
    stepsize=sample_length,
)

print(splits.keys())
# dict_keys(['poses3d', 'smpls', 'transforms', 'masks', 'activities', 'start_frames'])
