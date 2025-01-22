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
data_location = "pose-dataset/data"

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




poses = rearrange(
    splits["poses3d"],
    "n_seq n_frames n_person n_joints n_xyz -> (n_seq n_person) n_frames n_xyz n_joints"
)

masks = rearrange(
    splits["masks"],
    "n_seq n_frames n_person -> (n_seq n_person) n_frames"
)

activities = rearrange(
    splits["activities"],
    "n_seq n_frames n_person n_activities -> (n_seq n_person) n_frames n_activities"
)

print("poses shape:", poses.shape)         # [N, n_frames, n_xyz, n_joints]
print("masks shape:", masks.shape)         # [N, n_frames]
print("activities shape:", activities.shape) # [N, n_frames, n_activities]



# Convert masks to bool, in case they're not already
masks_bool = masks.astype(bool)

# sequences_to_keep: boolean array of shape [N]
sequences_to_keep = np.all(masks_bool, axis=1)  # True if ALL frames are valid in that sequence

poses = poses[sequences_to_keep]
masks = masks[sequences_to_keep]
activities = activities[sequences_to_keep]

print("Filtered poses shape:", poses.shape)



downsample_rate = 6
poses_downsampled = poses[:, ::downsample_rate, :, :]
masks_downsampled = masks[:, ::downsample_rate]
activities_downsampled = activities[:, ::downsample_rate, :]

print("Downsampled poses shape:", poses_downsampled.shape)
# e.g. [N, 5, 3, 29] if your sample_length was 30


L = poses_downsampled.shape[1]  # number of frames after downsampling
center_frame_idx = L // 2

# We'll store the center frame's activity labels for classification
activities_center = activities_downsampled[:, center_frame_idx, :]

print("Center frame activity labels shape:", activities_center.shape)
# [N, n_activities]



body_edges= [
    (1, 0),
    (0, 2),
    (1, 4),
    (4, 7),
    (7, 10),
    (2, 5),
    (5, 8),
    (8, 11),
    (16, 18),
    (18, 20),
    (17, 19),
    (19, 21),
    (16, 13),
    (17, 14),
    (13, 12),
    (14, 12),
    (12, 15),
    (15, 23),
    (24, 25),
    (24, 26),
    (25, 27),
    (26, 28),
]




#Edited part



def build_spatio_temporal_edges(n_frames, n_joints, body_edges):
    """
    Returns a list of edges for the spatio-temporal graph given the skeleton edges and the number of frames.
    Also returns a list of edge types for computing edge features.
    """
    edges = []
    edge_types = []  # To distinguish spatial vs temporal edges
    for t in range(n_frames):
        # Spatial edges in each frame
        for (j1, j2) in body_edges:
            node1 = t * n_joints + j1
            node2 = t * n_joints + j2
            edges.append((node1, node2))
            edges.append((node2, node1))
            edge_types.append("spatial")
            edge_types.append("spatial")

        # Temporal edges between frames
        if t < n_frames - 1:
            for j in range(n_joints):
                node_current = t * n_joints + j
                node_next = (t + 1) * n_joints + j
                edges.append((node_current, node_next))
                edges.append((node_next, node_current))
                edge_types.append("temporal")
                edge_types.append("temporal")

    return edges, edge_types


def build_graph_data(pose_sequence, body_edges):
    """
    pose_sequence: array of shape [n_frames, n_joints, n_xyz] or [n_frames, n_xyz, n_joints]

    Returns:
    - node_features: torch.FloatTensor [n_frames * n_joints, 3]
    - edge_index: torch.LongTensor [2, #edges]
    - edge_features: torch.FloatTensor [#edges, #edge_features]
    """
    # Check dimensions of input
    if pose_sequence.shape[-1] == 3:  # [n_frames, n_joints, n_xyz]
        node_features = pose_sequence.reshape(-1, 3)
    elif pose_sequence.shape[1] == 3:  # [n_frames, n_xyz, n_joints]
        pose_sequence = pose_sequence.transpose(0, 2, 1)  # Convert to [n_frames, n_joints, n_xyz]
        node_features = pose_sequence.reshape(-1, 3)
    else:
        raise ValueError("Unexpected shape for pose_sequence:", pose_sequence.shape)

    # Determine n_frames and n_joints
    n_frames = pose_sequence.shape[0]
    n_joints = pose_sequence.shape[1]

    # Build edges
    edges, edge_types = build_spatio_temporal_edges(n_frames, n_joints, body_edges)

    # Compute edge features
    edge_features = []
    for (src, dest), edge_type in zip(edges, edge_types):
        if edge_type == "spatial":
            # Spatial: Compute Euclidean distance between joints
            dist = torch.norm(node_features[src] - node_features[dest])
            edge_features.append([dist])  # Single feature for spatial edges
        elif edge_type == "temporal":
            # Temporal: Compute displacement vector magnitude between frames
            disp = torch.norm(node_features[src] - node_features[dest])
            edge_features.append([disp])  # Single feature for temporal edges

    # Convert edges and features to PyTorch tensors
    edge_index = torch.LongTensor(edges).T  # shape [2, #edges]
    edge_features = torch.FloatTensor(edge_features)  # shape [#edges, #features_per_edge]

    return torch.FloatTensor(node_features), edge_index, edge_features


graph_list = []
labels_list = []

for i in range(poses_downsampled.shape[0]):
    pose_seq = poses_downsampled[i]  # shape [L, 3, 29]
    label = activities_center[i]  # shape [82] multi-label

    # Build node features, edges, and edge features
    node_feats, edge_index, edge_feats = build_graph_data(pose_seq, body_edges)

    # Convert label to a torch tensor
    label_tensor = torch.FloatTensor(label)

    # Create a PyG Data object
    graph_data = Data(
        x=node_feats,  # [n_nodes, in_features]
        edge_index=edge_index,
        edge_attr=edge_feats,  # [#edges, edge_features]
        y=label_tensor  # multi-label
    )
    graph_list.append(graph_data)





from torch_geometric.nn import MessagePassing



class GINELayer(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim):
        super(GINELayer, self).__init__(aggr='add')  # "Add" aggregation.
        self.mlp = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.edge_encoder = nn.Linear(edge_dim, out_channels)
        self.node_encoder = nn.Linear(in_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights for edge and node encoders, and MLP
        nn.init.xavier_uniform_(self.edge_encoder.weight)
        nn.init.xavier_uniform_(self.node_encoder.weight)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x, edge_index, edge_attr):
        # Encode node and edge features
        x = x.float()  # Ensure x is FloatTensor
        edge_attr = edge_attr.float()  # Ensure edge_attr is FloatTensor
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        # Message Passing
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # Update node embeddings
        out = self.mlp(out)
        return out

    def message(self, x_j, edge_attr):
        # Compute messages
        return x_j + edge_attr

    def update(self, aggr_out):
        return aggr_out




class BigGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim, num_layers=10, dropout=0.5):
        super().__init__()
        self.dropout = dropout

        # First layer
        self.conv1 = GINELayer(in_channels, hidden_channels, edge_dim)
        self.bn1 = BatchNorm(hidden_channels)

        # Hidden layers
        self.convs = torch.nn.ModuleList([
            GINELayer(hidden_channels, hidden_channels, edge_dim) for _ in range(num_layers - 2)
        ])
        self.bns = torch.nn.ModuleList([
            BatchNorm(hidden_channels) for _ in range(num_layers - 2)
        ])

        # Last layer
        self.conv_out = GINELayer(hidden_channels, hidden_channels, edge_dim)
        self.bn_out = BatchNorm(hidden_channels)

        # Classification head
        self.fc = Linear(hidden_channels, out_channels)
        self.attention = GATConv(hidden_channels, hidden_channels, heads=4, concat=False)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Input layer
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = torch.relu(x)
        x = Dropout(self.dropout)(x)

        # Hidden layers
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = torch.relu(x)
            x = Dropout(self.dropout)(x)

        # Last layer with attention
        x = self.conv_out(x, edge_index, edge_attr)
        x = self.bn_out(x)
        x = torch.relu(x)
        x = self.attention(x, edge_index)

        # Global pooling to aggregate node-level features to graph-level features
        x = global_mean_pool(x, batch)

        # Classification layer
        x = self.fc(x)
        return x


# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move model to GPU
# model = SimpleGCN(in_channels=3, hidden_channels=64, out_channels=82).to(device)
model = BigGCN(in_channels=3, hidden_channels=256, out_channels=82, num_layers=10, dropout=0.4).to(device)

# Define optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.BCEWithLogitsLoss()

# Split data into training and validation
train_loader = DataLoader(graph_list[:int(len(graph_list) * 0.8)], batch_size=16, shuffle=True)
val_loader = DataLoader(graph_list[int(len(graph_list) * 0.8):], batch_size=16, shuffle=False)


# Training and evaluation loops
def train():
    model.train()
    total_loss = 0
    for batch_data in train_loader:
        batch_data = batch_data.to(device)  # Move data to GPU
        optimizer.zero_grad()
        pred = model(batch_data)
        if batch_data.y.ndim == 1:
            batch_data.y = batch_data.y.view(pred.shape)  # Ensure shape consistency
        loss = criterion(pred, batch_data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate(loader):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_data in loader:
            batch_data = batch_data.to(device)  # Move data to GPU
            pred = model(batch_data)
            if batch_data.y.ndim == 1:
                batch_data.y = batch_data.y.view(pred.shape)  # Ensure shape consistency
            loss = criterion(pred, batch_data.y)
            total_loss += loss.item()

            # Collect predictions and targets for metrics
            preds = torch.sigmoid(pred).cpu().numpy() > 0.5  # Threshold at 0.5
            targets = batch_data.y.cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets)

    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="micro")
    return total_loss / len(loader), accuracy, f1


# Main training loop
num_epochs = 100
for epoch in range(1, num_epochs + 1):
    train_loss = train()
    val_loss, val_accuracy, val_f1 = evaluate(val_loader)

    print(f"Epoch {epoch:02d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
          f"Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")
