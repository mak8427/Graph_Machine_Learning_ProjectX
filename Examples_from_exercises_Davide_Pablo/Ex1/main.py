import torch
import torch_geometric as pyg
import numpy as np
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from tqdm import tqdm

# Find device
if torch.cuda.is_available():  # NVIDIA
    device = torch.device('cuda')
elif torch.backends.mps.is_available():  # Apple Silicon
    device = torch.device('mps')
else:
    device = torch.device('cpu')  # Fallback


class GCNLayer(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, activation=torch.nn.functional.relu):
        super(GCNLayer, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=False)
        self.activation = activation

    def forward(self, H: torch.Tensor, adj: torch.Tensor):
        # Add self-loops to the adjacency matrix
        adj_with_self_loops = adj + torch.eye(adj.size(0), device=adj.device)
        # Compute degree matrix
        degree = adj_with_self_loops.sum(dim=1)
        # Compute D^{-1/2}
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0.0
        # Normalize adjacency matrix
        adj_norm = adj_with_self_loops * degree_inv_sqrt.view(-1, 1) * degree_inv_sqrt.view(1, -1)
        # Linear transformation
        H_lin = self.linear(H)
        # Message passing
        H_out = adj_norm @ H_lin
        # Apply activation
        if self.activation is not None:
            H_out = self.activation(H_out)
        return H_out


class MeanPooling(torch.nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, H: torch.Tensor):
        return H.mean(dim=0)


class SumPooling(torch.nn.Module):
    def __init__(self):
        super(SumPooling, self).__init__()

    def forward(self, H: torch.Tensor):
        return H.sum(dim=0)


class GraphGCN(torch.nn.Module):
    def __init__(self, in_features, hidden_features, num_layers=2, pooling='mean'):
        super(GraphGCN, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNLayer(in_features, hidden_features))
        for _ in range(num_layers - 1):
            self.layers.append(GCNLayer(hidden_features, hidden_features))
        if pooling == 'mean':
            self.pooling = MeanPooling()
        elif pooling == 'sum':
            self.pooling = SumPooling()
        else:
            raise ValueError("Unknown pooling method")
        self.classifier = torch.nn.Linear(hidden_features, 1)  # Output is a single value

    def forward(self, H_in: torch.Tensor, adj: torch.Tensor):
        H = H_in
        for layer in self.layers:
            H = layer(H, adj)
        H_pooled = self.pooling(H)
        out = self.classifier(H_pooled)
        return out  # No activation here, we'll use BCEWithLogitsLoss


class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, adjacencies, features, targets):
        self.adjacencies = adjacencies
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.adjacencies[idx], self.features[idx], self.targets[idx]

    def num_features(self):
        return self.features[0].shape[-1]

    def compute_class_weights(self):
        targets = torch.cat([t for t in self.targets], dim=0)
        class_sample_count = torch.bincount(targets.flatten())
        class_weights = 1.0 / class_sample_count.float()
        class_weights = class_weights / class_weights.sum()
        return class_weights


def extract_graphs_and_features(dataset):
    # First pass to collect all unique atom types
    atoms_to_index = {}
    atom_type_counter = 0
    print("Collecting unique atom types...")
    for data in tqdm(dataset):
        x = data.x  # Node features (num_nodes, num_node_features)
        atom_types = x[:, 0].tolist()
        for atom_type in atom_types:
            atom_type = int(atom_type)
            if atom_type not in atoms_to_index:
                atoms_to_index[atom_type] = atom_type_counter
                atom_type_counter += 1
    num_atom_types = len(atoms_to_index)

    # Now process the dataset
    all_adjacencies = []
    all_features = []
    all_targets = []

    print("Processing dataset...")
    for data in tqdm(dataset):
        x = data.x  # Node features (num_nodes, num_node_features)
        edge_index = data.edge_index  # Edge indices (2, num_edges)
        y = data.y  # Target(s)
        num_nodes = x.size(0)
        # Create adjacency matrix
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        adj[edge_index[0], edge_index[1]] = 1.0

        # Map atom types to indices
        atom_types = x[:, 0].tolist()
        atom_indices = [atoms_to_index[int(atom_type)] for atom_type in atom_types]
        atom_indices = torch.tensor(atom_indices, dtype=torch.long)
        # One-hot encode
        features = torch.nn.functional.one_hot(atom_indices, num_classes=num_atom_types).float()

        all_adjacencies.append(adj)
        all_features.append(features)
        all_targets.append(y)

    all_targets = torch.cat(all_targets, dim=0)
    return all_adjacencies, all_features, all_targets, atoms_to_index


batch_size = 1  # Set batch_size to 1 due to variable graph sizes

molHIV = PygGraphPropPredDataset(name="ogbg-molhiv")
split_idx = molHIV.get_idx_split()
all_adjacencies, all_features, all_targets, atoms_to_index = extract_graphs_and_features(molHIV)
all_targets = all_targets.to(torch.int64)
num_atom_types = len(atoms_to_index)

# Create datasets using split_idx indices
graph_dataset = GraphDataset(all_adjacencies, all_features, all_targets)
train_dataset = torch.utils.data.Subset(graph_dataset, split_idx["train"])
val_dataset = torch.utils.data.Subset(graph_dataset, split_idx["valid"])
test_dataset = torch.utils.data.Subset(graph_dataset, split_idx["test"])

# Create DataLoaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

evaluator = Evaluator(name='ogbg-molhiv')


def evaluate(model, loader):
    model.eval()

    y_true = []
    y_pred = []

    for adjacencies, features, targets in loader:
        adjacencies = adjacencies[0].to(device)  # Remove batch dimension
        features = features[0].to(device)
        targets = targets.to(device).float()

        with torch.no_grad():
            output = model(features, adjacencies)
            pred = torch.sigmoid(output)

        # Ensure targets and preds are 1D tensors
        targets = targets.view(-1)
        pred = pred.view(-1)

        y_pred.append(pred.cpu())
        y_true.append(targets.cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    # Ensure y_true and y_pred have shape (num_samples, num_tasks)
    y_true = y_true.unsqueeze(1)
    y_pred = y_pred.unsqueeze(1)

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)['rocauc']


# Compute class weights
all_targets_flat = all_targets.flatten()
num_pos = (all_targets_flat == 1).sum().float()
num_neg = (all_targets_flat == 0).sum().float()
pos_weight = num_neg / num_pos

# Model definition
model = GraphGCN(in_features=num_atom_types, hidden_features=64, num_layers=2, pooling='mean')
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

# Training loop
epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for adjacencies, features, targets in tqdm(train_loader):
        adjacencies = adjacencies[0].to(device)  # Remove batch dimension
        features = features[0].to(device)
        targets = targets.to(device).float()
        optimizer.zero_grad()
        output = model(features, adjacencies)  # Output shape: (1,)
        loss = loss_fn(output.view(-1), targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    # Evaluate on validation set
    val_rocauc = evaluate(model, val_loader)
    print(f"Validation ROC-AUC: {val_rocauc:.4f}")

# Evaluate on test set
test_rocauc = evaluate(model, test_loader)
print(f"Test ROC-AUC: {test_rocauc:.4f}")
