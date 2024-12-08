# graph_transformer_ogbg_molpcba_fixed.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.transforms import Compose
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_scipy_sparse_matrix
from ogb.graphproppred.mol_encoder import AtomEncoder
import torch_geometric.transforms as T
import scipy
import numpy as np
import logging
import sys
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

# ===============================
# 1. Configure Logging
# ===============================

# Create a logger
logger = logging.getLogger('GraphTransformerLogger')
logger.setLevel(logging.INFO)  # Set to INFO during normal training

# Create handlers
file_handler = logging.FileHandler('debug.log')
file_handler.setLevel(logging.DEBUG)  # Capture all logs in the file

# Optional: Create console handler if you still want some output on the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Display INFO and above on console

# Create formatter and add it to handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# ===============================
# 2. Device Configuration
# ===============================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Using device: {device}')

# Enable CUDNN benchmarking for optimized performance
torch.backends.cudnn.benchmark = True
logger.info("Enabled CUDNN benchmarking for optimized performance.")

# ===============================
# 3. Define Data Transformation Classes
# ===============================

# Define BondAttributeMapper with vectorized mapping
class BondAttributeMapper(object):
    def __init__(self):
        # Define maximum allowed values for each bond feature
        self.max_bond_type = 4
        self.max_bond_stereo = 2
        self.max_bond_conj = 2

    def __call__(self, data):
        if data.edge_attr is not None:
            # Ensure bond attributes are of type long
            data.edge_attr = data.edge_attr.long()

            # Clamp bond_type to [0, max_bond_type]
            data.edge_attr[:, 0] = torch.clamp(data.edge_attr[:, 0], max=self.max_bond_type)

            # Clamp bond_stereo to [0, max_bond_stereo]
            data.edge_attr[:, 1] = torch.clamp(data.edge_attr[:, 1], max=self.max_bond_stereo)

            # Clamp bond_conj to [0, max_bond_conj]
            data.edge_attr[:, 2] = torch.clamp(data.edge_attr[:, 2], max=self.max_bond_conj)
        else:
            # Assign default bond attributes if missing
            num_edges = data.edge_index.size(1)
            data.edge_attr = torch.zeros((num_edges, 3), dtype=torch.long).to(device)
            logger.debug(f"Assigned default bond attributes for graph with {num_edges} edges.")
        return data

# Define Laplacian Positional Encoding
class LaplacianPositionalEncoding(object):
    def __init__(self, pe_dim):
        self.pe_dim = pe_dim

    def __call__(self, data):
        num_nodes = data.num_nodes
        edge_index = data.edge_index

        # Convert edge_index to adjacency matrix
        adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tocsc()

        # Compute the Laplacian
        laplacian = scipy.sparse.csgraph.laplacian(adj, normed=True)

        # Compute the eigenvalues and eigenvectors
        try:
            eigenvalues, eigenvectors = scipy.linalg.eigh(laplacian.toarray())
            logger.debug(f"Computed Laplacian eigenvectors for graph with {num_nodes} nodes.")
        except Exception as e:
            logger.error(f"Error computing eigenvectors for graph with {num_nodes} nodes: {e}")
            eigenvalues, eigenvectors = np.linalg.eigh(laplacian.toarray())

        # Handle small graphs where num_nodes -1 < pe_dim
        actual_pe_dim = min(self.pe_dim, num_nodes - 1)
        if actual_pe_dim <= 0:
            pe = torch.zeros((num_nodes, self.pe_dim)).float().to(device)
            logger.debug(f"Graph with {num_nodes} nodes: Using zero-padding for LapPE.")
        else:
            pe = torch.from_numpy(eigenvectors[:, 1:1 + actual_pe_dim]).float().to(device)
            if actual_pe_dim < self.pe_dim:
                padding = torch.zeros((num_nodes, self.pe_dim - actual_pe_dim)).float().to(device)
                pe = torch.cat([pe, padding], dim=1)
                logger.debug(f"Graph with {num_nodes} nodes: Padding LapPE from {actual_pe_dim} to {self.pe_dim} dimensions.")
        data.lap_pe = pe  # Shape: [num_nodes, pe_dim]
        return data

# Define Random Walk Structural Encoding
class RandomWalkStructuralEncoding(object):
    def __init__(self, walk_length):
        self.walk_length = walk_length

    def __call__(self, data):
        num_nodes = data.num_nodes
        edge_index = data.edge_index

        adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tocsc()

        diag = []
        A = adj.copy()

        for k in range(1, self.walk_length + 1):
            A_power = A ** k
            diag_k = A_power.diagonal()
            diag.append(diag_k)

        diag = np.stack(diag, axis=1)  # Shape: [num_nodes, walk_length]
        rwse = torch.from_numpy(diag).float().to(device)
        desired_walk_length = self.walk_length
        if rwse.shape[1] < desired_walk_length:
            padding = torch.zeros((num_nodes, desired_walk_length - rwse.shape[1])).float().to(device)
            rwse = torch.cat([rwse, padding], dim=1)
            logger.debug(f"Graph with {num_nodes} nodes: Padding RWSE from {rwse.shape[1]} to {desired_walk_length} dimensions.")
        data.rwse = rwse  # Shape: [num_nodes, walk_length]
        return data

# ===============================
# 4. Define Model Components
# ===============================

# Define CustomBondEncoder
class CustomBondEncoder(nn.Module):
    def __init__(self, emb_dim=64, num_bond_types=5, num_bond_stereo=3, num_bond_conj=3):
        super().__init__()
        self.bond_embedding_list = nn.ModuleList([
            nn.Embedding(num_bond_types, emb_dim),    # bond type
            nn.Embedding(num_bond_stereo, emb_dim),   # bond stereo
            nn.Embedding(num_bond_conj, emb_dim)      # bond conjugation
        ])

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])
        return bond_embedding  # Shape: [num_edges, emb_dim]

# Define SignNet to ensure sign invariance
class SignNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(SignNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, pe):
        pe_pos = self.mlp(pe)
        pe_neg = self.mlp(-pe)
        pe_sign_invariant = pe_pos + pe_neg  # Ensures f(ν) = f(-ν)
        return pe_sign_invariant  # Shape: [num_nodes, out_dim]

# Define GraphTransformer using TransformerConv
from torch_geometric.nn import TransformerConv, global_mean_pool

class GraphTransformer(nn.Module):
    def __init__(self, hidden_dim, out_dim, num_layers, num_heads, dropout=0.1,
                 num_bond_types=5, num_bond_stereo=3, num_bond_conj=3):
        super(GraphTransformer, self).__init__()
        self.atom_encoder = AtomEncoder(emb_dim=hidden_dim)
        self.bond_encoder = CustomBondEncoder(emb_dim=hidden_dim,
                                              num_bond_types=num_bond_types,
                                              num_bond_stereo=num_bond_stereo,
                                              num_bond_conj=num_bond_conj)
        self.sign_net = SignNet(in_dim=10, hidden_dim=hidden_dim, out_dim=hidden_dim)
        self.rwse_mlp = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.transformer_convs = nn.ModuleList()
        for layer in range(num_layers):
            conv = TransformerConv(hidden_dim, hidden_dim // num_heads,
                                   heads=num_heads, dropout=dropout, edge_dim=hidden_dim)
            self.transformer_convs.append(conv)
            logger.debug(f"Initialized TransformerConv layer {layer + 1}/{num_layers} with out_channels={hidden_dim // num_heads}.")
        self.fc_out = nn.Linear(hidden_dim, out_dim)
        logger.info("GraphTransformer initialization complete.")

    def forward(self, data):
        x = self.atom_encoder(data.x.to(device))  # Shape: [num_nodes, hidden_dim]

        # Incorporate positional encodings
        pe = data.lap_pe.to(device)  # Shape: [num_nodes, pe_dim]
        pe = self.sign_net(pe)  # Shape: [num_nodes, hidden_dim]
        x = x + pe  # Incorporate positional encodings

        # Incorporate structural encodings
        rwse = data.rwse.to(device)  # Shape: [num_nodes, walk_length]
        rwse = self.rwse_mlp(rwse)  # Shape: [num_nodes, hidden_dim]
        x = x + rwse  # Incorporate structural encodings

        edge_index = data.edge_index.to(device)
        edge_attr = data.edge_attr.to(device)

        # Debug: Check edge_attr dimensions and values
        logger.debug(f"GraphTransformer - Edge_attr shape: {edge_attr.shape}")
        logger.debug(f"GraphTransformer - Edge_attr max per feature: {edge_attr.max(dim=0).values}")

        edge_attr = self.bond_encoder(edge_attr)  # Shape: [num_edges, emb_dim]
        logger.debug(f"GraphTransformer - Bond Embedding shape: {edge_attr.shape}")

        for conv in self.transformer_convs:
            x = conv(x, edge_index, edge_attr)  # Shape: [num_nodes, hidden_dim]
            x = F.relu(x)
            logger.debug(f"GraphTransformer - After TransformerConv: x shape: {x.shape}")

        x = global_mean_pool(x, data.batch)  # Shape: [batch_size, hidden_dim]
        out = self.fc_out(x)  # Shape: [batch_size, out_dim]
        return out

# Define PureTransformer Model
class PureTransformer(nn.Module):
    def __init__(self, hidden_dim, out_dim, num_layers, num_heads, dropout=0.1,
                 num_bond_types=5, num_bond_stereo=3, num_bond_conj=3):
        super(PureTransformer, self).__init__()
        self.atom_encoder = AtomEncoder(emb_dim=hidden_dim)
        self.bond_encoder = CustomBondEncoder(emb_dim=hidden_dim,
                                              num_bond_types=num_bond_types,
                                              num_bond_stereo=num_bond_stereo,
                                              num_bond_conj=num_bond_conj)
        self.sign_net = SignNet(in_dim=10, hidden_dim=hidden_dim, out_dim=hidden_dim)
        self.rwse_mlp = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, out_dim)
        logger.info("PureTransformer initialization complete.")

    def forward(self, data):
        x = self.atom_encoder(data.x.to(device))  # Shape: [num_nodes, hidden_dim]

        # Incorporate positional encodings
        pe = data.lap_pe.to(device)  # Shape: [num_nodes, pe_dim]
        pe = self.sign_net(pe)  # Shape: [num_nodes, hidden_dim]
        x = x + pe  # Incorporate positional encodings

        # Incorporate structural encodings
        rwse = data.rwse.to(device)  # Shape: [num_nodes, walk_length]
        rwse = self.rwse_mlp(rwse)  # Shape: [num_nodes, hidden_dim]
        x = x + rwse  # Incorporate structural encodings

        # Transformers expect input of shape (sequence_length, batch_size, embedding_dim)
        # In PyG batching, nodes from different graphs are concatenated, so batch_size=1
        x = x.unsqueeze(1)  # Shape: [num_nodes, 1, hidden_dim]

        # Apply Transformer Encoder
        x = self.transformer_encoder(x)  # Shape: [num_nodes, 1, hidden_dim]
        x = x.squeeze(1)  # Shape: [num_nodes, hidden_dim]

        # Global pooling
        x = global_mean_pool(x, data.batch)  # Shape: [batch_size, hidden_dim]
        out = self.fc_out(x)  # Shape: [batch_size, out_dim]
        return out

# Define GCN model
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, hidden_dim, out_dim, num_layers,
                 num_bond_types=5, num_bond_stereo=3, num_bond_conj=3):
        super(GCN, self).__init__()
        self.atom_encoder = AtomEncoder(emb_dim=hidden_dim)
        self.bond_encoder = CustomBondEncoder(emb_dim=hidden_dim,
                                              num_bond_types=num_bond_types,
                                              num_bond_stereo=num_bond_stereo,
                                              num_bond_conj=num_bond_conj)
        self.sign_net = SignNet(in_dim=10, hidden_dim=hidden_dim, out_dim=hidden_dim)
        self.rwse_mlp = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        logger.debug(f"Initialized GCNConv layer 1/{num_layers}.")
        for layer in range(1, num_layers):
            conv = GCNConv(hidden_dim, hidden_dim)
            self.convs.append(conv)
            logger.debug(f"Initialized GCNConv layer {layer + 1}/{num_layers}.")
        self.fc_out = nn.Linear(hidden_dim, out_dim)
        logger.info("GCN initialization complete.")

    def forward(self, data):
        x = self.atom_encoder(data.x.to(device))  # Shape: [num_nodes, hidden_dim]

        # Incorporate positional encodings
        pe = data.lap_pe.to(device)  # Shape: [num_nodes, pe_dim]
        pe = self.sign_net(pe)  # Shape: [num_nodes, hidden_dim]
        x = x + pe  # Incorporate positional encodings

        # Incorporate structural encodings
        rwse = data.rwse.to(device)  # Shape: [num_nodes, walk_length]
        rwse = self.rwse_mlp(rwse)  # Shape: [num_nodes, hidden_dim]
        x = x + rwse  # Incorporate structural encodings

        edge_index = data.edge_index.to(device)
        edge_attr = data.edge_attr.to(device)

        # Debug: Check edge_attr dimensions and values
        logger.debug(f"GCN - Edge_attr shape: {edge_attr.shape}")
        logger.debug(f"GCN - Edge_attr max per feature: {edge_attr.max(dim=0).values}")

        edge_attr = self.bond_encoder(edge_attr)  # Shape: [num_edges, emb_dim]
        logger.debug(f"GCN - Bond Embedding shape: {edge_attr.shape}")

        for conv in self.convs:
            x = conv(x, edge_index)  # Shape: [num_nodes, hidden_dim]
            x = F.relu(x)
            logger.debug(f"GCN - After GCNConv: x shape: {x.shape}")

        x = global_mean_pool(x, data.batch)  # Shape: [batch_size, hidden_dim]
        out = self.fc_out(x)  # Shape: [batch_size, out_dim]
        return out

# ===============================
# 5. Prepare Dataset with Transforms
# ===============================

transform = Compose([
    T.ToUndirected(),
    BondAttributeMapper(),  # Use the corrected mapper
    LaplacianPositionalEncoding(pe_dim=10),
    RandomWalkStructuralEncoding(walk_length=10)
])

# Load the dataset
logger.info("Loading dataset...")
dataset = PygGraphPropPredDataset(name='ogbg-molpcba', root='data/ogbg_molpcba', transform=transform)
logger.info(f"Number of graphs in the dataset: {len(dataset)}")
logger.info(f"Number of tasks: {dataset.num_tasks}")
logger.info(f"Example graph:")
logger.info(dataset[0])

# Data splitting
split_idx = dataset.get_idx_split()
train_dataset = dataset[split_idx['train']]
valid_dataset = dataset[split_idx['valid']]
test_dataset = dataset[split_idx['test']]

logger.info(f"Number of training graphs: {len(train_dataset)}")
logger.info(f"Number of validation graphs: {len(valid_dataset)}")
logger.info(f"Number of test graphs: {len(test_dataset)}")

# ===============================
# 6. Create DataLoaders
# ===============================

# Create DataLoaders with multiple workers and pin_memory for faster data transfer
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64, num_workers=4, pin_memory=True)

# ===============================
# 7. Inspect Bond Attributes in a Batch
# ===============================

def inspect_bond_attributes(loader):
    for batch_idx, data in tqdm(enumerate(loader)):
        print("x")
        if data.edge_attr is not None:
            max_values = data.edge_attr.max(dim=0).values
            logger.debug(f"Batch {batch_idx} - Max Bond Attributes: {max_values}")
        else:
            logger.debug(f"Batch {batch_idx} - No Bond Attributes Found.")
        # Inspect only the first few batches
        if batch_idx >= 5:
            break

logger.info("\nInspecting Bond Attributes in Training Data:")
inspect_bond_attributes(train_loader)

# ===============================
# 8. Initialize Evaluator and Loss Function
# ===============================

evaluator = Evaluator(name='ogbg-molpcba')
criterion = nn.BCEWithLogitsLoss()

# ===============================
# 9. Define Training and Evaluation Functions
# ===============================

def train_model(model, loader, optimizer, scaler):
    model.train()
    total_loss = 0
    progress_bar = tqdm(enumerate(loader), total=len(loader), desc="Training", leave=False)
    for batch_idx, data in progress_bar:
        data = data.to(device)
        optimizer.zero_grad()
        with autocast():
            out = model(data)  # Shape: [batch_size, num_tasks]

            # Mask out invalid targets (-1)
            is_labeled = data.y != -1  # Shape: [batch_size, num_tasks]
            if torch.sum(is_labeled).item() == 0:
                logger.debug(f"Batch {batch_idx}: No labeled data, skipping.")
                continue  # Skip if no labels are present

            # Clamp labels to [0,1]
            y_true = data.y.clone()
            y_true[~is_labeled] = 0  # Set unlabeled to 0 (won't affect loss)
            y_true = y_true.float()

            # Debug: Print label statistics
            if torch.any((y_true < 0) | (y_true > 1)):
                logger.warning(f"Batch {batch_idx}: Labels out of range [0, 1].")
                logger.warning(f"Labels: {y_true}")

            # Compute loss only on labeled data
            loss = criterion(out[is_labeled], y_true[is_labeled])

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * data.num_graphs

        # Update progress bar description every 100 batches
        if batch_idx % 100 == 0:
            progress_bar.set_postfix({'Batch Loss': loss.item()})

    average_loss = total_loss / len(loader.dataset)
    logger.debug(f"Average Training Loss: {average_loss:.4f}")
    return average_loss

def evaluate_model(model, loader):
    model.eval()
    y_true = []
    y_pred = []
    progress_bar = tqdm(loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for data in progress_bar:
            data = data.to(device)
            out = model(data)  # Shape: [batch_size, num_tasks]
            y_true.append(data.y.cpu())
            y_pred.append(out.cpu())
    y_true = torch.cat(y_true, dim=0)  # Shape: [num_graphs, num_tasks]
    y_pred = torch.cat(y_pred, dim=0)  # Shape: [num_graphs, num_tasks]
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    result = evaluator.eval(input_dict)
    logger.debug(f"Evaluation Result: {result}")
    return result["ap"]  # Average Precision

# ===============================
# 10. Initialize Models and Optimizers
# ===============================

hidden_dim = 64
out_dim = dataset.num_tasks  # 128 for ogbg-molpcba
num_layers = 3
num_heads = 4

# Instantiate GraphTransformer and PureTransformer
logger.info("\nInitializing GraphTransformer:")
model_transformer_conv = GraphTransformer(hidden_dim=hidden_dim, out_dim=out_dim,
                                          num_layers=num_layers, num_heads=num_heads).to(device)
optimizer_transformer_conv = torch.optim.Adam(model_transformer_conv.parameters(), lr=0.001)

logger.info("\nInitializing PureTransformer:")
model_pure_transformer = PureTransformer(hidden_dim=hidden_dim, out_dim=out_dim,
                                         num_layers=num_layers, num_heads=num_heads).to(device)
optimizer_pure_transformer = torch.optim.Adam(model_pure_transformer.parameters(), lr=0.001)

# Instantiate GCN model
logger.info("\nInitializing GCN:")
model_gcn = GCN(hidden_dim=hidden_dim, out_dim=out_dim, num_layers=num_layers).to(device)
optimizer_gcn = torch.optim.Adam(model_gcn.parameters(), lr=0.001)

# Initialize GradScaler for mixed precision
scaler = GradScaler()
logger.info("Initialized GradScaler for mixed precision training.")

# ===============================
# 11. Training Loop
# ===============================

num_epochs = 50

# Training GraphTransformer with TransformerConv layers
logger.info("\nStarting Training for GraphTransformer:")
for epoch in range(1, num_epochs + 1):
    loss = train_model(model_transformer_conv, train_loader, optimizer_transformer_conv, scaler)
    val_ap = evaluate_model(model_transformer_conv, valid_loader)
    test_ap = evaluate_model(model_transformer_conv, test_loader)
    logger.info(f'[TransformerConv] Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AP: {val_ap:.4f}, Test AP: {test_ap:.4f}')

# Training Pure Transformer Model
logger.info("\nStarting Training for PureTransformer:")
for epoch in range(1, num_epochs + 1):
    loss = train_model(model_pure_transformer, train_loader, optimizer_pure_transformer, scaler)
    val_ap = evaluate_model(model_pure_transformer, valid_loader)
    test_ap = evaluate_model(model_pure_transformer, test_loader)
    logger.info(f'[PureTransformer] Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AP: {val_ap:.4f}, Test AP: {test_ap:.4f}')

# Training GCN Model
logger.info("\nStarting Training for GCN:")
for epoch in range(1, num_epochs + 1):
    loss = train_model(model_gcn, train_loader, optimizer_gcn, scaler)
    val_ap = evaluate_model(model_gcn, valid_loader)
    test_ap = evaluate_model(model_gcn, test_loader)
    logger.info(f'[GCN] Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AP: {val_ap:.4f}, Test AP: {test_ap:.4f}')

#%%
