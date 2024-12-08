import torch
import torch_geometric as pyg
from torch_geometric.nn import MessagePassing, global_mean_pool
import torch_scatter
import torch.optim as optim
from sklearn.metrics import average_precision_score
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import networkx as nx

# Tasks:
# 1. Implement virtual nodes
# 2. Implement GINE (GIN + Edge features) based on the sparse implementation
# 3. Test everything on peptides-func
# 4. Draw the molecule peptides_train[0]

# TASK 1 & 2: GINE Layer with Virtual Nodes
class GINELayerWithVN(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim):
        super(GINELayerWithVN, self).__init__(aggr='add')  # "Add" aggregation.
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(out_channels, out_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels, out_channels)
        )
        self.edge_encoder = torch.nn.Linear(edge_dim, out_channels)
        self.node_encoder = torch.nn.Linear(in_channels, out_channels)
        self.virtual_node_mlp = torch.nn.Sequential(
            torch.nn.Linear(out_channels, out_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels, out_channels)
        )
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.edge_encoder.weight)
        torch.nn.init.xavier_uniform_(self.node_encoder.weight)
        for m in self.mlp:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
        for m in self.virtual_node_mlp:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x, edge_index, edge_attr, vn_embed, batch):
        # Encode node and edge features
        x = x.float()  # Ensure x is FloatTensor
        edge_attr = edge_attr.float()  # Ensure edge_attr is FloatTensor
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        # Add virtual node embedding to node features
        vn_expanded = vn_embed[batch]
        x = x + vn_expanded

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

# Updated GNN Model with Virtual Node and GINE Layer
class GNNWithVirtualNodeAndGINE(torch.nn.Module):
    def __init__(self, in_features, hidden_features, out_features, edge_attr_dim, num_layers=5):
        super(GNNWithVirtualNodeAndGINE, self).__init__()
        self.num_layers = num_layers
        self.hidden_features = hidden_features

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GINELayerWithVN(in_features if _ == 0 else hidden_features, hidden_features, edge_attr_dim))

        self.virtual_node_embedding = torch.nn.Embedding(1, hidden_features)
        torch.nn.init.constant_(self.virtual_node_embedding.weight.data, 0)

        self.mlp_virtual_node = torch.nn.Sequential(
            torch.nn.Linear(hidden_features, hidden_features),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_features, hidden_features),
            torch.nn.ReLU(),
        )

        self.fc = torch.nn.Linear(hidden_features, out_features)

    def forward(self, x, edge_index, edge_attr, batch):
        # Initialize virtual node embedding
        batch_size = batch.max().item() + 1
        vn_embed = self.virtual_node_embedding.weight.repeat(batch_size, 1)

        for conv in self.convs:
            x = conv(x, edge_index, edge_attr, vn_embed, batch)
            x = F.relu(x)

            # Update virtual node embedding
            vn_aggr = global_mean_pool(x, batch)
            vn_embed = vn_embed + self.mlp_virtual_node(vn_aggr)

        # Graph-level readout
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x

# Training and evaluation functions
def train(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        batch.x = batch.x.float()  # Convert node features to float
        batch.edge_attr = batch.edge_attr.float()  # Convert edge attributes to float
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = loss_fn(out, batch.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    average_loss = total_loss / len(loader)
    return average_loss

def evaluate(model, loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            batch.x = batch.x.float()  # Convert node features to float
            batch.edge_attr = batch.edge_attr.float()  # Convert edge attributes to float
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            y_pred.append(out.cpu())
            y_true.append(batch.y.cpu())
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    # Compute per-class AP
    ap_per_class = []
    for i in range(y_true.shape[1]):
        try:
            ap = average_precision_score(y_true[:, i], y_pred[:, i])
        except ValueError:
            ap = 0.0  # Handle cases where a class has no positive samples
        ap_per_class.append(ap)
    mean_ap = np.mean(ap_per_class)
    return mean_ap

def plot_results(epochs, train_losses, val_aps, learning_rates=None):
    epochs_range = range(1, epochs + 1)

    # Plot Training Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('Training_Loss.png')
    plt.show()

    # Plot Validation AP Score
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, val_aps, label='Validation AP Score', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Average Precision Score')
    plt.title('Validation AP Score over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('Validation_AP_Score.png')
    plt.show()

    # Plot Learning Rate if provided
    if learning_rates is not None:
        plt.figure(figsize=(10, 5))
        plt.plot(epochs_range, learning_rates, label='Learning Rate', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig('Learning_Rate.png')
        plt.show()

def main(epochs=100, lr=0.001, hidden_features=256):
    # Compute edge_attr_dim and num_tasks from the dataset
    edge_attr_dim = dataset[0].edge_attr.shape[1]
    num_tasks = dataset[0].y.shape[-1]

    # Initialize the model, optimizer, and loss function
    model = GNNWithVirtualNodeAndGINE(
        in_features=dataset.num_node_features,
        hidden_features=hidden_features,
        out_features=num_tasks,
        edge_attr_dim=edge_attr_dim,
        num_layers=5  # Increased depth as per the paper's suggestion
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.9, patience=10)

    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Lists to store losses and AP scores
    train_losses = []
    val_aps = []
    learning_rates = []

    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, loss_fn)
        val_ap = evaluate(model, val_loader)
        train_losses.append(train_loss)
        val_aps.append(val_ap)
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Validation AP Score: {val_ap:.4f}, Learning Rate: {current_lr:.6f}")
        scheduler.step(val_ap)

    # Final test evaluation
    test_ap = evaluate(model, test_loader)
    print(f"Test AP Score: {test_ap:.4f}")

    # Plotting the results
    plot_results(epochs, train_losses, val_aps)

# Task 4: Draw the molecule represented by peptides_train[0]
def draw_molecule(data, def_col=0):
    G = pyg.utils.to_networkx(data, to_undirected=True)
    node_features = data.x.numpy()
    edge_index = data.edge_index.numpy()
    edge_attr = data.edge_attr.numpy()
    bond_types = edge_attr[:, 0].astype(int)
    atom_types = None
    atom_type_indices = None
    for i, (u, v) in enumerate(zip(edge_index[0], edge_index[1])):
        G.edges[u, v]['bond_type'] = bond_types[i]
    if def_col == 0:
        atom_types = {
            5: 'C',
            6: 'N',
            7: 'O',
        }
        atom_type_indices = node_features[:, def_col].astype(int)
    elif def_col == 2:
        atom_types = {4: 'C', 3: 'O', 1: 'N'}
        atom_type_indices = node_features[:, def_col].astype(int)
    elif def_col == 4:
        atom_types = {1: 'C', 0: 'O', 2: 'N'}
        atom_type_indices = node_features[:, def_col].astype(int)
    bond_color_mapping = {
        0: 'black',
        1: 'blue',
        3: 'red',
    }
    edges = list(G.edges())
    edge_colors = []
    for u, v in edges:
        bond_type = G.edges[u, v]['bond_type']
        color = bond_color_mapping.get(bond_type, 'green')
        edge_colors.append(color)
    labels = {i: atom_types.get(atom_type_indices[i], 'X') for i in range(atom_type_indices.shape[0])}
    size=12
    plt.figure(figsize=(size, size))
    pos = nx.kamada_kawai_layout(G, scale=5)
    nx.draw(
        G, pos,
        with_labels=False,
        node_size=50,
        node_color='lightblue',
        edgelist=edges,
        edge_color=edge_colors,
        width=1.5
    )
    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=6,
        font_weight='bold'
    )
    plt.title('Molecule Visualization of peptides_train[0]')
    plt.axis('off')
    plt.savefig('Molecule_Visualization.png')
    plt.show()

if __name__ == "__main__":
    # Device configuration
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    # Load dataset and create data loaders
    dataset = pyg.datasets.LRGBDataset(root='dataset/peptides-func', name="Peptides-func")
    peptides_train = pyg.datasets.LRGBDataset(root='dataset/peptides-func', name="Peptides-func", split="train")
    peptides_val = pyg.datasets.LRGBDataset(root='dataset/peptides-func', name="Peptides-func", split="val")
    peptides_test = pyg.datasets.LRGBDataset(root='dataset/peptides-func', name="Peptides-func", split="test")

    batch_size = 32
    train_loader = pyg.loader.DataLoader(peptides_train, batch_size=batch_size, shuffle=True)
    val_loader = pyg.loader.DataLoader(peptides_val, batch_size=batch_size, shuffle=False)
    test_loader = pyg.loader.DataLoader(peptides_test, batch_size=batch_size, shuffle=False)

    # Check number of classes and label distribution
    num_classes = dataset[0].y.shape[-1]
    print(f"Number of classes: {num_classes}")

    all_labels = np.concatenate([data.y.numpy() for data in dataset], axis=0)
    label_distribution = np.mean(all_labels, axis=0)
    print(f"Label distribution: {label_distribution}")

    # Run the main training loop
    main(epochs=300, lr=0.001, hidden_features=128)

    # Draw the molecule for Task 4
    draw_molecule(peptides_train[0])

    # Seed already implemented in the universe: 42

#%%
