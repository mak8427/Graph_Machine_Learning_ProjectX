import torch
import torch_geometric as pyg
import torch_scatter
import torch.optim as optim
from sklearn.metrics import average_precision_score
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Tasks:
# 1. Implement virtual nodes
# 2. Implement GINE (GIN + Edge features) based on the sparse implementation
# 3. Test everything on peptides-func
# 4. Draw the molecule peptides_train[0] (Not included in this code)

# Standard GCN Layer
class GCNLayer(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, activation=torch.nn.functional.relu):
        super(GCNLayer, self).__init__()
        self.activation = activation
        self.W = torch.nn.Parameter(torch.zeros(in_features, out_features))
        torch.nn.init.kaiming_normal_(self.W)

    def forward(self, H: torch.Tensor, edge_index: torch.Tensor):
        out = H
        new_H = torch_scatter.scatter_add(H[edge_index[0]], edge_index[1], dim=0, dim_size=H.size(0))
        out = out + new_H
        out = out.matmul(self.W)
        if self.activation:
            out = self.activation(out)
        return out

# TASK 1: GCN Layer with Virtual Nodes
class GCNLayerWithVirtualNode(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, activation=torch.nn.functional.relu):
        super(GCNLayerWithVirtualNode, self).__init__()
        self.activation = activation
        self.W = torch.nn.Parameter(torch.zeros(in_features, out_features))
        torch.nn.init.kaiming_normal_(self.W)

        # Initialize virtual node as a buffer (not a parameter)
        self.register_buffer('virtual_node', torch.zeros(in_features))

    def forward(self, H: torch.Tensor, edge_index: torch.Tensor, batch):
        # Standard message passing
        out = H
        new_H = torch_scatter.scatter_add(H[edge_index[0]], edge_index[1], dim=0, dim_size=H.size(0))
        out = out + new_H

        # Aggregate information into virtual node per graph
        virtual_node_msg = torch_scatter.scatter_mean(out, batch, dim=0)

        # Distribute virtual node information back to nodes
        virtual_node_expanded = virtual_node_msg[batch]
        out = out + virtual_node_expanded

        # Apply transformation and activation
        out = out.matmul(self.W)
        if self.activation:
            out = self.activation(out)
        return out

# TASK 2: GINE Layer with Edge Features
class GINELayer(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, edge_attr_dim: int):
        super(GINELayer, self).__init__()
        # MLP for node features after aggregation
        self.node_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features, out_features),
            torch.nn.ReLU(),
            torch.nn.Linear(out_features, out_features)
        )
        # MLP for edge features
        self.edge_mlp = torch.nn.Linear(edge_attr_dim, in_features)

    def forward(self, H: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        # Ensure edge_attr is of the same dtype as H
        edge_attr = edge_attr.to(H.dtype)

        # Compute edge messages
        edge_messages = self.edge_mlp(edge_attr)

        # Sum node features and edge features
        messages = H[edge_index[0]] + edge_messages
        aggregated = torch_scatter.scatter_add(messages, edge_index[1], dim=0, dim_size=H.size(0))

        # Apply MLP to aggregated messages
        out = self.node_mlp(aggregated)
        return out

# Updated GNN Model with Virtual Node and GINE Layer
import torch.nn.functional as F

class GNNWithVirtualNodeAndGINE(torch.nn.Module):
    def __init__(self, in_features, hidden_features, out_features, edge_attr_dim):
        super(GNNWithVirtualNodeAndGINE, self).__init__()
        self.conv1 = GCNLayerWithVirtualNode(in_features, hidden_features)
        self.conv1 = GCNLayerWithVirtualNode(in_features, hidden_features)
        self.conv2 = GINELayer(hidden_features, hidden_features, edge_attr_dim)
        self.conv3 = GINELayer(hidden_features, hidden_features, edge_attr_dim)
        self.fc = torch.nn.Linear(hidden_features, out_features)

        # Projection layers to align dimensions if necessary
        self.proj1 = torch.nn.Linear(in_features, hidden_features) if in_features != hidden_features else None
        self.proj2 = torch.nn.Linear(hidden_features, hidden_features)

    def forward(self, x, edge_index, edge_attr, batch):
        # First layer with virtual node and skip connection
        residual = x
        x = self.conv1(x, edge_index, batch)

        # Apply projection if dimensions don't match
        if residual.size(-1) != x.size(-1):
            residual = self.proj1(residual)

        x = x + residual  # Adding skip connection

        # Second layer with GINE and skip connection
        residual = x
        x = self.conv2(x, edge_index, edge_attr)

        # Apply projection if dimensions don't match
        if residual.size(-1) != x.size(-1):
            residual = self.proj2(residual)

        x = x + residual  # Adding skip connection

        # Global pooling and final output layer
        x = torch_scatter.scatter_mean(x, batch, dim=0)
        x = self.fc(x)
        return x

# Training and evaluation functions
def train(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        batch.x = batch.x.float()  # Convert node features to float
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


def main(epochs=100, lr=0.001, hidden_features=32):
    # Compute edge_attr_dim and num_tasks from the dataset
    edge_attr_dim = dataset[0].edge_attr.shape[1]
    num_tasks = dataset[0].y.shape[-1]

    # Initialize the model, optimizer, and loss function
    model = GNNWithVirtualNodeAndGINE(
        in_features=dataset.num_node_features,
        hidden_features=hidden_features,
        out_features=num_tasks,
        edge_attr_dim=edge_attr_dim
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)


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
    main(epochs=300, lr=0.001, hidden_features=256)

    #Seed Already implemented in the universe : 42
