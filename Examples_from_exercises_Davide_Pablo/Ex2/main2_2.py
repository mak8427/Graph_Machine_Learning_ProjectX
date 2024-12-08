import torch
import torch_geometric as pyg
import numpy as np
from ogb.graphproppred import PygGraphPropPredDataset
import torch_scatter
import copy
import time
import random
from tqdm import tqdm
from ogb.graphproppred.mol_encoder import AtomEncoder
import matplotlib.pyplot as plt
import networkx as nx

# Find device
if torch.cuda.is_available():  # NVIDIA
    device = torch.device('cuda')
elif torch.backends.mps.is_available():  # Apple M1/M2
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print("Using device:", device)

# Define GCN Layer
class GCNLayer(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, activation=torch.nn.functional.relu):
        super(GCNLayer, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        self.activation = activation

    def forward(self, H: torch.Tensor, edge_index: torch.Tensor):
        edge_index, _ = pyg.utils.add_self_loops(edge_index, num_nodes=H.size(0))
        row, col = edge_index
        deg = pyg.utils.degree(row, H.size(0))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        H = self.linear(H)
        H = H[col] * norm.unsqueeze(-1)
        H = torch_scatter.scatter_add(H, row, dim=0)

        if self.activation is not None:
            H = self.activation(H)
        return H

# Modify GraphNet to include an embedding layer
class GraphNet(torch.nn.Module):
    def __init__(self, num_node_types: int, out_features: int, hidden_features: int, activation=torch.nn.functional.relu, dropout=0.1):
        super(GraphNet, self).__init__()
        self.embedding = torch.nn.Embedding(num_node_types, hidden_features)
        self.gcn1 = GCNLayer(hidden_features, hidden_features, activation)
        self.gcn2 = GCNLayer(hidden_features, hidden_features, activation)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(hidden_features, out_features)

    def forward(self, H: torch.Tensor, edge_index: torch.Tensor, batch=None):
        H = self.embedding(H.squeeze(-1))
        H = self.gcn1(H, edge_index)
        H = self.gcn2(H, edge_index)
        H = self.dropout(H)
        if batch is not None:
            H = pyg.nn.global_mean_pool(H, batch)
        out = self.linear(H)
        return out.squeeze()

def get_accuracy(model, graph, mask):
    model.eval()
    with torch.no_grad():
        outputs = model(graph.x, graph.edge_index)
    correct = (outputs[mask].argmax(-1) == graph.y[mask]).sum()
    return int(correct) / int(mask.sum())

if __name__ == '__main__':
    # Load Cora dataset
    print("Loading Cora dataset...")
    cora = pyg.datasets.Planetoid(root="dataset/cora", name="Cora")
    cora_graph = cora[0].to(device)
    print("Cora dataset loaded.")

    # Task 1: Find the second-biggest label class and what it stands for
    labels, counts = torch.unique(cora_graph.y, return_counts=True)
    sorted_counts = sorted(zip(labels.tolist(), counts.tolist()), key=lambda x: x[1], reverse=True)
    print("Label counts (sorted):", sorted_counts)

    # Mapping of labels to class names
    class_names = [
        'Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
        'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory'
    ]

    second_biggest_label = sorted_counts[1][0]
    second_biggest_class = class_names[second_biggest_label]
    print(f"The second-biggest label class is {second_biggest_label}, which stands for '{second_biggest_class}'.")

    # Model parameters for Cora
    in_features = cora_graph.num_node_features
    hidden_features = 16
    out_features = cora.num_classes
    learning_rate = 0.001
    weight_decay = 5e-4
    num_epochs = 400

    # Define GraphNetCora for Cora dataset
    class GraphNetCora(torch.nn.Module):
        def __init__(self, in_features: int, out_features: int, hidden_features: int, activation=torch.nn.functional.relu, dropout=0.1):
            super(GraphNetCora, self).__init__()
            self.gcn1 = GCNLayer(in_features, hidden_features, activation)
            self.dropout = torch.nn.Dropout(dropout)
            self.gcn2 = GCNLayer(hidden_features, out_features, activation=None)

        def forward(self, H: torch.Tensor, edge_index: torch.Tensor):
            H = self.gcn1(H, edge_index)
            H = self.dropout(H)
            H = self.gcn2(H, edge_index)
            return H

    # Initialize model, loss function, and optimizer for Cora
    model_cora = GraphNetCora(in_features, out_features, hidden_features, dropout=0.5).to(device)
    criterion_cora = torch.nn.CrossEntropyLoss()
    optimizer_cora = torch.optim.Adam(model_cora.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_mask = cora_graph.train_mask
    val_mask = cora_graph.val_mask
    test_mask = cora_graph.test_mask

    print("\nStarting training on Cora dataset...")
    # Training loop for Cora
    for epoch in range(num_epochs):
        model_cora.train()
        optimizer_cora.zero_grad()
        out = model_cora(cora_graph.x, cora_graph.edge_index)
        loss = criterion_cora(out[train_mask], cora_graph.y[train_mask])
        loss.backward()
        optimizer_cora.step()

        val_acc = get_accuracy(model_cora, cora_graph, val_mask)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch+1}, Loss: {loss.item():.4f}, Val Accuracy: {val_acc:.4f}')

    print("Finished training on Cora dataset.")

    print("\nStarting testing on Cora dataset...")
    test_acc = get_accuracy(model_cora, cora_graph, test_mask)
    print(f'Test Accuracy on Cora dataset: {test_acc:.4f}')
    print("Finished testing on Cora dataset.")

    # Load ZINC dataset
    print("\nLoading ZINC dataset...")
    dataset = pyg.datasets.ZINC(root='dataset/ZINC', split='train', subset=True)
    dataset_val = pyg.datasets.ZINC(root='dataset/ZINC', split='val', subset=True)
    dataset_test = pyg.datasets.ZINC(root='dataset/ZINC', split='test', subset=True)
    print("ZINC dataset loaded.")

    batch_size = 128
    num_workers = 0  # Set to 0 for compatibility

    train_loader = pyg.loader.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = pyg.loader.DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = pyg.loader.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Determine the number of node types in the ZINC dataset
    def get_num_node_types(dataset):
        node_types = set()
        for data in dataset:
            node_types.update(data.x.squeeze(-1).tolist())
        return max(node_types) + 1  # Assuming node types start from 0

    num_node_types = get_num_node_types(dataset)

    # Task 1: Count the number of HCO molecules in the train set
    # Atom types corresponding to H, C, and O
    atom_type_to_atomic_number = {
        0: 6,   # C
        1: 8,   # O
        2: 7,   # N
        3: 9,   # F
        4: 16,  # S
        5: 17,  # Cl
        6: 35,  # Br
        7: 1,   # H
        8: 15,  # P
        9: 53,  # I
        10: 5,  # B
        11: 14, # Si
        12: 34, # Se
        13: 52, # Te
        14: 85, # At
        15: 32, # Ge
        16: 50, # Sn
        17: 51, # Sb
        18: 83, # Bi
        19: 84, # Po
        20: 116 # Lv
    }

    # Identify atom types for H, C, O
    hco_atomic_numbers = [1, 6, 8]
    hco_atom_types = [atom_type for atom_type, atomic_num in atom_type_to_atomic_number.items() if atomic_num in hco_atomic_numbers]

    # Count the number of HCO molecules in the train set
    hco_molecule_count = 0
    for data in dataset:
        atom_types_in_molecule = set(data.x.squeeze(-1).tolist())
        if atom_types_in_molecule.issubset(hco_atom_types):
            hco_molecule_count += 1
    print(f"Number of HCO molecules in the train set: {hco_molecule_count}")

    # --- Integrated Atom Type Analysis and Visualization ---

    # Count the occurrences of each atom type in the training set
    atom_types = range(num_node_types)
    atom_counts = {atom_type: 0 for atom_type in atom_types}

    for data in dataset:
        for idx in data.x.squeeze(-1):
            atom_type = idx.item()
            atom_counts[atom_type] += 1

    atom_labels = [str(atom_type) for atom_type in atom_types]
    counts = [atom_counts[atom_type] for atom_type in atom_types]

    print("\nAtom type counts in the training set:")
    for atom_type in atom_types:
        count = atom_counts[atom_type]
        print(f"Atom type {atom_type}: {count} occurrences")

    # Plot the distribution of atom types
    plt.figure(figsize=(10, 6))
    plt.bar(atom_labels, counts, color='skyblue')
    plt.xlabel('Atom Types')
    plt.ylabel('Counts')
    plt.title('Distribution of Atom Types in ZINC Training Set')
    plt.show()

    def vis_zinc_molecule(pyg_graph, file_path):
        # Convert PyG graph to NetworkX graph
        nx_graph = pyg.utils.to_networkx(pyg_graph, to_undirected=True)

        # Mapping of atom types to chemical symbols
        atom_type_to_symbol = {
            0: 'C',
            1: 'O',
            2: 'N',
            3: 'F',
            4: 'S',
            5: 'Cl',
            6: 'Br',
            7: 'H',
            8: 'P',
            9: 'I',
            10: 'B',
            11: 'Si',
            12: 'Se',
            13: 'Te',
            14: 'At',
            15: 'Ge',
            16: 'Sn',
            17: 'Sb',
            18: 'Bi',
            19: 'Po',
            20: 'Lv'
        }

        # Get positions for nodes using a layout
        pos = nx.kamada_kawai_layout(nx_graph)

        # Draw nodes with labels
        node_labels = {i: atom_type_to_symbol.get(pyg_graph.x[i].item(), '?') for i in nx_graph.nodes()}
        nx.draw_networkx_nodes(nx_graph, pos, node_color='lightblue', node_size=500)
        nx.draw_networkx_labels(nx_graph, pos, labels=node_labels, font_size=12)

        # Draw edges
        nx.draw_networkx_edges(nx_graph, pos)

        plt.title('Visualization of a Molecule from ZINC Dataset')
        plt.axis('off')
        plt.savefig(file_path)  # Save the figure to the specified file path
        plt.close()  # Close the plot to free up memory

    # Visualize a sample molecule
    vis_zinc_molecule(dataset[5], 'molecule_visualization.png')

    # --- End of Integration ---

    # Model parameters for ZINC
    hidden_features = 128
    out_features = 1
    learning_rate = 0.001
    num_epochs = 70

    # Define OneHotEncoder
    class OneHotEncoder(torch.nn.Module):
        def __init__(self, num_classes):
            super(OneHotEncoder, self).__init__()
            self.num_classes = num_classes

        def forward(self, x):
            return torch.nn.functional.one_hot(x.squeeze(-1), num_classes=self.num_classes).float()

    # Define GraphNet with AtomEncoder
    class GraphNetAtomEncoder(torch.nn.Module):
        def __init__(self, hidden_features: int, out_features: int, activation=torch.nn.functional.relu, dropout=0.1):
            super(GraphNetAtomEncoder, self).__init__()
            self.embedding = AtomEncoder(hidden_features)
            self.gcn1 = GCNLayer(hidden_features, hidden_features, activation)
            self.gcn2 = GCNLayer(hidden_features, hidden_features, activation)
            self.dropout = torch.nn.Dropout(dropout)
            self.linear = torch.nn.Linear(hidden_features, out_features)

        def forward(self, H: torch.Tensor, edge_index: torch.Tensor, batch=None):
            H = self.embedding(H)
            H = self.gcn1(H, edge_index)
            H = self.gcn2(H, edge_index)
            H = self.dropout(H)
            if batch is not None:
                H = pyg.nn.global_mean_pool(H, batch)
            out = self.linear(H)
            return out.squeeze()

    # Define GraphNet with OneHotEncoder
    class GraphNetOneHot(torch.nn.Module):
        def __init__(self, num_node_types: int, out_features: int, hidden_features: int, activation=torch.nn.functional.relu, dropout=0.1):
            super(GraphNetOneHot, self).__init__()
            self.one_hot_encoder = OneHotEncoder(num_node_types)
            self.linear_in = torch.nn.Linear(num_node_types, hidden_features)
            self.gcn1 = GCNLayer(hidden_features, hidden_features, activation)
            self.gcn2 = GCNLayer(hidden_features, hidden_features, activation)
            self.dropout = torch.nn.Dropout(dropout)
            self.linear = torch.nn.Linear(hidden_features, out_features)

        def forward(self, H: torch.Tensor, edge_index: torch.Tensor, batch=None):
            H = self.one_hot_encoder(H)
            H = self.linear_in(H)
            H = self.gcn1(H, edge_index)
            H = self.gcn2(H, edge_index)
            H = self.dropout(H)
            if batch is not None:
                H = pyg.nn.global_mean_pool(H, batch)
            out = self.linear(H)
            return out.squeeze()

    # Initialize models
    model_atomencoder = GraphNetAtomEncoder(hidden_features, out_features, dropout=0.1).to(device)
    model_onehot = GraphNetOneHot(num_node_types, out_features, hidden_features, dropout=0.1).to(device)

    # Loss function and optimizers
    criterion = torch.nn.L1Loss()  # Mean Absolute Error
    optimizer_atomencoder = torch.optim.Adam(model_atomencoder.parameters(), lr=learning_rate)
    optimizer_onehot = torch.optim.Adam(model_onehot.parameters(), lr=learning_rate)

    # Training and evaluation functions
    def train_and_evaluate(model, optimizer, train_loader, val_loader, num_epochs, model_name=""):
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                out = model(data.x, data.edge_index, data.batch)
                loss = criterion(out, data.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * data.num_graphs
            total_loss /= len(train_loader.dataset)
            print(f'[{model_name}] Epoch: {epoch+1}, Training Loss: {total_loss:.4f}')

            # Validation
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for data in val_loader:
                    data = data.to(device)
                    out = model(data.x, data.edge_index, data.batch)
                    loss = criterion(out, data.y)
                    val_loss += loss.item() * data.num_graphs
                val_loss /= len(val_loader.dataset)
            print(f'[{model_name}] Validation Loss: {val_loss:.4f}')

    def test_model(model, test_loader, model_name=""):
        model.eval()
        with torch.no_grad():
            test_loss = 0
            for data in test_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.batch)
                loss = criterion(out, data.y)
                test_loss += loss.item() * data.num_graphs
            test_loss /= len(test_loader.dataset)
        print(f'[{model_name}] Test Loss on ZINC dataset: {test_loss:.4f}')

    # Training AtomEncoder model
    print("\nStarting training on ZINC dataset witMh AtomEncoder...")
    train_and_evaluate(model_atomencoder, optimizer_atomencoder, train_loader, val_loader, num_epochs, model_name="AtomEncoder")
    print("Finished training on ZINC dataset with AtomEncoder.")

    # Training OneHot Encoding model
    print("\nStarting training on ZINC dataset with OneHot Encoding...")
    train_and_evaluate(model_onehot, optimizer_onehot, train_loader, val_loader, num_epochs, model_name="OneHot")
    print("Finished training on ZINC dataset with OneHot Encoding.")

    # Testing AtomEncoder model
    print("\nStarting testing on ZINC dataset with AtomEncoder...")
    test_model(model_atomencoder, test_loader, model_name="AtomEncoder")
    print("Finished testing on ZINC dataset with AtomEncoder.")

    # Testing OneHot Encoding model
    print("\nStarting testing on ZINC dataset with OneHot Encoding...")
    test_model(model_onehot, test_loader, model_name="OneHot")
    print("Finished testing on ZINC dataset with OneHot Encoding.")

#%%
