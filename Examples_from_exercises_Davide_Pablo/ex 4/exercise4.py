import torch
import torch_geometric as pyg
from torch_cluster import random_walk
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class PQWalkSampler:
    def __init__(self, edge_index, p=1, q=1):
        """
        Sampler for generating biased random walks using Node2Vec strategy.
        Args:
            edge_index (torch.Tensor): Edge index of the graph.
            p (float): Return parameter.
            q (float): In-out parameter.
        """
        self.edge_index = edge_index
        self.p = p
        self.q = q

    def sample_walks(self, start_nodes, walk_length):
        """
        Sample biased random walks starting from the given nodes.
        Args:
            start_nodes (torch.Tensor): Starting nodes for the walks.
            walk_length (int): Length of each walk.
        Returns:
            torch.Tensor: Tensor of walks.
        """
        walks = random_walk(self.edge_index[0], self.edge_index[1], start_nodes, walk_length,
                            p=self.p, q=self.q)
        return walks

class Node2Vec(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super(Node2Vec, self).__init__()
        self.node_embeddings = nn.Embedding(num_nodes, embedding_dim)

    def forward(self, nodes):
        return self.node_embeddings(nodes)

def generate_negative_samples(batch_size, num_nodes, num_neg_samples):
    neg_samples = torch.randint(0, num_nodes, (batch_size, num_neg_samples))
    return neg_samples

def node2vec_loss(model, pos_u, pos_v, neg_v):
    u_emb = model(pos_u)
    v_emb = model(pos_v)
    pos_score = torch.mul(u_emb, v_emb).sum(dim=1)
    pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-15).mean()

    neg_emb = model(neg_v.view(-1))
    neg_emb = neg_emb.view(neg_v.size(0), neg_v.size(1), -1)
    u_emb_expanded = u_emb.unsqueeze(1)
    neg_score = torch.bmm(neg_emb, u_emb_expanded.transpose(1, 2)).squeeze()
    neg_loss = -torch.log(torch.sigmoid(-neg_score) + 1e-15).mean()

    return pos_loss + neg_loss

def evaluate_link_prediction(embeddings, edge_label_index, edge_label, method='dot'):
    z = embeddings.cpu().detach().numpy()
    src_indices = edge_label_index[0].cpu().numpy()
    dst_indices = edge_label_index[1].cpu().numpy()
    source_emb = z[src_indices]
    target_emb = z[dst_indices]

    if method == 'dot':
        scores = np.sum(source_emb * target_emb, axis=1)
        preds = (scores > 0).astype(int)
        probs = scores  # Use raw scores as probabilities for ROC curve
    elif method == 'hadamard':
        scores = np.prod(source_emb * target_emb, axis=1)
        preds = (scores > 0).astype(int)
        probs = scores
    elif method == 'concat':
        features = np.hstack([source_emb, target_emb])
        clf = LogisticRegression(max_iter=1000)
        clf.fit(features, edge_label.cpu().numpy())
        preds = clf.predict(features)
        probs = clf.predict_proba(features)[:, 1]
    elif method == 'abs_diff':
        features = np.abs(source_emb - target_emb)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(features, edge_label.cpu().numpy())
        preds = clf.predict(features)
        probs = clf.predict_proba(features)[:, 1]
    else:
        raise ValueError("Unknown method for combining embeddings.")

    roc_auc = roc_auc_score(edge_label.cpu().numpy(), probs)
    ap_score = average_precision_score(edge_label.cpu().numpy(), probs)

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(edge_label.cpu().numpy(), probs)
    roc_curve_data = (fpr, tpr)

    return roc_auc, ap_score, roc_curve_data

def get_accuracy(model, embeddings, y, mask):
    out = model(embeddings[mask])
    pred = out.argmax(dim=1)
    acc = accuracy_score(y[mask].cpu().numpy(), pred.cpu().detach().numpy())
    return acc

if __name__ == "__main__":
    # Set random seed
    seed = 42
    set_random_seed(seed)

    # Find device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    dataset = pyg.datasets.Planetoid(root='./dataset/cora', name='Cora')
    cora = dataset[0]

    # Split the data for link prediction
    link_splitter = pyg.transforms.RandomLinkSplit(is_undirected=True)
    train_data, val_data, test_data = link_splitter(cora)

    # Use the training edge_index for embedding training
    edge_index = train_data.edge_index.to(device)
    num_nodes = cora.num_nodes  # Nodes remain the same

    # Parameters
    num_walks_per_node = 10
    walk_length = 5
    context_size = 2
    embedding_dim = 128
    num_epochs = 1
    num_neg_samples = 5
    learning_rate = 0.01
    p = 1  # Return parameter
    q = 1  # In-out parameter

    # Initialize sampler with training edges
    sampler = PQWalkSampler(edge_index=edge_index, p=p, q=q)

    # Generate walks (fixed for the entire training process)
    start_nodes = torch.arange(num_nodes).repeat_interleave(num_walks_per_node).to(device)
    walks = sampler.sample_walks(start_nodes, walk_length)
    walks = walks.view(-1, walk_length + 1)

    # Create a dataset and dataloader
    class WalkDataset(torch.utils.data.Dataset):
        def __init__(self, walks):
            self.walks = walks

        def __len__(self):
            return self.walks.size(0)

        def __getitem__(self, idx):
            return self.walks[idx]

    walk_dataset = WalkDataset(walks)
    walk_loader = DataLoader(walk_dataset, batch_size=128, shuffle=True)

    # Initialize Node2Vec model and optimizer
    embedding_model = Node2Vec(num_nodes=num_nodes, embedding_dim=embedding_dim).to(device)
    optimizer = optim.Adam(embedding_model.parameters(), lr=learning_rate)

    # Lists to store training loss
    node2vec_losses = []

    # Training Node2Vec with fixed walks
    print("Training Node2Vec...")
    embedding_model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_walks in tqdm(walk_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch_walks = batch_walks.to(device)
            batch_loss = 0
            # Generate positive pairs and negative samples within the batch
            for walk in batch_walks:
                for i in range(len(walk)):
                    context_start = max(0, i - context_size)
                    context_end = min(len(walk), i + context_size + 1)

                    # Collect context nodes
                    pos_v_list = []

                    # Left context
                    if context_start < i:
                        left_context = walk[context_start:i]
                        if len(left_context) > 0:
                            pos_v_list.append(left_context)

                    # Right context
                    if i + 1 < context_end:
                        right_context = walk[i+1:context_end]
                        if len(right_context) > 0:
                            pos_v_list.append(right_context)

                    if len(pos_v_list) == 0:
                        continue  # No context nodes, skip

                    # Concatenate context nodes
                    pos_v = torch.cat(pos_v_list)

                    # Repeat central node to match size
                    pos_u = walk[i].repeat(len(pos_v))

                    # Generate negative samples
                    neg_v = generate_negative_samples(len(pos_u), num_nodes, num_neg_samples).to(device)

                    # Compute loss
                    loss = node2vec_loss(embedding_model, pos_u, pos_v.to(device), neg_v)
                    batch_loss += loss.item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            total_loss += batch_loss
        avg_loss = total_loss / len(walk_loader)
        node2vec_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Plot Node2Vec training loss
    plt.figure()
    plt.plot(range(1, num_epochs + 1), node2vec_losses, marker='o')
    plt.title('Node2Vec Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('node2vec_training_loss.png')  # Save the figure
    plt.show()

    # Node Classification on Cora
    cora = cora.to(device)
    print("Evaluating node classification on Cora...")

    # Using the professor's code for the classifier
    classifier_model = torch.nn.Sequential(
        torch.nn.Linear(embedding_dim, 256),  # Input layer
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),            # Hidden layer 2
        torch.nn.ReLU(),
        torch.nn.Linear(128, dataset.num_classes),  # Output layer
    )
    classifier_model = classifier_model.to(device)

    optimizer_cls = torch.optim.AdamW(classifier_model.parameters(), lr=0.01)  # Define an optimizer
    criterion = torch.nn.CrossEntropyLoss()  # Define loss function

    node2vec_embeddings = embedding_model.node_embeddings.weight.detach().to(device)

    # Lists to store training and validation metrics
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(400):  # 100 epochs
        classifier_model.train()
        optimizer_cls.zero_grad()
        out = classifier_model(node2vec_embeddings[cora.train_mask])  # Forward pass
        loss = criterion(out, cora.y[cora.train_mask])
        loss.backward()
        optimizer_cls.step()

        # Evaluate on validation set
        classifier_model.eval()
        with torch.no_grad():
            val_out = classifier_model(node2vec_embeddings[cora.val_mask])
            val_loss = criterion(val_out, cora.y[cora.val_mask])
            val_pred = val_out.argmax(dim=1)
            val_acc = accuracy_score(cora.y[cora.val_mask].cpu().numpy(), val_pred.cpu().numpy())

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        val_accuracies.append(val_acc)

        # Print out loss info
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Train Loss: {loss.item():.3e}, Val Loss: {val_loss.item():.3e}, Val Acc: {val_acc:.4f}")

    # Plot training and validation loss
    epochs = range(1, 401)
    plt.figure()
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Classifier Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('classifier_loss.png')  # Save the figure
    plt.show()

    # Plot validation accuracy
    plt.figure()
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.title('Classifier Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('classifier_accuracy.png')  # Save the figure
    plt.show()

    # Evaluate accuracy
    classifier_model.eval()
    train_acc = get_accuracy(classifier_model, node2vec_embeddings, cora.y, cora.train_mask)
    val_acc = get_accuracy(classifier_model, node2vec_embeddings, cora.y, cora.val_mask)
    test_acc = get_accuracy(classifier_model, node2vec_embeddings, cora.y, cora.test_mask)

    print(f"Node classification accuracy for Cora: {test_acc:.2f} (train: {train_acc:.2f}, val: {val_acc:.2f})")

    # Detailed Classification Report
    y_true = cora.y[cora.test_mask].cpu().numpy()
    y_pred = classifier_model(node2vec_embeddings[cora.test_mask]).argmax(dim=1).cpu().numpy()
    print("Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    # Link Prediction on Cora
    print("Evaluating link prediction on Cora...")
    # The positive and negative edges are in "edge_label_index" with "edge_label"
    # indicating whether an edge is a true edge or not.

    # Evaluate with different combination methods
    embeddings = node2vec_embeddings
    methods = ['dot', 'hadamard', 'concat', 'abs_diff']
    plt.figure()
    for method in methods:
        roc_auc, ap_score, roc_curve_data = evaluate_link_prediction(
            embeddings,
            test_data.edge_label_index,
            test_data.edge_label,
            method=method
        )
        fpr, tpr = roc_curve_data
        plt.plot(fpr, tpr, label=f'{method} (AUC = {roc_auc:.4f})')
        print(f"Method: {method}, ROC AUC: {roc_auc:.4f}, AP Score: {ap_score:.4f}")
    plt.title('ROC Curves for Link Prediction Methods')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig('link_prediction_roc_curves.png')  # Save the figure
    plt.show()

#%%
