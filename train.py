import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score
import wandb
from data_prep import HiKDataset, GraphDataset
from models import GraphTransformer

# Initialize Weights & Biases
wandb.init(project="graph-transformer-training")

# Configuration
config = {
    "batch_size": 128,
    "hidden_dim": 128,
    "input_dim": 3,
    "sample_length": 30,
    "step_size": 30,
    "num_heads": 4,
    "lr": 0.001,
    "epochs": 50
}
wandb.config.update(config)

# Data Preparation
data_location = "data"
dataset_name = "A"
dataset = HiKDataset(dataset_name, data_location, config["sample_length"], config["step_size"])
graph_dataset = GraphDataset(dataset)

train_size = int(0.8 * len(graph_dataset))
val_size = int(0.1 * len(graph_dataset))
test_size = len(graph_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(graph_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

# Model Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dim = graph_dataset[0].y.size(1)
model = GraphTransformer(config["input_dim"], config["hidden_dim"], output_dim, config["num_heads"]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
loss_fn = torch.nn.BCEWithLogitsLoss()

# Training Function
def train(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = loss_fn(out, batch.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# Evaluation Function
def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            preds = (torch.sigmoid(out) > 0.5).float()
            all_preds.append(preds.cpu())
            all_labels.append(batch.y.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return f1

# Training Loop
epochs = config["epochs"]
for epoch in range(epochs):
    train_loss = train(model, train_loader, optimizer, loss_fn)
    val_f1 = evaluate(model, val_loader)

    # Log metrics to wandb
    wandb.log({"train_loss": train_loss, "val_f1": val_f1})

    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val F1: {val_f1:.4f}")

# Testing
test_f1 = evaluate(model, test_loader)
wandb.log({"test_f1": test_f1})
print(f"Test F1 Score: {test_f1:.4f}")
