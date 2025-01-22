import torch
from torch_geometric.nn import DataParallel  # <-- use PyG DataParallel
from sklearn.metrics import f1_score
import wandb
from data_prep import generate_dataloaders
from models import GraphTransformer
import os

# Initialize WandB
wandb.init(project="graph-transformer-training", entity="kaisar-dauletbek")

# Configuration
config = {
    "batch_size": 128,
    "hidden_dim": 256,
    "input_dim": 3,
    "num_heads": 16,
    "lr": 0.001,
    "epochs": 100,
    "model_save_path": "best_model.pth"
}
wandb.config.update(config)


# dataset_name = "A"

# Load dataset and create dataloaders
graph_dataset, train_loader, val_loader, test_loader = generate_dataloaders(config["batch_size"])

# Model Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dim = graph_dataset[0].y.size(1)

model = GraphTransformer(
    config["input_dim"],
    config["hidden_dim"],
    output_dim,
    config["num_heads"]
).to(device)

# Optimizer, Scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=1e-6)
loss_fn = torch.nn.BCEWithLogitsLoss()

# Training Function
def train(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)  # <-- Move entire batch to the correct device
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
            batch = batch.to(device)  # <-- Move entire batch to the correct device
            out = model(batch)
            preds = (torch.sigmoid(out) > 0.5).float()
            all_preds.append(preds.cpu())
            all_labels.append(batch.y.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return f1


# Training Loop (same as before, but without DataParallel checks)
best_val_f1 = 0.0
for epoch in range(config["epochs"]):
    train_loss = train(model, train_loader, optimizer, loss_fn)
    val_f1 = evaluate(model, val_loader)

    # Save best model
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), config["model_save_path"])
        print(f"Best model saved with F1: {best_val_f1:.4f}")

    scheduler.step()

    # WandB logging
    wandb.log({
        "train_loss": train_loss,
        "val_f1": val_f1,
        "best_val_f1": best_val_f1,
        "learning_rate": scheduler.get_last_lr()[0]
    })

    print(f"Epoch {epoch+1}/{config['epochs']} "
          f"| Train Loss: {train_loss:.4f} "
          f"| Val F1: {val_f1:.4f} "
          f"| LR: {scheduler.get_last_lr()[0]:.6f}")

# Load best model and test
model.load_state_dict(torch.load(config["model_save_path"]))
model = model.to(device)

test_f1 = evaluate(model, test_loader)
wandb.log({"test_f1": test_f1})
print(f"Test F1 Score: {test_f1:.4f}")
