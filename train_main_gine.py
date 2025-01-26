import torch
from torch_geometric.nn import DataParallel  # <-- Use PyG DataParallel
from sklearn.metrics import f1_score
import wandb
from hik.data.scene import Scene
from models import GINE  # Import the GINE model
from data_prep import generate_dataloaders
from tqdm import tqdm  # Import tqdm for progress bars
import os

# Initialize WandB
wandb.init(project="main-gine-training", entity="kaisar-dauletbek")

# Configuration
config = {
    "batch_size": 1028,
    "hidden_dim": 512,  # Hidden dimension for GINE
    "input_dim": 4,  # Node feature dimension +3 for the sit_stand label
    "num_layers": 8,  # Number of GINE layers
    "lr": 0.001,
    "epochs": 200,
    "model_save_path": "best_main_gine_model.pth",
    "early_stop_patience": 10  # Patience for early stopping
}
wandb.config.update(config)

# Load dataset and create dataloaders
graph_dataset, train_loader, val_loader, test_loader = generate_dataloaders(config["batch_size"])

# Model Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dim = graph_dataset[0].y.size(1)
# output_dim = 80

# Load the sit_stand model
sit_stand_model = GINE(
    num_features=3,
    num_classes=3,
    hidden_channels=256,
    num_layers=8,
    use_batch_norm=True
).to(device)
sit_stand_model.load_state_dict(torch.load("best_sit_stand.pth"))
sit_stand_model = sit_stand_model.to(device)
sit_stand_model.eval()


model = GINE(
    num_features=config["input_dim"],
    num_classes=output_dim,
    hidden_channels=128,
    num_layers=config["num_layers"],
    use_batch_norm=True
).to(device)

# Optimizer, Scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6)
loss_fn = torch.nn.BCEWithLogitsLoss()  # Multi-label classification loss

# Early Stopper Class
class EarlyStopper:
    def __init__(self, patience, mode="max"):
        self.patience = patience
        self.mode = mode
        self.best_metric = None
        self.counter = 0
        self.early_stop = False

    def step(self, metric):
        if self.best_metric is None or (self.mode == "max" and metric > self.best_metric) or (self.mode == "min" and metric < self.best_metric):
            self.best_metric = metric
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

# Instantiate Early Stopper
early_stopper = EarlyStopper(patience=config["early_stop_patience"], mode="max")

# Training Function
def train(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    loader = tqdm(loader, desc="Training", leave=False)
    for batch in loader:
        batch = batch.to(device)  # Move entire batch to the correct device
        optimizer.zero_grad()
        sit_stand_out = sit_stand_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        #sit_stand_out is batch_size x 3, we need to add it to the input features as column entries
        # thus, we need to repeate each row of the sit_stand_out tensor 145 times
        # sit_stand_out = sit_stand_out.repeat(145, 1)
        sit_stand_pred = torch.argmax(sit_stand_out, dim=1)
        sit_stand_pred = torch.repeat_interleave(sit_stand_pred, 145, dim=0).unsqueeze(0).T
        batch.x = torch.cat((batch.x, sit_stand_pred), 1)
        # print(sit_stand_out.shape)
        # print(batch.x.shape)
        # input()
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
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
    loader = tqdm(loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)  # Move entire batch to the correct device
            sit_stand_out = sit_stand_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            # sit_stand_out = sit_stand_out.repeat(145, 1)
            sit_stand_pred = torch.argmax(sit_stand_out, dim=1)
            sit_stand_pred = torch.repeat_interleave(sit_stand_pred, 145, dim=0).unsqueeze(0).T
            batch.x = torch.cat((batch.x, sit_stand_pred), 1)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            preds = (torch.sigmoid(out) > 0.5).float()
            all_preds.append(preds.cpu())
            all_labels.append(batch.y.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return f1

# Training Loop
best_val_f1 = 0.0
for epoch in range(config["epochs"]):
    train_loss = train(model, train_loader, optimizer, loss_fn)
    val_f1 = evaluate(model, val_loader)

    # Save best model
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), config["model_save_path"])
        print(f"Best model saved with F1: {best_val_f1:.4f}")

    # Pass validation F1 score to the scheduler
    scheduler.step(val_f1)

    # Check early stopping
    early_stopper.step(val_f1)
    if early_stopper.early_stop:
        print(f"Early stopping triggered at epoch {epoch+1}")
        break

    # WandB logging
    wandb.log({
        "train_loss": train_loss,
        "val_f1": val_f1,
        "best_val_f1": best_val_f1,
        "learning_rate": optimizer.param_groups[0]['lr']
    })

    print(f"Epoch {epoch+1}/{config['epochs']} "
          f"| Train Loss: {train_loss:.4f} "
          f"| Val F1: {val_f1:.4f} "
          f"| LR: {optimizer.param_groups[0]['lr']:.6f}")

# Load best model and test
model.load_state_dict(torch.load(config["model_save_path"]))
model = model.to(device)

test_f1 = evaluate(model, test_loader)
wandb.log({"test_f1": test_f1})
print(f"Test F1 Score: {test_f1:.4f}")
