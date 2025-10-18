import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import random
import numpy as np
import json
from pathlib import Path
from config import get_config
from model.gpt import GPTConfig, GPT


# Setup configuration and device
cfg = get_config()
device = cfg["system"]["device"]
torch.manual_seed(cfg["system"]["seed"])
random.seed(cfg["system"]["seed"])
np.random.seed(cfg["system"]["seed"])

#load tokenized data
train_data = torch.load(cfg["data"]["train_file"])
val_data = torch.load(cfg["data"]["val_file"])

def create_dataset(data, block_size):
    xs, ys = [], []
    for i in range(len(data) - block_size):
        x = data[i : i + block_size]
        y = data[i + 1 : i + block_size + 1]
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)

X_train, Y_train = create_dataset(train_data, cfg["training"]["block_size"])
X_val, Y_val = create_dataset(val_data, cfg["training"]["block_size"])

train_ds = TensorDataset(X_train, Y_train)
val_ds = TensorDataset(X_val, Y_val)

train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True)
val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False)


# load model from model/gpt.py  
with open(cfg["data"]["vocab_file"], "r", encoding="utf-8") as f:
    vocab_data = json.load(f)
vocab_size = len(vocab_data["stoi"])

model_cfg = GPTConfig(
    vocab_size = cfg["tokenizer"]["vocab_size"],
    block_size=cfg["training"]["block_size"],
    n_embd=cfg["model"]["embed_dim"],
    n_head=cfg["model"]["num_heads"],
    n_layer=cfg["model"]["num_layers"],
    dropout=cfg["model"]["dropout"]
)

model = GPT(model_cfg).to(device)
#setup optimizer and loss function
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg["training"]["learning_rate"],
    weight_decay=cfg["training"]["weight_decay"]
)
criterion = nn.CrossEntropyLoss()

#Training loop
def evaluate(loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            total_loss += loss.item()
    return total_loss / len(loader)

best_val_loss = float("inf")

for epoch in range(cfg["training"]["n_epochs"]):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
        optimizer.step()
        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)
    val_loss = evaluate(val_loader)

    print(f"Epoch {epoch+1}/{cfg['training']['n_epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

#save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        Path("checkpoints").mkdir(exist_ok=True)
        torch.save(model.state_dict(), "checkpoints/best_model.pt")
        print("Saved new best model!")

print("\nTraining complete. Best val loss:", best_val_loss)
