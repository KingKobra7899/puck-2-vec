import copy
import json

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from puck2vec import Puck2Vec
from puck2vec_dataset import SeqDataset

LR = 1e-4
WEIGHT_DECAY = 1e-3
BATCH_SIZE = 32
MAX_EPOCHS = 64
PATIENCE = 5
VAL_FRAC = 0.1
N_FRAMES = 40
TRANSFORMER_DIM = 512
TRANS_LAYERS = 4
TRANS_HEADS = 4
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_PATH = "puck2vec.pth"

print(f"device: {DEVICE}", flush=True)
df = pd.read_csv("data/processed_ppt.csv")
dataset = SeqDataset(df, n_frames=N_FRAMES)
del df

model = Puck2Vec(n_frames=dataset.n_frames, transformer_dim=TRANSFORMER_DIM,
                 trans_layers=TRANS_LAYERS, trans_heads=TRANS_HEADS).to(DEVICE)

torch.manual_seed(SEED)
n_val = int(len(dataset) * VAL_FRAC)
train_set, val_set = random_split(dataset, [len(dataset) - n_val, n_val])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


def batch_loss(batch):
    b = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
    move_logits = model(player_x=b["x"], player_y=b["y"], off=b["off"],
                        player_dxs=b["dx"], player_dys=b["dy"])
    return F.cross_entropy(move_logits.flatten(0, 2), b["move"].flatten())


history = {"train": [], "val": []}
best_val, best_state, bad_epochs = float("inf"), None, 0

for epoch in range(MAX_EPOCHS):
    model.train()
    total = 0.0
    for i, batch in enumerate(train_loader):
        loss = batch_loss(batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()
        if (i + 1) % 200 == 0:
            print(f"epoch {epoch + 1} | batch {i + 1}/{len(train_loader)} | loss {loss.item():.4f}", flush=True)
    train_loss = total / len(train_loader)

    model.eval()
    with torch.no_grad():
        val_loss = sum(batch_loss(batch).item() for batch in val_loader) / len(val_loader)

    history["train"].append(train_loss)
    history["val"].append(val_loss)
    print(f"epoch {epoch + 1} | train {train_loss:.4f} | val {val_loss:.4f}", flush=True)

    if val_loss < best_val:
        best_val, best_state, bad_epochs = val_loss, copy.deepcopy(model.state_dict()), 0
        model_cpu = copy.deepcopy(model).cpu()
        model_cpu.load_state_dict({k: v.cpu() for k, v in best_state.items()})
        torch.save({"model": model_cpu, "player_index": dataset.player_index}, OUT_PATH)  # checkpoint every improvement
    else:
        bad_epochs += 1
        if bad_epochs >= PATIENCE:
            print(f"early stopping, val loss hasn't improved in {PATIENCE} epochs", flush=True)
            break

json.dump(history, open("history.json", "w"))
print(f"best val loss {best_val:.4f}, saved to {OUT_PATH}", flush=True)
