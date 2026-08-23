import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from dataset  import get_dataloaders
from gnn_model import GNNClassifier, count_parameters


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        logits, _ = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(logits, batch.y.squeeze())
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch.y.squeeze().cpu().numpy())

    n = len(loader.dataset)
    avg_loss  = total_loss / n
    macro_f1  = f1_score(all_labels, all_preds, average="macro",  zero_division=0)
    micro_f1  = f1_score(all_labels, all_preds, average="micro",  zero_division=0)
    accuracy  = accuracy_score(all_labels, all_preds)

    return avg_loss, macro_f1, micro_f1, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch in loader:
        batch = batch.to(device)
        logits, _ = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(logits, batch.y.squeeze())

        total_loss += loss.item() * batch.num_graphs
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch.y.squeeze().cpu().numpy())

    n = len(loader.dataset)
    avg_loss = total_loss / n
    macro_f1 = f1_score(all_labels, all_preds, average="macro",  zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average="micro",  zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, macro_f1, micro_f1, accuracy, all_preds, all_labels


def plot_training_curves(history: dict, save_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()

    axes[1].plot(history["train_macro_f1"], label="Train")
    axes[1].plot(history["val_macro_f1"],   label="Val")
    axes[1].set_title("Macro-F1"); axes[1].set_xlabel("Epoch"); axes[1].legend()

    axes[2].plot(history["train_acc"], label="Train")
    axes[2].plot(history["val_acc"],   label="Val")
    axes[2].set_title("Accuracy"); axes[2].set_xlabel("Epoch"); axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Training curves saved → {save_path}")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f" Task 2: GNN Genre Classification  [{args.gnn_type}]")
    print(f" Device : {device}")
    print(f"{'='*60}\n")

    train_loader, val_loader, test_loader, genre_to_idx, idx_to_genre = get_dataloaders(
        splits_dir    = args.splits_dir,
        graph_dir     = args.graph_dir,
        processed_dir = args.processed_dir,
        batch_size    = args.batch_size,
        sim_threshold = args.sim_threshold,
        num_workers   = args.num_workers,
    )
    num_classes = len(genre_to_idx)
    print(f" Classes: {num_classes}  |  "
          f"Train: {len(train_loader.dataset)}  "
          f"Val: {len(val_loader.dataset)}  "
          f"Test: {len(test_loader.dataset)}\n")

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{args.results_dir}/genre_mapping.json", "w") as f:
        json.dump({"genre_to_idx": genre_to_idx, "idx_to_genre": idx_to_genre}, f, indent=2)

    model = GNNClassifier(
        input_dim      = 140,
        hidden_dim     = args.hidden_dim,
        gnn_output_dim = args.gnn_output_dim,
        num_classes    = num_classes,
        num_layers     = args.num_layers,
        dropout        = args.dropout,
        gnn_type       = args.gnn_type,
    ).to(device)

    print(f" Model: {args.gnn_type}  —  {count_parameters(model):,} trainable parameters")

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    history = {k: [] for k in [
        "train_loss", "train_macro_f1", "train_micro_f1", "train_acc",
        "val_loss",   "val_macro_f1",   "val_micro_f1",   "val_acc",
    ]}

    best_val_macro_f1 = -1.0
    ckpt_path = f"{args.results_dir}/task2_{args.gnn_type}_best.pt"

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_mf1, tr_uf1, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        vl_loss, vl_mf1, vl_uf1, vl_acc, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_macro_f1"].append(tr_mf1)
        history["train_micro_f1"].append(tr_uf1)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_macro_f1"].append(vl_mf1)
        history["val_micro_f1"].append(vl_uf1)
        history["val_acc"].append(vl_acc)

        print(f" Epoch {epoch:03d}/{args.epochs}  "
              f"| Train Loss {tr_loss:.4f}  Macro-F1 {tr_mf1:.4f}  Acc {tr_acc:.4f}  "
              f"| Val   Loss {vl_loss:.4f}  Macro-F1 {vl_mf1:.4f}  Acc {vl_acc:.4f}")

        if vl_mf1 > best_val_macro_f1:
            best_val_macro_f1 = vl_mf1
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "val_macro_f1": vl_mf1,
                "genre_to_idx": genre_to_idx,
                "args":         vars(args),
            }, ckpt_path)
            print(f"   ✓ Best model saved (val Macro-F1 = {vl_mf1:.4f})")

    print(f"\n Loading best checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    ts_loss, ts_mf1, ts_uf1, ts_acc, ts_preds, ts_labels = evaluate(
        model, test_loader, criterion, device
    )
    print(f"\n{'─'*50}")
    print(f" TEST RESULTS  [{args.gnn_type}]")
    print(f"   Loss      : {ts_loss:.4f}")
    print(f"   Macro-F1  : {ts_mf1:.4f}")
    print(f"   Micro-F1  : {ts_uf1:.4f}")
    print(f"   Accuracy  : {ts_acc:.4f}")
    print(f"{'─'*50}\n")

    metrics = {
        "model"           : args.gnn_type,
        "best_epoch"      : int(ckpt["epoch"]),
        "test_loss"       : ts_loss,
        "test_macro_f1"   : ts_mf1,
        "test_micro_f1"   : ts_uf1,
        "test_accuracy"   : ts_acc,
        "best_val_macro_f1": best_val_macro_f1,
        "history"         : history,
        "hyperparams"     : vars(args),
    }
    metrics_path = f"{args.results_dir}/task2_{args.gnn_type}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f" Metrics saved → {metrics_path}")

    plots_dir = f"{args.results_dir}/plots"
    Path(plots_dir).mkdir(exist_ok=True)
    plot_training_curves(history, f"{plots_dir}/task2_{args.gnn_type}_training_curves.png")

    return metrics


def parse_args():
    p = argparse.ArgumentParser(description="Task 2: GNN Genre Classification")

    p.add_argument("--splits_dir",    default="data/splits")
    p.add_argument("--graph_dir",     default="data/processed/graphs")
    p.add_argument("--processed_dir", default="data/processed/audio_features")
    p.add_argument("--results_dir",   default="results")

    p.add_argument("--gnn_type",       default="GraphSAGE", choices=["GraphSAGE", "GAT"])
    p.add_argument("--hidden_dim",     type=int,   default=256)
    p.add_argument("--gnn_output_dim", type=int,   default=256)
    p.add_argument("--num_layers",     type=int,   default=3)
    p.add_argument("--dropout",        type=float, default=0.3)

    p.add_argument("--epochs",        type=int,   default=30)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--sim_threshold", type=float, default=0.7)
    p.add_argument("--num_workers",   type=int,   default=4)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metrics = train(args)