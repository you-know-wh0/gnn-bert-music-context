import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix,
    precision_recall_curve, average_precision_score,
    classification_report,
)
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
from dataset   import get_dataloaders
from gnn_model import GNNClassifier



@torch.no_grad()
def collect_predictions(model, loader, device):
    """
    Returns all_preds, all_labels, all_probs (softmax), all_embeddings.
    """
    model.eval()
    all_preds, all_labels, all_probs, all_embeds = [], [], [], []

    for batch in loader:
        batch  = batch.to(device)
        logits, g = model(batch.x, batch.edge_index, batch.batch)
        probs  = F.softmax(logits, dim=1)

        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(batch.y.squeeze().cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_embeds.extend(g.cpu().numpy())

    return (np.array(all_preds), np.array(all_labels),
            np.array(all_probs),  np.array(all_embeds))




def compute_auc_pr(labels, probs, num_classes):
    """
    Compute mean AUC-PR using one-vs-rest binarisation.
    """
    bin_labels = label_binarize(labels, classes=list(range(num_classes)))
    if num_classes == 2:
        bin_labels = np.hstack([1 - bin_labels, bin_labels])

    per_class_ap = []
    for k in range(num_classes):
        if bin_labels[:, k].sum() == 0:
            continue
        ap = average_precision_score(bin_labels[:, k], probs[:, k])
        per_class_ap.append(ap)

    return float(np.mean(per_class_ap)), per_class_ap




def plot_confusion_matrix(labels, preds, class_names, save_path):
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.3,
    )
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (normalised)")
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix → {save_path}")


def plot_auc_pr(labels, probs, num_classes, class_names, save_path):
    bin_labels = label_binarize(labels, classes=list(range(num_classes)))
    if num_classes == 2:
        bin_labels = np.hstack([1 - bin_labels, bin_labels])

    fig, ax = plt.subplots(figsize=(10, 8))
    for k in range(num_classes):
        if bin_labels[:, k].sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(bin_labels[:, k], probs[:, k])
        ap = average_precision_score(bin_labels[:, k], probs[:, k])
        ax.plot(rec, prec, lw=1.2, label=f"{class_names[k]} (AP={ap:.2f})")

    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curves (one-vs-rest per genre)")
    ax.legend(loc="lower left", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  AUC-PR curves → {save_path}")


def plot_tsne(embeddings, labels, class_names, save_path, title="t-SNE of GNN embeddings"):

    print("  Running t-SNE (this may take ~1 min for large sets)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    proj = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(12, 9))
    palette = plt.cm.get_cmap("tab20", len(class_names))

    for k, genre in enumerate(class_names):
        mask = labels == k
        ax.scatter(
            proj[mask, 0], proj[mask, 1],
            c=[palette(k)], label=genre,
            s=10, alpha=0.7, edgecolors="none",
        )

    ax.legend(loc="best", fontsize=7, markerscale=2, ncol=2)
    ax.set_title(title); ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  t-SNE plot → {save_path}")



def print_comparison_table(
    gnn_macro: float, gnn_micro: float, gnn_auc: float,
    cnn_macro: float = None, cnn_micro: float = None, cnn_auc: float = None,
    idx_to_genre: dict = None,
):
    """
    Print a comparison table.
    CNN numbers come from Person 1's results or can be passed as arguments.
    """
    num_classes = len(idx_to_genre) if idx_to_genre else 16
    random_macro = 1.0 / num_classes  
    random_auc   = 0.5               

    rows = [
        ("Random (majority/uniform)", f"{random_macro:.4f}", "—",            f"{random_auc:.4f}"),
        ("CNN  (mel-spectrogram, P1)",
            f"{cnn_macro:.4f}" if cnn_macro is not None else "— (run Person 1)",
            f"{cnn_micro:.4f}" if cnn_micro is not None else "—",
            f"{cnn_auc:.4f}"   if cnn_auc   is not None else "—"),
        ("GNN-only  [ours, Task 2]",  f"{gnn_macro:.4f}", f"{gnn_micro:.4f}", f"{gnn_auc:.4f}"),
    ]

    col_w = [34, 12, 12, 10]
    header = f"{'Model':<{col_w[0]}} {'Macro-F1':>{col_w[1]}} {'Micro-F1':>{col_w[2]}} {'AUC-PR':>{col_w[3]}}"
    sep    = "─" * sum(col_w)

    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in rows:
        print(f"{r[0]:<{col_w[0]}} {r[1]:>{col_w[1]}} {r[2]:>{col_w[2]}} {r[3]:>{col_w[3]}}")
    print(f"{sep}\n")




def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plots_dir = Path(args.results_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt         = torch.load(args.checkpoint, weights_only=False, map_location=device)
    genre_to_idx = ckpt["genre_to_idx"]
    idx_to_genre = {int(i): g for g, i in genre_to_idx.items()}
    class_names  = [idx_to_genre[i] for i in range(len(idx_to_genre))]
    num_classes  = len(genre_to_idx)
    saved_args   = ckpt.get("args", {})
    gnn_type     = saved_args.get("gnn_type", "GraphSAGE")

    _, _, test_loader, _, _ = get_dataloaders(
        splits_dir    = args.splits_dir,
        graph_dir     = args.graph_dir,
        processed_dir = args.processed_dir,
        batch_size    = args.batch_size,
        num_workers   = args.num_workers,
    )

    model = GNNClassifier(
        input_dim      = 140,
        hidden_dim     = saved_args.get("hidden_dim",     256),
        gnn_output_dim = saved_args.get("gnn_output_dim", 256),
        num_classes    = num_classes,
        num_layers     = saved_args.get("num_layers",     3),
        dropout        = saved_args.get("dropout",        0.3),
        gnn_type       = gnn_type,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Model: {gnn_type}  |  Test tracks: {len(test_loader.dataset)}\n")

    preds, labels, probs, embeds = collect_predictions(model, test_loader, device)

    macro_f1 = f1_score(labels, preds, average="macro",  zero_division=0)
    micro_f1 = f1_score(labels, preds, average="micro",  zero_division=0)
    accuracy = accuracy_score(labels, preds)
    mean_auc, per_class_ap = compute_auc_pr(labels, probs, num_classes)

    print(f" Macro-F1  : {macro_f1:.4f}")
    print(f" Micro-F1  : {micro_f1:.4f}")
    print(f" Accuracy  : {accuracy:.4f}")
    print(f" Mean AUC-PR: {mean_auc:.4f}\n")
    print(classification_report(labels, preds, target_names=class_names, zero_division=0))

    # ── Comparison table ─────────────────────────────────────────────────
    # Try to load CNN baseline results from Person 1
    cnn_macro = cnn_micro = cnn_auc = None
    cnn_results_path = Path(args.results_dir) / "cnn_baseline_metrics.json"
    if cnn_results_path.exists():
        with open(cnn_results_path) as f:
            cnn_res = json.load(f)
        cnn_macro = cnn_res.get("test_macro_f1")
        cnn_micro = cnn_res.get("test_micro_f1")
        cnn_auc   = cnn_res.get("test_auc_pr")

    print_comparison_table(macro_f1, micro_f1, mean_auc,
                           cnn_macro, cnn_micro, cnn_auc, idx_to_genre)

    plot_confusion_matrix(
        labels, preds, class_names,
        save_path=str(plots_dir / f"task2_{gnn_type}_confusion.png")
    )
    plot_auc_pr(
        labels, probs, num_classes, class_names,
        save_path=str(plots_dir / f"task2_{gnn_type}_auc_pr.png")
    )
    if not args.skip_tsne:
        plot_tsne(
            embeds, labels, class_names,
            save_path=str(plots_dir / f"task2_{gnn_type}_tsne.png"),
            title=f"t-SNE: GNN [{gnn_type}] embeddings by genre",
        )

    out = {
        "model"         : gnn_type,
        "test_macro_f1" : macro_f1,
        "test_micro_f1" : micro_f1,
        "test_accuracy" : accuracy,
        "test_auc_pr"   : mean_auc,
        "per_class_auc" : {class_names[i]: float(v) for i, v in enumerate(per_class_ap)},
    }
    out_path = Path(args.results_dir) / f"task2_{gnn_type}_test_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n Results saved → {out_path}")



def parse_args():
    p = argparse.ArgumentParser(description="Task 2: GNN Evaluation")
    p.add_argument("--checkpoint",    required=True, help="Path to .pt checkpoint")
    p.add_argument("--splits_dir",    default="data/splits")
    p.add_argument("--graph_dir",     default="data/processed/graphs")
    p.add_argument("--processed_dir", default="data/processed/audio_features")
    p.add_argument("--results_dir",   default="results")
    p.add_argument("--batch_size",    type=int, default=64)
    p.add_argument("--num_workers",   type=int, default=4)
    p.add_argument("--skip_tsne",     action="store_true", help="Skip t-SNE (slow)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)