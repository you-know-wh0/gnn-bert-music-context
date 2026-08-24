import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(__file__))
from dataset import make_datasets, make_loaders
from gnn_model import GNNClassifier
from metrics import multilabel_metrics, random_baseline, singlelabel_metrics, tune_thresholds

PLOTS = "plots"


@torch.no_grad()
def predict(model, loader, device, multi):
    model.eval()
    probs, ys, embeds = [], [], []
    for batch in loader:
        batch = batch.to(device)
        logits, g = model(batch.x, batch.edge_index, batch.batch)
        probs.append((torch.sigmoid(logits) if multi else logits.softmax(1)).cpu().numpy())
        ys.append((batch.y if multi else batch.y.view(-1)).cpu().numpy())
        embeds.append(g.cpu().numpy())
    return np.concatenate(probs), np.concatenate(ys), np.concatenate(embeds)


def plot_curves(history, out):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ep = [h["epoch"] for h in history]
    axes[0].plot(ep, [h["train_loss"] for h in history], label="train")
    axes[0].plot(ep, [h["val_loss"] for h in history], label="val")
    axes[0].set_title("BCE loss")
    axes[1].plot(ep, [h["train_macro_f1"] for h in history], label="train")
    axes[1].plot(ep, [h["val_macro_f1"] for h in history], label="val")
    axes[1].set_title("Macro-F1")
    axes[2].plot(ep, [h["val_auc_pr"] for h in history], color="darkgreen", label="val")
    axes[2].set_title("Validation AUC-PR")
    for ax in axes:
        ax.set_xlabel("epoch"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("GNN training curves", fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def plot_per_class(vocab, gnn_ap, cnn_ap, support, out):
    pad = lambda a: (list(a) + [None] * len(vocab))[:len(vocab)] if a else None
    gnn_ap, cnn_ap = pad(gnn_ap), pad(cnn_ap)
    order = np.argsort(-np.array(support))
    x = np.arange(len(order)); w = 0.4
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w / 2, [gnn_ap[i] or 0.0 for i in order], w, label="GNN", color="steelblue")
    if cnn_ap:
        ax.bar(x + w / 2, [cnn_ap[i] or 0.0 for i in order], w, label="CNN", color="indianred")
    ax.set_xticks(x); ax.set_xticklabels([vocab[i] for i in order], rotation=45, ha="right")
    ax.set_ylabel("AUC-PR"); ax.legend(); ax.grid(alpha=0.3, axis="y")
    ax.set_title("Per-genre AUC-PR (genres ordered by test-set frequency)")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def plot_tsne(embeds, y, vocab, out, multi, max_points=3000):
    idx = np.random.default_rng(0).choice(len(embeds), min(max_points, len(embeds)), replace=False)
    proj = TSNE(2, perplexity=30, random_state=0, init="pca").fit_transform(embeds[idx])
    label = y[idx].argmax(1) if multi else y[idx].astype(int)
    fig, ax = plt.subplots(figsize=(11, 8))
    cmap = plt.get_cmap("tab20", len(vocab))
    for k in range(len(vocab)):
        m = label == k
        if m.sum():
            ax.scatter(proj[m, 0], proj[m, 1], s=8, color=cmap(k), label=vocab[k], alpha=0.7,
                       edgecolors="none")
    ax.legend(fontsize=7, ncol=2, markerscale=2, loc="best")
    ax.set_title("t-SNE of GNN graph embeddings g, coloured by primary genre")
    ax.axis("off")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def comparison_table(rows):
    head = "| Model | Macro-F1 | Micro-F1 | AUC-PR |\n|---|---|---|---|"
    body = "\n".join(f"| {n} | {m['macro_f1']:.4f} | {m['micro_f1']:.4f} | {m['auc_pr']:.4f} |"
                     for n, m in rows)
    return head + "\n" + body


def main(checkpoint, results_dir="results", skip_tsne=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(PLOTS).mkdir(exist_ok=True)
    ckpt = torch.load(checkpoint, weights_only=False, map_location=device)
    cfg, vocab = ckpt["config"], ckpt["vocab"]
    multi = cfg["labels"] == "multi"

    datasets, _ = make_datasets(
        labels=cfg["labels"], top_k=cfg["top_k"], graph=cfg["graph"], n_nodes=cfg["n_nodes"],
        feat=cfg["feat"], with_std=cfg["with_std"], edge_mode=cfg["edge_mode"],
        k=cfg["k"], tau=cfg["tau"])
    loaders = make_loaders(datasets, cfg["batch_size"])

    model = GNNClassifier(datasets["training"][0].x.shape[1], len(vocab), cfg["hidden"],
                          cfg["out_dim"], cfg["layers"], cfg["dropout"], cfg["kind"],
                          cfg["readout"]).to(device)
    model.load_state_dict(ckpt["model_state"])

    val_p, val_y, _ = predict(model, loaders["validation"], device, multi)
    th = tune_thresholds(val_y, val_p) if multi else None
    test_p, test_y, embeds = predict(model, loaders["test"], device, multi)
    gnn = multilabel_metrics(test_y, test_p, th) if multi else singlelabel_metrics(test_y.astype(int), test_p)

    rows = [("Random (label-prior)", random_baseline(test_y if multi else test_y.astype(int),
                                                     cfg["labels"], n_classes=len(vocab)))]
    cnn_path = f"{results_dir}/cnn_baseline_{cfg['labels']}.json"
    cnn_ap = None
    if os.path.exists(cnn_path):
        cnn = json.load(open(cnn_path))
        rows.append(("CNN mel-spectrogram (B2, Person 1 arch.)", cnn["test"]))
        cnn_ap = cnn["per_class_ap"]
    rows.append((f"GNN {cfg['kind']} ({cfg['graph']} graph)", gnn))

    table = comparison_table(rows)
    print("\n" + table + "\n")

    run = Path(checkpoint).stem
    hist_path = f"{results_dir}/{run}.json"
    if os.path.exists(hist_path):
        plot_curves(json.load(open(hist_path))["history"], f"{PLOTS}/{run}_training_curves.png")
    support = test_y.sum(0) if multi else np.bincount(test_y.astype(int), minlength=len(vocab))
    plot_per_class(vocab, gnn["per_class_ap"], cnn_ap, support, f"{PLOTS}/{run}_per_genre_ap.png")
    if not skip_tsne:
        plot_tsne(embeds, test_y, vocab, f"{PLOTS}/{run}_tsne.png", multi)

    out = {"checkpoint": checkpoint, "config": cfg, "vocab": vocab,
           "comparison": {n: {k: v for k, v in m.items() if k != "per_class_ap"} for n, m in rows},
           "gnn_per_class_ap": dict(zip(vocab, gnn["per_class_ap"])),
           "markdown_table": table}
    json.dump(out, open(f"{results_dir}/metrics.json", "w"), indent=2)
    print(f"saved {results_dir}/metrics.json and plots in {PLOTS}/")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--results_dir", default="results")
    p.add_argument("--skip_tsne", action="store_true")
    a = p.parse_args()
    main(a.checkpoint, a.results_dir, a.skip_tsne)
