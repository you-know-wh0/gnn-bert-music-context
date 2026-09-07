"""Person 4 -- ablation experiments: BERT-only vs GNN-only vs early
concatenation vs cross-attention, on the same MusicCaps split, reported
as one comparison table + plot (project rubric: "comparison between
BERT-only / GNN-only / early concatenation / cross-attention")."""
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from fusion_interface import make_dummy_datasets
from train_fusion import default_cfg, train

RESULTS_DIR = "results/fusion"
PLOTS = "plots"
MODES = ("bert_only", "gnn_only", "concat", "cross_attention")


def run_ablation(datasets=None, vocab=None, bert_checkpoint=None, gnn_checkpoint=None,
                 epochs=15, results_dir=RESULTS_DIR):
    """Train all four fusion modes on the *same* data split so the
    comparison is apples-to-apples, then return one row of test metrics
    per mode. Pass in real datasets/vocab (see fusion_interface.py's
    contract) once the paired dataset exists; defaults to a synthetic
    smoke test otherwise so the ablation code itself is runnable today."""
    if datasets is None:
        print("[evaluate_fusion] no datasets passed in -- running ablation on "
              "synthetic smoke-test data. Pass real datasets/vocab once available.")
        datasets, vocab = make_dummy_datasets()
    rows = []
    for mode in MODES:
        cfg = default_cfg(mode=mode, epochs=epochs, results_dir=results_dir,
                          run_name=f"fusion_{mode}", bert_checkpoint=bert_checkpoint,
                          gnn_checkpoint=(gnn_checkpoint if mode != "bert_only" else None),
                          freeze_gnn=(gnn_checkpoint is not None))
        print(f"\n=== ablation: {mode} ===", flush=True)
        res, _ = train(cfg, datasets, vocab, verbose=True, save=True)
        rows.append((mode, res["test"]))
    return rows


def comparison_table(rows):
    head = "| Model | Macro-F1 | Micro-F1 | AUC-PR |\n|---|---|---|---|"
    body = "\n".join(f"| {n} | {m['macro_f1']:.4f} | {m['micro_f1']:.4f} | {m['auc_pr']:.4f} |"
                     for n, m in rows)
    return head + "\n" + body


def plot_ablation(rows, out=None):
    Path(PLOTS).mkdir(exist_ok=True)
    out = out or f"{PLOTS}/fusion_ablation.png"
    names = [n for n, _ in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for metric, color in (("macro_f1", "steelblue"), ("micro_f1", "indianred"),
                          ("auc_pr", "seagreen")):
        ax.plot(names, [m[metric] for _, m in rows], marker="o", label=metric, color=color)
    ax.set_ylabel("score"); ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend()
    ax.set_title("Fusion ablation: BERT-only vs GNN-only vs concat vs cross-attention")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def main(datasets=None, vocab=None, bert_checkpoint=None, gnn_checkpoint=None,
        epochs=15, results_dir=RESULTS_DIR):
    rows = run_ablation(datasets, vocab, bert_checkpoint, gnn_checkpoint, epochs, results_dir)
    table = comparison_table(rows)
    print("\n" + table + "\n")
    plot_ablation(rows)
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    json.dump({"comparison": dict(rows), "markdown_table": table},
              open(f"{results_dir}/ablation_summary.json", "w"), indent=2)
    print(f"saved {results_dir}/ablation_summary.json and {PLOTS}/fusion_ablation.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bert_checkpoint", default=None)
    p.add_argument("--gnn_checkpoint", default=None)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--results_dir", default=RESULTS_DIR)
    a = p.parse_args()
    main(a.bert_checkpoint, a.gnn_checkpoint, a.epochs, a.results_dir)