import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from dataset import make_datasets
from train_gnn import default_cfg, train

DATA_KEYS = ("graph", "n_nodes", "feat", "with_std", "edge_mode", "k", "tau")


def sweep_configs():
    runs = []

    for mode in ("temporal", "knn", "threshold", "full"):
        runs.append((f"graph_{mode}", dict(edge_mode=mode)))
    runs.append(("graph_chord", dict(graph="chord")))

    for n in (6, 15, 30, 60):
        runs.append((f"nodes_{n}", dict(n_nodes=n)))

    for kind in ("GraphSAGE", "GAT", "GCN", "GIN"):
        runs.append((f"encoder_{kind}", dict(kind=kind)))

    for layers in (2, 3, 4):
        runs.append((f"layers_{layers}", dict(layers=layers)))

    for feat, std in (("mel", False), ("chroma", False), ("mel+chroma", False), ("mel+chroma", True)):
        runs.append((f"feat_{feat.replace('+','')}{'_std' if std else ''}",
                     dict(feat=feat, with_std=std)))

    runs.append(("readout_meanmax", dict(readout="mean+max")))

    seen, unique = set(), []
    for name, over in runs:
        key = tuple(sorted(over.items()))
        if key in seen:
            continue
        seen.add(key)
        unique.append((name, over))
    return unique


def run_sweep(labels="multi", top_k=20, epochs=60, results_dir="results/experiments"):
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    configs = [(n, default_cfg(labels=labels, top_k=top_k, epochs=epochs,
                               results_dir=results_dir, run_name=n, **o))
               for n, o in sweep_configs()]
    configs.sort(key=lambda c: tuple(str(c[1][k]) for k in DATA_KEYS))

    results, cache_key, datasets, vocab = [], None, None, None
    for i, (name, cfg) in enumerate(configs, 1):
        key = tuple(cfg[k] for k in DATA_KEYS)
        if key != cache_key:
            datasets, vocab = make_datasets(
                labels=cfg["labels"], top_k=cfg["top_k"], graph=cfg["graph"],
                n_nodes=cfg["n_nodes"], feat=cfg["feat"], with_std=cfg["with_std"],
                edge_mode=cfg["edge_mode"], k=cfg["k"], tau=cfg["tau"])
            cache_key = key
        print(f"\n[{i}/{len(configs)}] {name}", flush=True)
        res, _ = train(cfg, datasets, vocab, verbose=True, save=True)
        results.append({"name": name, **res["test"], "params": res["params"],
                        "nodes": int(datasets["training"][0].num_nodes),
                        "edges": int(datasets["training"][0].edge_index.shape[1]),
                        "epochs_run": res["epochs_run"], "config": res["config"]})
        json.dump(results, open(f"{results_dir}/sweep.json", "w"), indent=2)
    return results


def sweep_table(path="results/experiments/sweep.json"):
    rows = json.load(open(path))
    groups = {}
    for r in rows:
        groups.setdefault(r["name"].split("_")[0], []).append(r)
    lines = []
    for g, items in groups.items():
        lines.append(f"\n### {g}\n")
        lines.append("| variant | nodes | edges | params | Macro-F1 | Micro-F1 | AUC-PR |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in sorted(items, key=lambda x: -x["auc_pr"]):
            lines.append(f"| {r['name'].split('_',1)[1]} | {r['nodes']} | {r['edges']} | "
                         f"{r['params']:,} | {r['macro_f1']:.4f} | {r['micro_f1']:.4f} | "
                         f"{r['auc_pr']:.4f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--labels", default="multi")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--table", action="store_true")
    a = p.parse_args()
    if a.table:
        print(sweep_table())
    else:
        run_sweep(labels=a.labels, epochs=a.epochs)
        print(sweep_table())
