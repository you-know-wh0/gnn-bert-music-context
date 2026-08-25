import json, os, sys
sys.path.insert(0, "src")
from dataset import make_datasets
from train_gnn import default_cfg, train

BEST = dict(n_nodes=60, with_std=True, layers=2, feat="mel+chroma")
RUNS = [
    ("best_gat_knn",     dict(kind="GAT", edge_mode="knn", **BEST)),
    ("best_sage_knn",    dict(kind="GraphSAGE", edge_mode="knn", **BEST)),
    ("best_gat_full",    dict(kind="GAT", edge_mode="full", **BEST)),
    ("best_gat_meanmax", dict(kind="GAT", edge_mode="knn", readout="mean+max", **BEST)),
    ("chord_smoothed",   dict(graph="chord", chord_smooth=21, kind="GAT", layers=2)),
]

out = []
for name, over in RUNS:
    cfg = default_cfg(labels="multi", epochs=80, results_dir="results/final",
                      run_name=name, **over)
    ds, vocab = make_datasets(**{k: cfg[k] for k in
        ("labels","top_k","graph","n_nodes","feat","with_std","edge_mode","k","tau","chord_smooth")})
    print(f"\n=== {name} ===", flush=True)
    res, _ = train(cfg, ds, vocab, verbose=True, save=True)
    out.append({"name": name, **res["test"], "params": res["params"],
                "nodes": int(ds["training"][0].num_nodes),
                "edges": int(ds["training"][0].edge_index.shape[1])})
    json.dump(out, open("results/final/final_runs.json", "w"), indent=2)

print("\n| run | nodes | edges | params | Macro-F1 | Micro-F1 | AUC-PR |")
print("|---|---|---|---|---|---|---|")
for r in sorted(out, key=lambda x: -x["auc_pr"]):
    print(f"| {r['name']} | {r['nodes']} | {r['edges']} | {r['params']:,} | "
          f"{r['macro_f1']:.4f} | {r['micro_f1']:.4f} | {r['auc_pr']:.4f} |")
