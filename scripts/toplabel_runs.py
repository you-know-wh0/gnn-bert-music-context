import sys
sys.path.insert(0, "src")
from dataset import make_datasets
from train_gnn import default_cfg, train
import cnn_baseline

cfg = default_cfg(labels="top", n_nodes=60, with_std=True, layers=2, kind="GAT",
                  edge_mode="knn", epochs=80, results_dir="results/final",
                  run_name="best_gat_knn_top")
ds, vocab = make_datasets(**{k: cfg[k] for k in
    ("labels", "top_k", "graph", "n_nodes", "feat", "with_std",
     "edge_mode", "k", "tau", "chord_smooth")})
print("=== GNN (16-class genre_top) ===", flush=True)
train(cfg, ds, vocab, verbose=True, save=True)
print("\n=== CNN (16-class genre_top) ===", flush=True)
cnn_baseline.main(labels="top", epochs=12)
