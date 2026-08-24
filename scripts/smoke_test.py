import json, os, shutil, sys, numpy as np
sys.path.insert(0, "src")

TMP = "data/processed/_smoke"
shutil.rmtree(TMP, ignore_errors=True); os.makedirs(TMP)
rng = np.random.default_rng(0)

ids = []
for name in ("training", "validation", "test"):
    split = json.load(open(f"data/splits/{name}.json"))
    take = list(split)[:600 if name == "training" else 200]
    ids += take
    for tid in take:
        # fake Person-1 format with a weak genre-dependent signal
        bias = (hash(split[tid]["genre"]) % 7) * 0.4
        mel = (rng.standard_normal((6, 128, 215)) + bias).astype(np.float32)
        ch = np.abs(rng.standard_normal((6, 12, 215)) + bias * 0.3).astype(np.float32)
        np.savez(f"{TMP}/{tid}.npz", mel=mel, chroma=ch, sample_rate=22050)
print(f"created {len(ids)} fake npz")

import dataset, cnn_baseline, visualize_graphs, export_graphs
for m in (dataset, cnn_baseline, visualize_graphs, export_graphs):
    m.FEATURE_DIR = TMP
dataset.CACHE_DIR = f"{TMP}/cache"
cnn_baseline.CACHE = f"{TMP}/cache/mel_f16.npy"; cnn_baseline.IDS = f"{TMP}/cache/mel_ids.json"
os.makedirs(f"{TMP}/cache", exist_ok=True)

from train_gnn import default_cfg, train
import evaluate_gnn

for labels, graph in (("multi", "segment"), ("top", "segment"), ("multi", "chord")):
    cfg = default_cfg(labels=labels, graph=graph, epochs=3, patience=99, n_nodes=15,
                      results_dir=f"{TMP}/results", run_name=f"smoke_{labels}_{graph}")
    ds, vocab = dataset.make_datasets(labels=labels, graph=graph, n_nodes=15, workers=4)
    print(f"\n--- {labels}/{graph}: train {len(ds['training'])} graphs, "
          f"x {tuple(ds['training'][0].x.shape)} ---")
    train(cfg, ds, vocab, verbose=True, save=True)

print("\n--- edge modes ---")
for mode in ("temporal", "knn", "threshold", "full"):
    ds, vocab = dataset.make_datasets(graph="segment", n_nodes=15, edge_mode=mode, workers=4)
    print(f"{mode:10s} edges {ds['training'][0].edge_index.shape[1]}")

print("\n--- cnn baseline ---")
cnn_baseline.main(labels="multi", epochs=1, results_dir=f"{TMP}/results", workers=2)

print("\n--- evaluate ---")
evaluate_gnn.PLOTS = f"{TMP}/plots"
evaluate_gnn.main(f"{TMP}/results/smoke_multi_segment.pt", results_dir=f"{TMP}/results")

print("\n--- export + visualize ---")
export_graphs.OUT = f"{TMP}/graph_samples"; export_graphs.export(6, n_nodes=15)
visualize_graphs.PLOTS = f"{TMP}/plots"
ex = visualize_graphs.pick_examples()
visualize_graphs.plot_segment_graphs(ex, n_nodes=15)
visualize_graphs.plot_chord_graphs(ex)
visualize_graphs.plot_similarity_matrices(ex, n_nodes=15)
print("stats:", visualize_graphs.plot_graph_stats(n_tracks=50, n_nodes=15))
print("\nSMOKE TEST PASSED")
