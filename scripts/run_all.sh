#!/usr/bin/env bash
set -e
PY=${PY:-python}

$PY scripts/extract_features_gpu.py          # mp3 -> mel/chroma memmaps (GPU)
$PY src/labels.py                            # label vocab + artist-leakage check
$PY src/export_graphs.py 24                  # 24 example .pt/.json graphs
$PY src/visualize_graphs.py                  # graph figures
$PY src/cnn_baseline.py --labels multi       # B2 baseline
$PY scripts/final_runs.py                    # best GNN configurations
$PY src/evaluate_gnn.py --checkpoint results/final/best_gat_knn.pt
$PY src/experiments.py --labels multi        # full ablation sweep
