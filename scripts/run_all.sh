#!/usr/bin/env bash
set -e
PY=${PY:-python}

$PY src/labels.py
$PY scripts/extract_features.py 30
$PY src/export_graphs.py 24
$PY src/visualize_graphs.py
$PY src/cnn_baseline.py --labels multi --epochs 12
$PY src/train_gnn.py --run_name gnn_main
$PY src/evaluate_gnn.py --checkpoint results/gnn_main.pt
$PY src/experiments.py --labels multi --epochs 60
