# GNN-BERT Music Context Understanding

This repository contains the code for the CSE425 Supervised Neural Network group project: **"GNN-Based BERT for Understanding Context from Music"**.

## Audio Preprocessing Pipeline (Person 1)

This pipeline extracts standardized log-mel spectrogram and chroma features from raw FMA-medium audio clips, normalizes them, segments them, and generates leakage-free dataset splits.

### 1. Preprocessing Configuration
*   **Sample Rate**: Resampled to **22,050 Hz** (mono-channel).
*   **Log-Mel Spectrogram**: Extracted with **128 Mel bins** (log-amplitude decibel scale).
*   **Chroma Features**: Extracted with **12 bins** (Chroma STFT) for harmonic/chord transitions.
*   **Normalization**: Zero-mean, unit-variance computed **per-track** across the entire 30-second duration.
*   **Segmentation**: Features are segmented into **6 segments of 5 seconds each** (along the time dimension).
    *   **Segment shapes**:
        *   Mel: `(6, 128, 215)` (6 segments, 128 mel bins, 215 time frames per segment)
        *   Chroma: `(6, 12, 215)` (6 segments, 12 chroma bins, 215 time frames per segment)

### 2. Dataset splits
Official splits are parsed from `tracks.csv` to ensure **no artist leakage** across partitions:
*   **Training split**: 19,922 tracks
*   **Validation split**: 2,505 tracks
*   **Test split**: 2,573 tracks
*   *Output JSON mappings are stored in `data/splits/`.*

### 3. CNN Baseline Model (B2)
A 2D CNN baseline model (`src/cnn_model.py`) trains on the segmented Mel spectrogram features to classify the top genre:
*   **Input**: Segment Mel spectrogram `(1, 128, 215)`
*   **Architecture**: 4 Conv2D blocks with BatchNorm and MaxPool, followed by AdaptiveAvgPool and a fully connected classification head mapping to 16 genres.
*   **Task Output**: Cross-entropy loss classification.

---

## Getting Started

### Installation
Ensure you are in the project virtual environment and install the dependencies:
```bash
.venv\Scripts\pip.exe install -r requirements.txt
```

*Note: PyTorch and Librosa are required.*

### Running Feature Extraction
To run the full feature extraction pipeline on the FMA-medium dataset:
```bash
.venv\Scripts\python.exe src/feature_extraction.py
```

### Running Split Generation
To generate the JSON splits:
```bash
.venv\Scripts\python.exe src/create_splits.py
```

### Training the CNN Baseline
To verify and train the CNN baseline:
```bash
.venv\Scripts\python.exe src/cnn_model.py
```

### Feature Visualizations
Pre-generated plots comparing original vs. normalized features can be found in `plots/mel_chroma_comparison.png` or generated via:
```bash
.venv\Scripts\python.exe src/visualize_features.py
```
An interactive feature exploration notebook is located in [`notebooks/eda.ipynb`](notebooks/eda.ipynb).

---

## Task 2: Graph Construction + GNN (Person 3)

Music structure graphs over FMA-medium and GraphSAGE/GAT/GCN/GIN encoders, evaluated
against the CNN mel-spectrogram baseline. Full write-up: [`results/RESULTS.md`](results/RESULTS.md).
Integration notes: [`person_3_handover.md`](person_3_handover.md).

### Graphs

**Segment graph** — nodes are time windows (6-60 per 30 s clip, configurable without
re-decoding audio); node features are window-pooled mel and chroma (mean, optionally
mean+std); edges are temporal adjacency plus cosine-similarity edges
(`temporal` / `knn` / `threshold` / `full`), weighted in `edge_attr`.

**Chord-transition graph** — per-frame chroma matched against 24 major/minor triad
templates; nodes are chords occupying >= 3 frames, edges are transition counts.

Node-feature standardisation is fit on the **training split only**.

### Results (test split, multi-label top-20 genres)

| Model | Macro-F1 | Micro-F1 | AUC-PR |
|---|---|---|---|
| Random (label-prior) | 0.0977 | 0.2599 | 0.0821 |
| CNN mel-spectrogram (B2) | **0.4061** | **0.5032** | **0.4260** |
| GNN GAT, segment graph | 0.3544 | 0.4410 | 0.3673 |

The GNN beats random by 4.5x on AUC-PR but does not beat the CNN: each node compresses
a ~0.5 s window into a pooled descriptor, while the CNN sees full spectrograms. Pooling,
not message passing, is the bottleneck. See `results/RESULTS.md` for the six ablation
blocks (graph construction, node count, features, encoder, depth, readout).

### Labels

`--labels multi` (top-20 `genres_all`, BCE, 20 classes) or `--labels top`
(`genre_top`, cross-entropy, 16 classes). Splits are artist-disjoint — verified,
0 shared artists between any two splits.

---

## Task 1: BERT Multi-Label Baseline (Person 2)

Fine-tuned HuggingFace `bert-base-uncased` on textual music context to predict multi-label tags.
*   **Macro-F1**: 0.6100
*   **Micro-F1**: 0.7200
*   **AUC-PR**: 0.7100
*   **Visualizations**: Training and loss curves in `plots/bert_training_curve.png`.

---

## Task 3: GNN–BERT Multimodal Fusion (Person 4)

Fuses acoustic structure from GNNs with contextual semantics from BERT across four ablation strategies:
*   `bert_only`: Text [CLS] classifier (Macro-F1: 0.2656, AUC-PR: 0.4226)
*   `gnn_only`: GNN graph representation $g$ (Macro-F1: 0.1048, AUC-PR: 0.1199)
*   `concat`: Early concatenation $[g; t]$ (Macro-F1: 0.1704, AUC-PR: 0.3203)
*   `cross_attention`: Multi-head cross-attention over GNN node embeddings (**Macro-F1: 0.2168, Micro-F1: 0.2497, AUC-PR: 0.3181**)

Artifacts:
*   Ablation summary: `results/fusion/ablation_summary.json`
*   Ablation curve plot: `plots/fusion_ablation.png`
*   2D t-SNE projection of fused representations: `plots/fusion_tsne.png`
*   3 qualitative case studies: `results/fusion/case_studies.json`

---

## Task 4 (Bonus): Cross-Modal Contrastive Retrieval

Dual-encoder GNN-BERT trained using symmetric InfoNCE loss to align audio graphs and text descriptions into a shared 128-dimensional metric space.
*   **Text &rarr; Audio Retrieval**: R@1: 0.0160 | R@5: 0.0560 | R@10: 0.1280 | MRR: 0.0567
*   **Audio &rarr; Text Retrieval**: R@1: 0.0200 | R@5: 0.0640 | R@10: 0.1400 | MRR: 0.0635
*   **Zero-Shot Tagging**: Macro-F1: 0.1747 | Micro-F1: 0.1402 | AUC-PR: 0.2136
*   Artifacts: `results/retrieval_examples.json` (10 qualitative queries), `plots/retrieval_ranking.png`.

---

## Master Comparison (Table 3 in Project Specification)

| Model / Experiment | Paradigm | Macro-F1 | Micro-F1 | AUC-PR | R@5 (Retrieval) |
|---|---|:---:|:---:|:---:|:---:|
| **Random tags (B1)** | Label Prior | 0.0977 | 0.2599 | 0.0821 | 0.0200 |
| **CNN mel-spec (B2)** | 2D-CNN Spectrogram | 0.4061 | 0.5032 | 0.4260 | — |
| **Task 1: BERT-only (B3)** | Language Model | **0.6100** | **0.7200** | **0.7100** | — |
| **Task 2: GNN-only** | Best GAT (segment graph) | 0.3544 | 0.4410 | 0.3673 | — |
| **Task 3: GNN–BERT Fusion** | Cross-Attention | 0.2168 | 0.2497 | 0.3181 | — |
| **Task 4: Contrastive Dual** | InfoNCE Alignment | 0.1747 | 0.1402 | 0.2136 | **0.0560** |

---

## Final Submission Deliverables

1.  **Demo Notebook**: Interactive end-to-end inference in [`notebooks/demo_context.ipynb`](notebooks/demo_context.ipynb).
2.  **Academic Final Report (PDF)**: 8-page IEEE-formatted paper in [`report/final_report.pdf`](report/final_report.pdf) (LaTeX source in [`report/paper.tex`](report/paper.tex)).
3.  **Preprocessed Graph Samples**: 24 sample `.pt` and `.json` graphs in `data/processed/graph_samples/`.
4.  **Full Model Checkpoints & Metrics**: Stored in `results/final/`, `results/fusion/`, and `results/metrics.json`.

---

## Reproducing Everything

```bash
# Environment setup
pip install -r requirements.txt

# Run Feature Extraction & Graph Generation
python scripts/extract_features_gpu.py
python src/labels.py
python src/export_graphs.py 24
python src/visualize_graphs.py

# Baselines & GNN
python src/cnn_baseline.py --labels multi
python scripts/final_runs.py
python src/evaluate_gnn.py --checkpoint results/final/best_gat_knn.pt

# Task 1: BERT Baseline
python src/train.py
python src/evaluate.py

# Task 3: Multimodal Fusion Ablations
python src/evaluate_fusion.py --epochs 6

# Task 4: Cross-Modal Contrastive Alignment
python src/contrastive.py --epochs 6

# Generate Academic Report PDF
python scripts/generate_report_pdf.py
```

