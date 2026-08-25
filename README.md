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

### Feature pipeline note

`librosa.feature.chroma_stft` segfaults on this environment, so features are produced by
`scripts/extract_features_gpu.py`, which runs the spectral stage on GPU using librosa's own
mel/chroma filterbanks (mel r = 0.995, chroma r = 0.9999 vs. librosa; 1.5 min for 25k tracks).
Output is two float16 memmaps under `data/processed/store/` (9 GB) rather than 25,000 `.npz`
files (72 GB); read it with `src/features_store.py`.

### Running

```bash
pip install -r requirements.txt
python scripts/extract_features_gpu.py
python src/labels.py
python scripts/final_runs.py
python src/evaluate_gnn.py --checkpoint results/final/best_gat_knn.pt
python src/experiments.py --labels multi
```
