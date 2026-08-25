# Person 3 Handover — Graph Construction + GNN

**Role:** music structure graphs, GNN encoders, GNN evaluation vs. the CNN baseline.
Consumes Person 1's audio features; produces the graph embedding `g` that Person 4 fuses with BERT.

---

## 1. What changed in the feature pipeline (read this first)

Person 1's original CPU pipeline could not be run to completion on this machine:
`librosa.feature.chroma_stft` **segfaults** (its `estimate_tuning` → numba path), which
killed multiprocessing workers silently. `librosa.stft`, `melspectrogram`, and
`librosa.filters.*` are all fine — only `chroma_stft` crashes.

Features are now produced by `scripts/extract_features_gpu.py`, which keeps the same
recipe but runs the spectral stage on the GPU:

```
mp3 --(soundfile, CPU)--> waveform --(GPU)--> resample 22,050 Hz
    -> STFT (n_fft 2048, hop 512)
    -> log-mel  (128 bins, librosa mel filterbank, power_to_db ref=max)
    -> chroma   (12 bins, librosa chroma filterbank, per-frame inf-norm)
    -> per-track z-score -> 6 segments x 215 frames
```

Chroma is computed as `chroma_filterbank @ |STFT|^2` then inf-normalised — librosa's own
formula minus the tuning estimation that crashes.

**Agreement with librosa** (on files where librosa runs): mel r = 0.995, chroma r = 0.9999.
The residual is the resampler (torchaudio vs. soxr).

**Runtime: 1.5 min for all 25,000 tracks** (272 files/s) vs. ~35 min projected for the CPU path.

### Storage changed from 25,000 `.npz` files to two memmaps

| | path | shape | dtype | size |
|---|---|---|---|---|
| mel | `data/processed/store/mel_f16.npy` | (25000, 6, 128, 215) | float16 | 8.2 GB |
| chroma | `data/processed/store/chroma_f16.npy` | (25000, 6, 12, 215) | float16 | 0.8 GB |
| ids | `data/processed/store/track_ids.json` | 25000 | str | — |

9 GB total instead of 72 GB, and far faster to read during training.

```python
import features_store
mel, chroma = features_store.get("000002")     # (6,128,215), (6,12,215) float32
mel, chroma = features_store.stitch("000002")  # (128,1290), (12,1290) time-concatenated
features_store.usable_ids()                    # excludes the 20 undecodable tracks
```

20 of 25,000 files failed to decode (listed in `results/extraction_failures.json`) —
these include FMA's known-corrupt `099134`, `108925`, `133297`. They are excluded
everywhere via `usable_ids()`.

---

## 2. Labels

Two formulations, selected by one flag, both from the same splits:

| mode | source | classes | loss | used for |
|---|---|---|---|---|
| `multi` (primary) | `track.genres_all`, top-20 by training frequency | 20 | BCE | project's multi-label formulation, Macro/Micro-F1, AUC-PR |
| `top` | `track.genre_top` | 16 | cross-entropy | matches Person 1's CNN and Person 4's handover |

`multi` covers 18,294 of 19,922 training tracks; the rest carry none of the top-20 genres
and are dropped. Split JSONs gained additive `genres_all` and `artist_id` keys — the
existing `genre` and `subset` keys are untouched, so Person 1/2/4 code still works.

**Artist leakage: 0.** Verified empirically, not assumed:
`train∩val = 0, train∩test = 0, val∩test = 0` distinct artists (`labels.check_artist_leakage`).

---

## 3. Graph construction (`src/graph_builder.py`)

### Segment graph (primary)
Nodes are time windows; the 6×215-frame layout is re-cut into any node count, so node
granularity is a free parameter — no audio is re-decoded.

- **Node features:** window-pooled mel and/or chroma, mean or mean+std → 140-d or 280-d
- **Edges** (`edge_mode`): `temporal` (i↔i+1) · `knn` (temporal + k nearest by cosine, default k=3) ·
  `threshold` (temporal + cos > τ) · `full`
- **Edge weights** in `edge_attr` (1.0 for temporal, cosine for similarity edges)
- Node features are standardised with a scaler **fit on the training split only**

### Chord-transition graph
Per-frame chroma is matched against 24 major/minor triad templates; nodes are the chords
that occupy ≥3 frames, edges are observed transitions weighted by count.
Node features = triad one-hot (24) + mean chroma (12) + duration share (1) = 37-d.
`smooth` applies a median filter (21 frames ≈ 0.5 s) to the chord sequence.

---

## 4. Interface for Person 4 (fusion)

```python
from gnn_model import GNNClassifier
import torch

ckpt  = torch.load("results/final/best_gat_knn.pt", weights_only=False)
cfg   = ckpt["config"]
model = GNNClassifier(280, len(ckpt["vocab"]), cfg["hidden"], cfg["out_dim"],
                      cfg["layers"], cfg["dropout"], cfg["kind"], cfg["readout"])
model.load_state_dict(ckpt["model_state"])

g = model.encode(batch)   # -> FloatTensor [batch_size, 256]
```

**Contract:** `encode(batch)` takes a PyG `Batch` and returns `[B, 256]`. That is the `g`
in `z = Fusion(g, t)`. The checkpoint also stores the tuned per-class `thresholds`.

To get matching graph batches:

```python
from dataset import make_datasets, make_loaders
datasets, vocab = make_datasets(labels="multi", n_nodes=60, with_std=True, edge_mode="knn")
loaders = make_loaders(datasets, batch_size=256)
```

Every `Data` object carries `.track_id`, so graphs align with BERT text by track id.

---

## 5. Results (FMA-medium test split, multi-label top-20 genres)

| Model | Macro-F1 | Micro-F1 | AUC-PR |
|---|---|---|---|
| Random (label-prior) | 0.0977 | 0.2599 | 0.0821 |
| CNN mel-spectrogram (B2, Person 1 architecture) | **0.4061** | **0.5032** | **0.4260** |
| GNN GAT, segment graph (Task 2, ours) | 0.3544 | 0.4410 | 0.3673 |

**The CNN beats the GNN, and that is the honest result.** The GNN sees each 0.5 s window
compressed to a 280-d pooled descriptor; the CNN sees the full 128×215 spectrogram per
segment and averages six per-segment predictions. The graph adds structural information
but discards the time–frequency detail the CNN exploits. Both were trained on identical
splits, labels, and thresholds.

Best GNN configuration: **GAT, 60 nodes, mel+chroma mean+std (280-d), 2 layers, kNN edges (k=3)**,
208k parameters, ~35 s to train.

---

## 6. Files

| file | role |
|---|---|
| `src/features_store.py` | memmap accessor for mel/chroma |
| `src/labels.py` | label vocab, split augmentation, artist-leakage check |
| `src/graph_builder.py` | segment + chord graph construction |
| `src/gnn_model.py` | GraphSAGE / GAT / GCN / GIN encoders + classifier |
| `src/dataset.py` | cached graph datasets and loaders |
| `src/train_gnn.py` | training with threshold tuning and early stopping |
| `src/evaluate_gnn.py` | metrics, comparison table, t-SNE, per-genre AP |
| `src/experiments.py` | graph/encoder/depth/feature sweep |
| `src/cnn_baseline.py` | trains Person 1's CNN (replicated for the comparison row only) |
| `src/visualize_graphs.py` | graph, self-similarity and statistics figures |
| `src/export_graphs.py` | 24 example graphs as `.pt` + `.json` |
| `scripts/extract_features_gpu.py` | GPU feature extraction |
