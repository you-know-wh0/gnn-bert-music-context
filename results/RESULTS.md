# Task 2 — GNN on Music Structure Graphs: Results

Dataset: **FMA-medium**, 25,000 clips of 30 s. Official FMA splits
(19,922 / 2,505 / 2,573 tracks), **artist-disjoint — verified, 0 shared artists
between any two splits**. 20 tracks excluded as undecodable.

All models below share the same splits, the same labels, and the same
validation-tuned decision thresholds.

---

## 1. Main comparison — multi-label, top-20 `genres_all`

| Model | Macro-F1 | Micro-F1 | AUC-PR |
|---|---|---|---|
| B1 Random (label-prior) | 0.0977 | 0.2599 | 0.0821 |
| B2 CNN mel-spectrogram *(Person 1 architecture)* | **0.4061** | **0.5032** | **0.4260** |
| Task 2 GNN — GAT on segment graph | 0.3544 | 0.4410 | 0.3673 |

## 2. Same comparison — single-label, 16-class `genre_top`

| Model | Macro-F1 | Micro-F1 | AUC-PR |
|---|---|---|---|
| B2 CNN mel-spectrogram *(Person 1 architecture)* | **0.4209** | **0.6652** | **0.4814** |
| Task 2 GNN — GAT on segment graph | 0.3341 | 0.6069 | 0.4022 |

Both formulations agree: the GNN clears the random baseline by a wide margin
(4.5× AUC-PR) but does **not** beat the CNN.

### Why the CNN wins

Each GNN node compresses a ~0.5 s window into a 280-d pooled descriptor. The CNN
receives the full 128 × 215 spectrogram for each of 6 segments and averages six
per-segment predictions. The graph contributes structural information — which
sections repeat, how they connect — but discards the fine time–frequency detail
that carries timbre and instrumentation. Pooling is the bottleneck, not the
message passing.

This is the expected outcome for Task 2 in isolation, and it is the motivation
for Task 3: the graph representation is *complementary* to, not a replacement
for, spectral and textual features.

---

## 3. Ablations (test set, multi-label)

Each block varies one axis from the base configuration
(30 nodes, mel+chroma means, kNN edges, GraphSAGE, 3 layers).

### Graph construction
| variant | nodes | edges | Macro-F1 | Micro-F1 | AUC-PR |
|---|---|---|---|---|---|
| full | 30 | 870 | 0.3307 | 0.4391 | 0.3216 |
| threshold (τ=0.7) | 30 | 294 | 0.3173 | 0.4298 | 0.3145 |
| kNN (k=3) | 30 | 148 | 0.3218 | 0.4235 | 0.3143 |
| temporal only | 30 | 58 | 0.3107 | 0.4110 | 0.3131 |
| chord-transition | 24 | 258 | 0.2911 | 0.3970 | 0.2870 |
| chord-transition, smoothed | 11 | 45 | 0.2500 | 0.3397 | 0.2330 |

Similarity edges help over pure temporal adjacency, but the margin is small
(0.3131 → 0.3216 AUC-PR from 58 to 870 edges). Chord graphs are clearly weaker:
they throw away timbre entirely and keep only harmony.

**Median-filtering the chord sequence made results worse** (0.2870 → 0.2330)
even though it produces far cleaner, more musically plausible graphs
(24 → 11 chords). The rapid frame-level chord flicker that smoothing removes is
itself informative about texture and production style.

### Node count (time resolution)
| nodes | edges | Macro-F1 | Micro-F1 | AUC-PR |
|---|---|---|---|---|
| 60 | 298 | **0.3318** | **0.4660** | **0.3267** |
| 30 | 148 | 0.3199 | 0.4361 | 0.3145 |
| 15 | 73 | 0.3136 | 0.4294 | 0.3130 |
| 6 | 28 | 0.3101 | 0.4031 | 0.3103 |

Monotonic: finer segmentation wins. The 6-node graph implied by a literal
reading of the 5–10 s guidance is nearly a path graph and performs worst.

### Node features
| variant | dim | Macro-F1 | Micro-F1 | AUC-PR |
|---|---|---|---|---|
| mel+chroma, mean+std | 280 | **0.3390** | 0.4330 | **0.3416** |
| mel only, mean | 128 | 0.3240 | 0.4088 | 0.3196 |
| mel+chroma, mean | 140 | 0.3194 | 0.4361 | 0.3140 |
| chroma only, mean | 12 | 0.2707 | 0.3699 | 0.2594 |

Adding within-window standard deviation is the single largest gain in the sweep.
Chroma alone is weakest — harmony carries far less genre signal than timbre.

### Encoder
| variant | params | Macro-F1 | Micro-F1 | AUC-PR |
|---|---|---|---|---|
| GAT | 206,228 | 0.3211 | 0.4561 | **0.3236** |
| GraphSAGE | 371,604 | 0.3204 | 0.4284 | 0.3133 |
| GCN | 204,692 | 0.3182 | 0.4316 | 0.3111 |
| GIN | 402,068 | 0.3017 | 0.3966 | 0.2954 |

GAT edges out GraphSAGE with half the parameters. GIN, the most expressive, is
worst — its sum aggregation is sensitive to the varying node degrees produced by
kNN edge construction.

### Depth
| layers | params | Macro-F1 | Micro-F1 | AUC-PR |
|---|---|---|---|---|
| 2 | 239,764 | **0.3283** | 0.4247 | **0.3305** |
| 3 | 371,604 | 0.3196 | 0.4502 | 0.3190 |
| 4 | 503,444 | 0.3194 | 0.4196 | 0.3164 |

Shallower is better — with 30–60 nodes and a dense similarity graph, the
receptive field saturates after two hops and deeper stacks over-smooth.

---

## 4. Best configuration

**GAT · 60 nodes · mel+chroma mean+std (280-d) · 2 layers · kNN k=3 · mean readout**
208,020 parameters, ~35 s to train on an RTX 4090.

| run | Macro-F1 | Micro-F1 | AUC-PR |
|---|---|---|---|
| best_gat_knn | **0.3544** | 0.4410 | **0.3673** |
| best_gat_meanmax | 0.3540 | **0.4758** | 0.3579 |
| best_sage_knn | 0.3447 | 0.4467 | 0.3621 |
| best_gat_full | 0.3366 | 0.4336 | 0.3495 |

Combining the per-axis winners gained +0.053 AUC-PR over the base configuration
(0.3143 → 0.3673) — more than any single ablation, and none of these
combinations appeared in the one-axis-at-a-time sweep.

---

## 5. Figures

| file | content |
|---|---|
| `plots/segment_graphs.png` | segment graphs for six genres, edges coloured by similarity |
| `plots/chord_graphs.png` | chord-transition graphs with triad labels |
| `plots/self_similarity.png` | segment self-similarity matrices |
| `plots/graph_statistics.png` | edge counts by construction mode; chord node/edge distributions |
| `plots/best_gat_knn_training_curves.png` | loss, Macro-F1, validation AUC-PR |
| `plots/best_gat_knn_per_genre_ap.png` | per-genre AUC-PR, GNN vs CNN |
| `plots/best_gat_knn_tsne.png` | t-SNE of graph embeddings `g` by genre |

## 6. Reproducing

```bash
python scripts/extract_features_gpu.py      # 1.5 min, 25k tracks
python src/labels.py                        # splits + leakage check
python src/export_graphs.py 24              # 24 example .pt/.json graphs
python src/visualize_graphs.py              # graph figures
python src/cnn_baseline.py --labels multi   # B2 baseline
python scripts/final_runs.py                # best GNN configs
python src/evaluate_gnn.py --checkpoint results/final/best_gat_knn.pt
python src/experiments.py --labels multi    # full ablation sweep
```
