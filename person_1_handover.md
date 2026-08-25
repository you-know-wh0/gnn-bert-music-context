# CSE425 Group Project: Preprocessing, Baseline, & Integration Handbook
**Role: Person 1 (Dataset + Audio Preprocessing + CNN Baseline)**

---

## 1. PREPROCESSING WORKFLOW & ARCHITECTURE
Below is the sequential workflow implemented for the audio feature extraction stage:

```
+--------------------+
|  Raw FMA Audio     | (25,000 MP3 tracks, 30s each)
+--------------------+
          |
          v
+--------------------+
|  Resample: 22.05kHz| (Mono channel conversion)
+--------------------+
          |
          v
+--------------------+
| Feature Extraction | -> 128-bin Mel Spectrogram
|                    | -> 12-bin Chroma STFT
+--------------------+
          |
          v
+--------------------+
| Normalize / Track  | (Zero-mean, unit-variance)
+--------------------+
          |
          v
+--------------------+
|  Time Segmentation | (Split into 6 segments of 5s each)
+--------------------+
          |
          v
+--------------------+
|   Save (.npz)      | (Saved to data/processed/audio_features/)
+--------------------+
```

### Specifications Applied (from Project Document)
*   **Resampling**: All audio resampled to **22,050 Hz**.
*   **Log-Mel Spectrogram**: 128 Mel bins, converted to Decibel scale (`librosa.power_to_db`).
*   **Chroma Features**: 12 bins extracted using Chroma Short-Time Fourier Transform (`chroma_stft`).
*   **Normalization**: Computed track-wise (across the entire 30s duration) to maintain relative intensity changes.
*   **Segmentation**: Sliced into exactly **6 segments** (representing ~5 seconds of time per segment).
*   **Split Mappings**: Extracted splits from FMA's metadata to create official partitions (**19,922 training, 2,505 validation, and 2,573 test tracks**) stored in `data/splits/` to guarantee **no artist leakage**.

---

## 2. INTEGRATION HANDOFF FOR TEAM MEMBERS

### 👥 For Person 2: Graph Construction & GNN Specialist
All preprocessed segment features have been saved to `data/processed/audio_features/{track_id}.npz` where `track_id` is a 6-digit padded string (e.g., `000002.npz`).

#### Data Structure inside `.npz`
*   `data["mel"]`: Array of shape `(6, 128, 215)` (6 segments, 128 Mel bins, 215 time frames).
*   `data["chroma"]`: Array of shape `(6, 12, 215)` (6 segments, 12 chroma bins, 215 time frames).

#### How to consume this for GNN Graphs:
1.  **Nodes**: Each of the 6 segments corresponds to a node in your music structure graph (Nodes $V = \{0, 1, 2, 3, 4, 5\}$).
2.  **Node Features**: Average the time dimension of each segment to get a static feature vector for each node:
    *   Mel features: `np.mean(mel, axis=-1)` -> Shape `(6, 128)`
    *   Chroma features: `np.mean(chroma, axis=-1)` -> Shape `(6, 12)`
3.  **Edges**: Establish edges based on:
    *   *Temporal Adjacency*: Connect node $i$ to node $i+1$.
    *   *Segment Similarity*: Calculate cosine similarity between node features (e.g., chroma) and add edges if similarity exceeds threshold $\tau$.

#### Python Loading Template for Person 2:
```python
import numpy as np

# Load features for a specific track
track_id = "000002"
npz_data = np.load(f"data/processed/audio_features/{track_id}.npz")

mel_features = npz_data["mel"]      # shape: (6, 128, 215)
chroma_features = npz_data["chroma"]  # shape: (6, 12, 215)

# Calculate static node features for GNN (mean-pooled over time)
node_features_mel = np.mean(mel_features, axis=-1)  # shape: (6, 128)
node_features_chroma = np.mean(chroma_features, axis=-1)  # shape: (6, 12)
```

---

### 👥 For Person 3: Text & BERT Specialist
You must align text descriptions (lyrics/captions/tags) with the preprocessed audio files using the unique `track_id`.

#### Alignment Guidelines:
1.  Use the 6-digit `track_id` as the primary key.
2.  Import the splits (`data/splits/training.json`, `validation.json`, `test.json`) to determine which subset a track belongs to. This ensures that your BERT train/test inputs exactly match the GNN and CNN baselines.
3.  For BERT tokenization, pad/truncate your token sequence to **128–256 tokens** as specified.

#### Python Split Loader for Person 3:
```python
import json

# Load train splits
with open("data/splits/training.json", "r") as f:
    train_splits = json.load(f)

# Iterate over tracks
for track_id, info in train_splits.items():
    genre = info["genre"]
    # 1. Fetch text data matching track_id
    # 2. Tokenize text using BERT tokenizer
    # 3. Output shape: [L, d] text representation
```

---

### 👥 For Person 4: Integration, Fusion & Experimentation
You are responsible for fusing GNN representations ($g$) and BERT representations ($t$) to predict the genre.

#### Target Labels:
The genres are located in the split files under the `"genre"` key (corresponds to `"genre_top"` in FMA). There are 16 unique top-level genres.

#### PyTorch Dataset Collating:
Ensure you handle segment shape discrepancies! Some FMA files are slightly shorter than 30 seconds, leading to a difference in time frames (e.g., 196 instead of 215). To prevent dataloader batch stack crashes, apply padding/truncating:

```python
# Padding/truncating snippet for DataLoader safety
target_frames = 215
if mel_seg.shape[1] < target_frames:
    pad_width = target_frames - mel_seg.shape[1]
    mel_seg = np.pad(mel_seg, ((0, 0), (0, pad_width)), mode='constant')
elif mel_seg.shape[1] > target_frames:
    mel_seg = mel_seg[:, :target_frames]
```

#### Fusion Readout Setup:
*   **Audio/Graph Embedding ($g$)**: Output size from GNN mean-readout -> `[batch_size, g_dim]`
*   **Text/BERT Embedding ($t$)**: CLS token vector -> `[batch_size, t_dim]`
*   **Recommendation**: Implement both **early concatenation** (`z = CONCAT(g, t)`) and **cross-attention** (`Q = gW_Q, K = H_text W_K`) as required for the Task 3 ablation tables.
