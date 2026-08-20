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
