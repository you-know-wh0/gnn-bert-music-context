import os
import json
import torch
from pathlib import Path
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from graph_builder import build_graph, build_genre_mapping
from typing import Dict, List, Optional, Tuple



def build_genre_mapping_from_splits(
    train_split_path: str,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build genre ↔ index mapping from the training split only.
    Must use training split to prevent test-set leakage.

    Returns
    -------
    genre_to_idx : {"Blues": 0, "Classical": 1, ...}
    idx_to_genre : {0: "Blues", 1: "Classical", ...}
    """
    with open(train_split_path, "r") as f:
        train_split = json.load(f)
    return build_genre_mapping(train_split)




class MusicGraphDataset(Dataset):
    """
    Loads pre-built PyG graph (.pt) files for each track.

    If a .pt graph doesn't exist yet, it is built on-the-fly from Person 1's
    .npz features and cached for future runs.

    Parameters
    ----------
    split_file    : path to training.json / validation.json / test.json
    graph_dir     : directory containing {track_id}.pt files
    genre_to_idx  : genre name → integer index (built from training split)
    processed_dir : Person 1's audio_features/ directory (for on-the-fly builds)
    sim_threshold : cosine similarity threshold for graph edges
    """

    def __init__(
        self,
        split_file:    str,
        graph_dir:     str,
        genre_to_idx:  Dict[str, int],
        processed_dir: str  = "data/processed/audio_features",
        sim_threshold: float = 0.7,
    ):
        self.graph_dir     = Path(graph_dir)
        self.genre_to_idx  = genre_to_idx
        self.processed_dir = processed_dir
        self.sim_threshold = sim_threshold

        self.graph_dir.mkdir(parents=True, exist_ok=True)

        with open(split_file, "r") as f:
            split_data = json.load(f)

        self.samples: List[Tuple[str, int]] = []
        for track_id, info in split_data.items():
            genre = info.get("genre", "")
            if not genre or genre not in genre_to_idx:
                continue
            self.samples.append((track_id, genre_to_idx[genre]))

        print(f"  Dataset loaded: {len(self.samples)} tracks from {Path(split_file).name}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        track_id, genre_idx = self.samples[idx]
        pt_path = self.graph_dir / f"{track_id}.pt"

        if pt_path.exists():
            graph = torch.load(str(pt_path), weights_only=False)
        else:
            npz_path = Path(self.processed_dir) / f"{track_id}.npz"
            if not npz_path.exists():
                raise FileNotFoundError(
                    f"Missing .npz for track {track_id}. "
                    f"Run Person 1's feature_extraction.py first."
                )
            graph = build_graph(
                track_id      = track_id,
                processed_dir = self.processed_dir,
                genre_idx     = genre_idx,
                sim_threshold = self.sim_threshold,
            )
            torch.save(graph, str(pt_path))

        return graph


def get_dataloaders(
    splits_dir:    str  = "data/splits",
    graph_dir:     str  = "data/processed/graphs",
    processed_dir: str  = "data/processed/audio_features",
    batch_size:    int  = 32,
    sim_threshold: float = 0.7,
    num_workers:   int  = 4,
) -> Tuple[PyGDataLoader, PyGDataLoader, PyGDataLoader, Dict[str, int], Dict[int, str]]:
    """
    Build all three split dataloaders.

    Returns
    -------
    train_loader, val_loader, test_loader, genre_to_idx, idx_to_genre
    """
    train_json = os.path.join(splits_dir, "training.json")
    val_json   = os.path.join(splits_dir, "validation.json")
    test_json  = os.path.join(splits_dir, "test.json")

    genre_to_idx, idx_to_genre = build_genre_mapping_from_splits(train_json)
    print(f"Genre mapping: {len(genre_to_idx)} classes")

    kwargs = dict(
        graph_dir     = graph_dir,
        genre_to_idx  = genre_to_idx,
        processed_dir = processed_dir,
        sim_threshold = sim_threshold,
    )

    train_ds = MusicGraphDataset(train_json, **kwargs)
    val_ds   = MusicGraphDataset(val_json,   **kwargs)
    test_ds  = MusicGraphDataset(test_json,  **kwargs)

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers)

    train_loader = PyGDataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = PyGDataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = PyGDataLoader(test_ds,  shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, genre_to_idx, idx_to_genre


if __name__ == "__main__":
    train_loader, val_loader, test_loader, g2i, i2g = get_dataloaders(batch_size=4)

    batch = next(iter(train_loader))
    print("Sample batch:")
    print(f"  x shape      : {batch.x.shape}")
    print(f"  edge_index   : {batch.edge_index.shape}")
    print(f"  y            : {batch.y}")
    print(f"  batch vector : {batch.batch.shape}")