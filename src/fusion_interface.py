import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
 
 
def fusion_collate(batch):
    """Batches a list of {input_ids, attention_mask, graph, label} dicts.
    BERT tensors stack normally; graphs batch via PyG's Batch.from_data_list
    (PyG's own DataLoader only knows how to batch bare Data objects, not
    this mixed dict, hence a hand-rolled collate_fn)."""
    graphs = Batch.from_data_list([b["graph"] for b in batch])
    out = {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "x": graphs.x, "edge_index": graphs.edge_index, "batch": graphs.batch,
        "label": torch.stack([b["label"] for b in batch]),
    }
    id_key = "ytid" if "ytid" in batch[0] else ("track_id" if "track_id" in batch[0] else None)
    if id_key:
        out["id"] = [b[id_key] for b in batch]
    return out
 
 
class DummyFusionDataset(Dataset):
    """Synthetic stand-in satisfying the contract above -- lets the fusion
    architecture, training loop, and ablation runner be built and tested
    end-to-end before any real paired dataset exists. NOT real data: do
    not report metrics trained on this as project results.
    """
    def __init__(self, n=256, seq_len=32, n_nodes=60, gnn_in_dim=280, num_labels=50, seed=0):
        self.n, self.seq_len, self.n_nodes = n, seq_len, n_nodes
        self.gnn_in_dim, self.num_labels, self.seed = gnn_in_dim, num_labels, seed
 
    def __len__(self):
        return self.n
 
    def __getitem__(self, i):
        g = torch.Generator().manual_seed(self.seed * 100003 + i)
        input_ids = torch.randint(0, 30522, (self.seq_len,), generator=g)  # bert-base-uncased vocab size
        attention_mask = torch.ones(self.seq_len, dtype=torch.long)
        x = torch.randn(self.n_nodes, self.gnn_in_dim, generator=g)
        edge_index = torch.randint(0, self.n_nodes, (2, self.n_nodes * 3), generator=g)
        graph = Data(x=x, edge_index=edge_index, num_nodes=self.n_nodes)
        label = (torch.rand(self.num_labels, generator=g) > 0.85).float()
        return {"input_ids": input_ids, "attention_mask": attention_mask,
                "graph": graph, "label": label}
 
 
def make_dummy_datasets(n_train=256, n_val=64, n_test=64, num_labels=50, gnn_in_dim=280):
    """For local smoke-testing only -- NOT real data. gnn_in_dim=280 matches
    final_runs.py's best_gat_knn config ((128 mel + 12 chroma) * 2 for
    with_std=True), so a real GNN checkpoint still loads with the right
    input shape once you switch to real data."""
    vocab = [f"tag_{i}" for i in range(num_labels)]
    datasets = {
        "training": DummyFusionDataset(n_train, num_labels=num_labels, gnn_in_dim=gnn_in_dim, seed=0),
        "validation": DummyFusionDataset(n_val, num_labels=num_labels, gnn_in_dim=gnn_in_dim, seed=1),
        "test": DummyFusionDataset(n_test, num_labels=num_labels, gnn_in_dim=gnn_in_dim, seed=2),
    }
    return datasets, vocab