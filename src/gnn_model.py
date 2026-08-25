import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv, global_max_pool, global_mean_pool

CONVS = ("GraphSAGE", "GAT", "GCN", "GIN")


def make_conv(kind, in_dim, out_dim, heads=4):
    if kind == "GraphSAGE":
        return SAGEConv(in_dim, out_dim)
    if kind == "GCN":
        return GCNConv(in_dim, out_dim)
    if kind == "GIN":
        mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))
        return GINConv(mlp)
    if kind == "GAT":
        return GATConv(in_dim, out_dim // heads, heads=heads, concat=True)
    raise ValueError(kind)


class GNNEncoder(nn.Module):
    def __init__(self, in_dim, hidden=256, out_dim=256, layers=3, dropout=0.3,
                 kind="GraphSAGE", readout="mean"):
        super().__init__()
        dims = [in_dim] + [hidden] * (layers - 1) + [out_dim]
        self.convs = nn.ModuleList([make_conv(kind, dims[i], dims[i + 1]) for i in range(layers)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(dims[i + 1]) for i in range(layers)])
        self.drop = nn.Dropout(dropout)
        self.readout = readout
        self.out_dim = out_dim * (2 if readout == "mean+max" else 1)

    def forward(self, x, edge_index, batch):
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = F.relu(norm(conv(x, edge_index)))
            if i < len(self.convs) - 1:
                x = self.drop(x)
        if self.readout == "mean+max":
            g = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        else:
            g = global_mean_pool(x, batch)
        return g, x


class GNNClassifier(nn.Module):
    def __init__(self, in_dim, num_classes, hidden=256, out_dim=256, layers=3,
                 dropout=0.3, kind="GraphSAGE", readout="mean"):
        super().__init__()
        self.encoder = GNNEncoder(in_dim, hidden, out_dim, layers, dropout, kind, readout)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.out_dim, hidden // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden // 2, num_classes))

    def forward(self, x, edge_index, batch):
        g, _ = self.encoder(x, edge_index, batch)
        return self.head(g), g

    def encode(self, batch):
        g, _ = self.encoder(batch.x, batch.edge_index, batch.batch)
        return g


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    from torch_geometric.data import Batch, Data
    graphs = [Data(x=torch.randn(30, 140),
                   edge_index=torch.randint(0, 30, (2, 90)),
                   y=torch.rand(1, 20).round()) for _ in range(8)]
    b = Batch.from_data_list(graphs)
    for kind in CONVS:
        m = GNNClassifier(140, 20, kind=kind)
        logits, g = m(b.x, b.edge_index, b.batch)
        print(f"{kind:10s} logits {tuple(logits.shape)}  g {tuple(g.shape)}  "
              f"encode {tuple(m.encode(b).shape)}  params {count_params(m):,}")
