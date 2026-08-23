import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from typing import Tuple



class GraphSAGEEncoder(nn.Module):
    """
    Multi-layer GraphSAGE encoder.

    Implements the exact update rule from the project spec:
        h^(l+1)_i = σ( W^(l) · CONCAT(h^(l)_i, MEAN_{j∈N(i)} h^(l)_j) )

    Parameters
    ----------
    input_dim  : node feature dimensionality  (140 = 128 mel + 12 chroma)
    hidden_dim : hidden layer width           (256 recommended)
    output_dim : graph embedding size         (256 recommended)
    num_layers : number of SAGEConv layers    (3 recommended)
    dropout    : dropout probability          (0.3 recommended)
    """

    def __init__(
        self,
        input_dim:  int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        dropout:    float = 0.3,
    ):
        super().__init__()

        dims = (
            [input_dim]
            + [hidden_dim] * (num_layers - 1)
            + [output_dim]
        )

        self.convs = nn.ModuleList([
            SAGEConv(dims[i], dims[i + 1])
            for i in range(num_layers)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(dims[i + 1])
            for i in range(num_layers)
        ])
        self.dropout   = nn.Dropout(dropout)
        self.num_layers = num_layers

    def forward(
        self,
        x:          torch.Tensor,   
        edge_index: torch.Tensor,   
        batch:      torch.Tensor,   
    ) -> Tuple[torch.Tensor, torch.Tensor]:
       
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            if i < self.num_layers - 1:
                x = self.dropout(x)

        g = global_mean_pool(x, batch)  
        return g, x




class GATEncoder(nn.Module):
    def __init__(
        self,
        input_dim:  int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        heads:      int = 4,
        dropout:    float = 0.3,
    ):
        super().__init__()

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

        for i in range(num_layers):
            if i == 0:
                in_ch  = input_dim
                out_ch = hidden_dim // heads
                h      = heads
                concat = True
            elif i < num_layers - 1:
                in_ch  = hidden_dim
                out_ch = hidden_dim // heads
                h      = heads
                concat = True
            else:
                in_ch  = hidden_dim
                out_ch = output_dim
                h      = 1
                concat = False

            self.convs.append(
                GATConv(in_ch, out_ch, heads=h, concat=concat, dropout=dropout)
            )
            out_size = out_ch * h if concat else out_ch
            self.bns.append(nn.BatchNorm1d(out_size))

    def forward(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        batch:      torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            if i < self.num_layers - 1:
                x = self.dropout(x)

        g = global_mean_pool(x, batch)
        return g, x



class GNNClassifier(nn.Module):
    def __init__(
        self,
        input_dim:      int,
        hidden_dim:     int,
        gnn_output_dim: int,
        num_classes:    int,
        num_layers:     int   = 3,
        dropout:        float = 0.3,
        gnn_type:       str   = "GraphSAGE",
    ):
        super().__init__()

        if gnn_type == "GraphSAGE":
            self.encoder = GraphSAGEEncoder(
                input_dim, hidden_dim, gnn_output_dim, num_layers, dropout
            )
        elif gnn_type == "GAT":
            self.encoder = GATEncoder(
                input_dim, hidden_dim, gnn_output_dim, num_layers, dropout=dropout
            )
        else:
            raise ValueError(f"Unknown gnn_type '{gnn_type}'. Use 'GraphSAGE' or 'GAT'.")

        self.classifier = nn.Sequential(
            nn.Linear(gnn_output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self.gnn_output_dim = gnn_output_dim
        self.num_classes    = num_classes

    def forward(
        self,
        x:          torch.Tensor,   
        edge_index: torch.Tensor,   
        batch:      torch.Tensor,   
    ) -> Tuple[torch.Tensor, torch.Tensor]:
       
        g, _ = self.encoder(x, edge_index, batch)
        logits = self.classifier(g)
        return logits, g

    def encode(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        batch:      torch.Tensor,
    ) -> torch.Tensor:
        g, _ = self.encoder(x, edge_index, batch)
        return g




class MelCNNBaseline(nn.Module):
    """
    Replicated from Person 1's cnn_model.py for direct apples-to-apples comparison
    inside evaluate_gnn.py.

    Input  : (B, 1, 128, 215)
    Output : (B, num_classes)
    """

    def __init__(self, num_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)



def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    B          = 4   
    N_NODES    = 6   
    INPUT_DIM  = 140
    NUM_CLASSES = 16

    x= torch.randn(B * N_NODES, INPUT_DIM)
    src = [0,1,1,2,2,3,3,4,4,5] * B
    dst = [1,0,2,1,3,2,4,3,5,4] * B
    offsets = [i * N_NODES for i in range(B) for _ in range(10)]
    src = [s + o for s, o in zip(src, offsets)]
    dst = [d + o for d, o in zip(dst, offsets)]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    batch = torch.tensor([i for i in range(B) for _ in range(N_NODES)])

    for gnn_type in ["GraphSAGE", "GAT"]:
        model = GNNClassifier(
            input_dim=INPUT_DIM, hidden_dim=256, gnn_output_dim=256,
            num_classes=NUM_CLASSES, gnn_type=gnn_type,
        )
        logits, g = model(x, edge_index, batch)
        print(f"[{gnn_type}] logits: {logits.shape}  g: {g.shape}  "
              f"params: {count_parameters(model):,}")

    print("All smoke tests passed.")