"""Person 4 -- GNN + BERT fusion for multi-label music tag prediction.

FusionModel wraps a (fine-tunable or frozen) BERT text encoder and a
(fine-tunable or frozen) GNN graph encoder and combines their per-track
representations with one of four strategies, so the same class covers
both the final model and every ablation the project rubric asks for:

    bert_only        -- classifier on the BERT [CLS] embedding alone
    gnn_only          -- classifier on the pooled GNN graph embedding alone
    concat            -- classifier on [BERT_cls ; GNN_graph]  (simple fusion)
    cross_attention   -- BERT [CLS] attends over the GNN's per-node embeddings,
                         producing a music-conditioned text representation
                         before the classifier (the "stronger fusion" the
                         project doc recommends over plain concatenation)

Checkpoint loading is deliberately permissive (strict=False + a printed
missing/unexpected count) so a GNN encoder trained on FMA
(results/final/best_gat_knn.pt) can be dropped in here as-is, as long as
whoever builds the paired dataset uses a matching node-feature dim --
see fusion_interface.py for the exact data-shape contract this expects.
"""
import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch
from transformers import BertModel

from gnn_model import GNNEncoder


def load_bert_backbone(model_name="bert-base-uncased", checkpoint=None, freeze=False, device="cpu"):
    """Build a BertModel, optionally initialised from a BertTagClassifier
    checkpoint saved by train.py (state dict keys are 'bert.*' + 'classifier.*')."""
    bert = BertModel.from_pretrained(model_name)
    if checkpoint is not None:
        sd = torch.load(checkpoint, map_location=device)
        bert_sd = {k[len("bert."):]: v for k, v in sd.items() if k.startswith("bert.")}
        missing, unexpected = bert.load_state_dict(bert_sd, strict=False)
        if missing or unexpected:
            print(f"[fusion] bert backbone load: missing={len(missing)} unexpected={len(unexpected)}")
    if freeze:
        for p in bert.parameters():
            p.requires_grad = False
    return bert


def load_gnn_encoder(in_dim, checkpoint=None, freeze=False, device="cpu", **enc_kwargs):
    """Build a GNNEncoder, optionally initialised from a GNNClassifier
    checkpoint saved by train_gnn.py (state dict keys are 'encoder.*' + 'head.*')."""
    enc = GNNEncoder(in_dim, **enc_kwargs)
    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        sd = ckpt["model_state"] if "model_state" in ckpt else ckpt
        enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
        missing, unexpected = enc.load_state_dict(enc_sd, strict=False)
        if missing or unexpected:
            print(f"[fusion] gnn encoder load: missing={len(missing)} unexpected={len(unexpected)}")
    if freeze:
        for p in enc.parameters():
            p.requires_grad = False
    return enc


class CrossAttentionFusion(nn.Module):
    """A single BERT [CLS] query attends over the GNN's per-node embeddings.

    Uses the GNN's node-level output (before pooling), not just the pooled
    graph vector, so the fused representation can pick out which
    segments/chords of the track are most relevant to the caption/tags.
    """
    def __init__(self, text_dim, node_dim, proj_dim=256, heads=4, dropout=0.1):
        super().__init__()
        self.q_proj = nn.Linear(text_dim, proj_dim)
        self.kv_proj = nn.Linear(node_dim, proj_dim)
        self.attn = nn.MultiheadAttention(proj_dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(proj_dim)
        self.out_dim = text_dim + proj_dim

    def forward(self, text_emb, node_emb, node_batch):
        """text_emb: (B, text_dim). node_emb: (N_total_nodes, node_dim) from a
        PyG-batched graph. node_batch: (N_total_nodes,) graph index per node."""
        dense, valid = to_dense_batch(self.kv_proj(node_emb), node_batch)  # (B,max_n,proj_dim),(B,max_n)
        q = self.q_proj(text_emb).unsqueeze(1)                             # (B, 1, proj_dim)
        attended, _ = self.attn(q, dense, dense, key_padding_mask=~valid)
        attended = self.norm(attended.squeeze(1))
        return torch.cat([text_emb, attended], dim=1)


class FusionModel(nn.Module):
    def __init__(self, num_labels, gnn_in_dim, mode="cross_attention",
                 bert_name="bert-base-uncased", bert_checkpoint=None, freeze_bert=False,
                 gnn_checkpoint=None, freeze_gnn=False,
                 gnn_hidden=256, gnn_out=256, gnn_layers=2, gnn_dropout=0.3,
                 gnn_kind="GAT", gnn_readout="mean", head_hidden=256, head_dropout=0.3,
                 device="cpu"):
        super().__init__()
        assert mode in ("bert_only", "gnn_only", "concat", "cross_attention")
        self.mode = mode
        text_dim = 768

        self.bert = None
        if mode in ("bert_only", "concat", "cross_attention"):
            self.bert = load_bert_backbone(bert_name, bert_checkpoint, freeze_bert, device)

        self.gnn = None
        if mode in ("gnn_only", "concat", "cross_attention"):
            self.gnn = load_gnn_encoder(gnn_in_dim, gnn_checkpoint, freeze_gnn, device,
                                        hidden=gnn_hidden, out_dim=gnn_out, layers=gnn_layers,
                                        dropout=gnn_dropout, kind=gnn_kind, readout=gnn_readout)

        if mode == "bert_only":
            fused_dim = text_dim
        elif mode == "gnn_only":
            fused_dim = self.gnn.out_dim
        elif mode == "concat":
            fused_dim = text_dim + self.gnn.out_dim
        else:  # cross_attention
            self.cross_attn = CrossAttentionFusion(text_dim, gnn_out)
            fused_dim = self.cross_attn.out_dim

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, head_hidden), nn.ReLU(),
            nn.Dropout(head_dropout), nn.Linear(head_hidden, num_labels))

    def forward(self, input_ids=None, attention_mask=None, x=None, edge_index=None, batch=None):
        text_emb = graph_emb = node_emb = None
        if self.bert is not None:
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            text_emb = out.last_hidden_state[:, 0, :]
        if self.gnn is not None:
            graph_emb, node_emb = self.gnn(x, edge_index, batch)

        if self.mode == "bert_only":
            fused = text_emb
        elif self.mode == "gnn_only":
            fused = graph_emb
        elif self.mode == "concat":
            fused = torch.cat([text_emb, graph_emb], dim=1)
        else:
            fused = self.cross_attn(text_emb, node_emb, batch)

        return self.classifier(fused)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Shape sanity-check with dummy tensors -- no real checkpoints/data/network needed.
    from torch_geometric.data import Batch, Data

    B, N, gnn_in = 4, 60, 140
    input_ids = torch.randint(0, 1000, (B, 16))
    attn_mask = torch.ones(B, 16, dtype=torch.long)
    graphs = [Data(x=torch.randn(N, gnn_in), edge_index=torch.randint(0, N, (2, N * 3)))
             for _ in range(B)]
    batch = Batch.from_data_list(graphs)

    for mode in ("bert_only", "gnn_only", "concat", "cross_attention"):
        m = FusionModel(num_labels=50, gnn_in_dim=gnn_in, mode=mode, gnn_layers=2)
        logits = m(input_ids=input_ids, attention_mask=attn_mask, x=batch.x,
                   edge_index=batch.edge_index, batch=batch.batch)
        print(f"{mode:16s} logits {tuple(logits.shape)}  params {count_params(m):,}")