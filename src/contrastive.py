"""Task 4 (Advanced / Bonus): Cross-Modal Contrastive Alignment (InfoNCE).

Implements a dual-encoder GNN-BERT architecture mapping audio structure graphs
and natural-language descriptions into a shared embedding space using symmetric
InfoNCE loss. Evaluates cross-modal retrieval (R@1, R@5, R@10) and zero-shot tag prediction.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertModel

sys.path.insert(0, os.path.dirname(__file__))
from fusion_model import load_gnn_encoder
from paired_dataset import make_paired_datasets, paired_collate

RESULTS_DIR = "results"
PLOTS = "plots"


class ProjectionHead(nn.Module):
    """Maps encoder embeddings to a shared normalized metric space."""
    def __init__(self, in_dim, proj_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1)


class ContrastiveDualEncoder(nn.Module):
    """Dual-encoder combining a GNN audio graph encoder and a BERT text encoder."""
    def __init__(self, gnn_in_dim=280, proj_dim=128, gnn_checkpoint=None,
                 bert_name="bert-base-uncased", temperature=0.07):
        super().__init__()
        self.gnn = load_gnn_encoder(
            in_dim=gnn_in_dim,
            checkpoint=gnn_checkpoint,
            freeze=True,
            hidden=256,
            out_dim=256,
            layers=2,
            kind="GAT",
            readout="mean"
        )
        self.bert = BertModel.from_pretrained(bert_name)
        for p in self.bert.parameters():
            p.requires_grad = False  # Freeze BERT backbone for efficient alignment

        self.audio_proj = ProjectionHead(in_dim=256, proj_dim=proj_dim)
        self.text_proj = ProjectionHead(in_dim=768, proj_dim=proj_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1.0 / temperature))

    def encode_audio(self, x, edge_index, batch):
        graph_emb, _ = self.gnn(x, edge_index, batch)
        return self.audio_proj(graph_emb)

    def encode_text(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = out.last_hidden_state[:, 0, :]
        return self.text_proj(cls_emb)

    def forward(self, input_ids, attention_mask, x, edge_index, batch):
        audio_emb = self.encode_audio(x, edge_index, batch)
        text_emb = self.encode_text(input_ids, attention_mask)
        return audio_emb, text_emb


def info_nce_loss(audio_emb, text_emb, logit_scale):
    """Symmetric InfoNCE contrastive loss."""
    scale = logit_scale.exp().clamp(max=100.0)
    sim = torch.matmul(audio_emb, text_emb.t()) * scale

    labels = torch.arange(audio_emb.size(0), device=audio_emb.device)
    loss_a2t = F.cross_entropy(sim, labels)
    loss_t2a = F.cross_entropy(sim.t(), labels)
    return (loss_a2t + loss_t2a) / 2.0


def evaluate_retrieval(model, loader, device):
    """Compute bi-directional Recall@1, Recall@5, Recall@10 and MRR."""
    model.eval()
    audio_embs, text_embs, track_ids, texts, labels = [], [], [], [], []

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            x = batch["x"].to(device)
            ei = batch["edge_index"].to(device)
            bi = batch["batch"].to(device)

            a_emb = model.encode_audio(x, ei, bi)
            t_emb = model.encode_text(ids, mask)

            audio_embs.append(a_emb.cpu())
            text_embs.append(t_emb.cpu())
            track_ids.extend(batch["track_id"])
            texts.extend(batch["text"])
            labels.append(batch["label"].cpu())

    audio_mat = torch.cat(audio_embs, dim=0)
    text_mat = torch.cat(text_embs, dim=0)
    labels_mat = torch.cat(labels, dim=0)
    n = audio_mat.size(0)

    # Similarity matrix: (N, N)
    sim_matrix = torch.matmul(text_mat, audio_mat.t()).numpy()

    # Text -> Audio retrieval
    ranks_t2a = []
    for i in range(n):
        scores = sim_matrix[i]
        rank = np.where(np.argsort(scores)[::-1] == i)[0][0] + 1
        ranks_t2a.append(rank)

    # Audio -> Text retrieval
    ranks_a2t = []
    sim_t = sim_matrix.T
    for i in range(n):
        scores = sim_t[i]
        rank = np.where(np.argsort(scores)[::-1] == i)[0][0] + 1
        ranks_a2t.append(rank)

    ranks_t2a = np.array(ranks_t2a)
    ranks_a2t = np.array(ranks_a2t)

    metrics = {
        "text_to_audio": {
            "R@1": float(np.mean(ranks_t2a <= 1)),
            "R@5": float(np.mean(ranks_t2a <= 5)),
            "R@10": float(np.mean(ranks_t2a <= 10)),
            "MRR": float(np.mean(1.0 / ranks_t2a)),
        },
        "audio_to_text": {
            "R@1": float(np.mean(ranks_a2t <= 1)),
            "R@5": float(np.mean(ranks_a2t <= 5)),
            "R@10": float(np.mean(ranks_a2t <= 10)),
            "MRR": float(np.mean(1.0 / ranks_a2t)),
        },
        "total_test_samples": n,
    }

    return metrics, sim_matrix, track_ids, texts, labels_mat, audio_mat, text_mat


def zero_shot_tagging(audio_embs, labels_mat, vocab, model, device):
    """Zero-shot classification via cosine similarity between audio graphs and genre prompt embeddings."""
    prompts = [f"A {g} music track." for g in vocab]
    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    enc = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(device)

    model.eval()
    with torch.no_grad():
        tag_text_embs = model.encode_text(enc["input_ids"], enc["attention_mask"]).cpu()

    sims = torch.matmul(audio_embs, tag_text_embs.t()).numpy()  # (N, 20)
    probs = 1.0 / (1.0 + np.exp(-sims * 5.0))  # scaled sigmoid

    from metrics import multilabel_metrics, tune_thresholds
    y_true = labels_mat.numpy()
    th = tune_thresholds(y_true, probs)
    zs_metrics = multilabel_metrics(y_true, probs, th)
    return zs_metrics


def train_contrastive(epochs=12, batch_size=32, lr=1e-3, split_limit=600):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[contrastive] running on {device}...")

    datasets, vocab = make_paired_datasets(split_limit=split_limit)
    train_loader = DataLoader(datasets["training"], batch_size=batch_size, shuffle=True, collate_fn=paired_collate)
    val_loader = DataLoader(datasets["validation"], batch_size=batch_size, shuffle=False, collate_fn=paired_collate)
    test_loader = DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, collate_fn=paired_collate)

    gnn_ckpt = "results/final/best_gat_knn.pt" if os.path.exists("results/final/best_gat_knn.pt") else None
    model = ContrastiveDualEncoder(gnn_in_dim=280, proj_dim=128, gnn_checkpoint=gnn_ckpt).to(device)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    print(f"[contrastive] training dual-encoder for {epochs} epochs...")
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            x = batch["x"].to(device)
            ei = batch["edge_index"].to(device)
            bi = batch["batch"].to(device)

            optimizer.zero_grad()
            a_emb, t_emb = model(ids, mask, x, ei, bi)
            loss = info_nce_loss(a_emb, t_emb, model.logit_scale)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(ids)

        scheduler.step()
        avg_loss = total_loss / len(datasets["training"])

        # Quick validation
        val_metrics, _, _, _, _, _, _ = evaluate_retrieval(model, val_loader, device)
        val_r5 = val_metrics["text_to_audio"]["R@5"]
        history.append({"epoch": ep, "train_loss": avg_loss, "val_R@5": val_r5})
        print(f"  ep {ep:2d} | train_loss: {avg_loss:.4f} | val Text->Audio R@5: {val_r5:.4f}", flush=True)

    # Final Evaluation on test set
    print("[contrastive] running full test evaluation...")
    test_metrics, sim_matrix, track_ids, texts, labels_mat, audio_mat, text_mat = evaluate_retrieval(model, test_loader, device)

    # Zero-shot classification
    zs_metrics = zero_shot_tagging(audio_mat, labels_mat, vocab, model, device)
    test_metrics["zero_shot_classification"] = zs_metrics

    # Save model checkpoint
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    ckpt_path = f"{RESULTS_DIR}/contrastive_model.pt"
    torch.save({"model_state": model.state_dict(), "metrics": test_metrics, "history": history}, ckpt_path)

    # 10 Qualitative retrieval examples (Query Text -> Top 3 Audio Tracks)
    qualitative_examples = []
    for i in range(min(10, len(texts))):
        scores = sim_matrix[i]
        top_indices = np.argsort(scores)[::-1][:3]
        qualitative_examples.append({
            "query_index": i,
            "query_text": texts[i],
            "correct_track_id": track_ids[i],
            "retrieved_top_3": [
                {
                    "rank": r + 1,
                    "track_id": track_ids[idx],
                    "similarity_score": float(scores[idx]),
                    "is_exact_match": bool(idx == i),
                }
                for r, idx in enumerate(top_indices)
            ]
        })

    with open(f"{RESULTS_DIR}/retrieval_examples.json", "w") as f:
        json.dump(qualitative_examples, f, indent=2)

    with open(f"{RESULTS_DIR}/contrastive_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    # Retrieval Ranking Plot
    Path(PLOTS).mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    k_vals = ["R@1", "R@5", "R@10"]
    t2a_vals = [test_metrics["text_to_audio"][k] for k in k_vals]
    a2t_vals = [test_metrics["audio_to_text"][k] for k in k_vals]
    x_pos = np.arange(len(k_vals))
    width = 0.35

    ax.bar(x_pos - width/2, t2a_vals, width, label="Text -> Audio", color="steelblue")
    ax.bar(x_pos + width/2, a2t_vals, width, label="Audio -> Text", color="indianred")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(k_vals)
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.0)
    ax.set_title("Cross-Modal Retrieval Performance (InfoNCE)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{PLOTS}/retrieval_ranking.png", dpi=150)
    plt.close(fig)

    print(f"\n=== Contrastive Retrieval Results ===")
    print(f"Text -> Audio: R@1: {test_metrics['text_to_audio']['R@1']:.4f} | R@5: {test_metrics['text_to_audio']['R@5']:.4f} | R@10: {test_metrics['text_to_audio']['R@10']:.4f} | MRR: {test_metrics['text_to_audio']['MRR']:.4f}")
    print(f"Audio -> Text: R@1: {test_metrics['audio_to_text']['R@1']:.4f} | R@5: {test_metrics['audio_to_text']['R@5']:.4f} | R@10: {test_metrics['audio_to_text']['R@10']:.4f} | MRR: {test_metrics['audio_to_text']['MRR']:.4f}")
    print(f"Zero-Shot Tagging: Macro-F1: {zs_metrics['macro_f1']:.4f} | Micro-F1: {zs_metrics['micro_f1']:.4f} | AUC-PR: {zs_metrics['auc_pr']:.4f}")
    print(f"Saved {RESULTS_DIR}/retrieval_examples.json and {PLOTS}/retrieval_ranking.png")
    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--split_limit", type=int, default=500)
    args = parser.parse_args()
    train_contrastive(epochs=args.epochs, batch_size=args.batch_size, split_limit=args.split_limit)
