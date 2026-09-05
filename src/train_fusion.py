import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from fusion_interface import fusion_collate, make_dummy_datasets
from fusion_model import FusionModel, count_params
from metrics import multilabel_metrics, tune_thresholds


def run_epoch(model, loader, criterion, device, optimizer=None):
    model.train() if optimizer else model.eval()
    total, probs, ys = 0.0, [], []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        x = batch["x"].to(device)
        ei = batch["edge_index"].to(device)
        bi = batch["batch"].to(device)
        y = batch["label"].to(device)
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(input_ids=ids, attention_mask=mask, x=x, edge_index=ei, batch=bi)
            loss = criterion(logits, y)
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += loss.item() * y.size(0)
        probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        ys.append(y.detach().cpu().numpy())
    return total / len(loader.dataset), np.concatenate(probs), np.concatenate(ys)


def train(cfg, datasets=None, vocab=None, verbose=True, save=True):
    """datasets/vocab should come from whoever builds the paired
    (text, graph, label) data -- see fusion_interface.py's contract. If
    none are passed, this falls back to synthetic smoke-test data so the
    fusion architecture and training loop can still be exercised and
    demoed on their own."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    if datasets is None:
        if verbose:
            print("[train_fusion] no datasets passed in -- running on synthetic "
                  "smoke-test data (fusion_interface.make_dummy_datasets). Pass "
                  "real datasets/vocab once the paired dataset exists.", flush=True)
        datasets, vocab = make_dummy_datasets()
    loaders = {n: DataLoader(d, batch_size=cfg["batch_size"], shuffle=(n == "training"),
                             collate_fn=fusion_collate)
              for n, d in datasets.items()}

    gnn_in_dim = datasets["training"][0]["graph"].x.shape[1]
    model = FusionModel(
        num_labels=len(vocab), gnn_in_dim=gnn_in_dim, mode=cfg["mode"],
        bert_checkpoint=cfg["bert_checkpoint"], freeze_bert=cfg["freeze_bert"],
        gnn_checkpoint=cfg["gnn_checkpoint"], freeze_gnn=cfg["freeze_gnn"],
        gnn_hidden=cfg["gnn_hidden"], gnn_out=cfg["gnn_out"], gnn_layers=cfg["gnn_layers"],
        gnn_dropout=cfg["gnn_dropout"], gnn_kind=cfg["gnn_kind"], gnn_readout=cfg["gnn_readout"],
        head_hidden=cfg["head_hidden"], head_dropout=cfg["head_dropout"], device=device,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])

    if verbose:
        print(f"fusion[{cfg['mode']}] | {len(vocab)} labels | "
              f"{count_params(model):,} trainable params | "
              f"{len(datasets['training'])} train clips", flush=True)

    history, best, best_state, best_th, patience = [], -1.0, None, None, 0
    t0 = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss, _, _ = run_epoch(model, loaders["training"], criterion, device, opt)
        with torch.no_grad():
            va_loss, va_p, va_y = run_epoch(model, loaders["validation"], criterion, device)
        sched.step()

        th = tune_thresholds(va_y, va_p)
        va_m = multilabel_metrics(va_y, va_p, th)
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
                        "val_macro_f1": va_m["macro_f1"], "val_micro_f1": va_m["micro_f1"],
                        "val_auc_pr": va_m["auc_pr"]})
        if verbose:
            print(f"  ep {epoch:3d} | loss {tr_loss:.4f}/{va_loss:.4f} | "
                  f"val macroF1 {va_m['macro_f1']:.4f} microF1 {va_m['micro_f1']:.4f} "
                  f"AUC-PR {va_m['auc_pr']:.4f}", flush=True)

        if va_m["auc_pr"] > best:
            best, patience, best_th = va_m["auc_pr"], 0, th
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg["patience"]:
                if verbose:
                    print(f"  early stop at epoch {epoch}", flush=True)
                break

    model.load_state_dict(best_state)
    with torch.no_grad():
        _, te_p, te_y = run_epoch(model, loaders["test"], criterion, device)
    test_m = multilabel_metrics(te_y, te_p, best_th)

    result = {"config": cfg, "vocab": vocab,
              "test": {k: v for k, v in test_m.items() if k != "per_class_ap"},
              "per_class_ap": test_m["per_class_ap"], "best_val_auc_pr": best,
              "epochs_run": len(history), "train_minutes": (time.time() - t0) / 60,
              "params": count_params(model), "history": history}

    if verbose:
        print(f"  TEST macroF1 {test_m['macro_f1']:.4f} | microF1 {test_m['micro_f1']:.4f} | "
              f"AUC-PR {test_m['auc_pr']:.4f} | {result['train_minutes']:.1f} min", flush=True)

    if save:
        Path(cfg["results_dir"]).mkdir(parents=True, exist_ok=True)
        name = cfg["run_name"]
        torch.save({"model_state": best_state, "config": cfg, "vocab": vocab,
                    "thresholds": best_th}, f"{cfg['results_dir']}/{name}.pt")
        json.dump(result, open(f"{cfg['results_dir']}/{name}.json", "w"), indent=2)
    return result, model


def default_cfg(**over):
    # gnn_* defaults mirror final_runs.py's "best_gat_knn" config
    # (kind="GAT", layers=2, hidden=256, out_dim=256), so that checkpoint's
    # encoder weights load here as a warm start once real graphs exist.
    # (n_nodes/feat/edge_mode etc. are graph-*construction* config -- that's
    # whoever builds the paired dataset's concern, not fusion's.)
    cfg = dict(mode="cross_attention",
               bert_checkpoint=None, freeze_bert=False,
               gnn_checkpoint=None, freeze_gnn=False,
               gnn_hidden=256, gnn_out=256, gnn_layers=2, gnn_dropout=0.3,
               gnn_kind="GAT", gnn_readout="mean",
               head_hidden=256, head_dropout=0.3,
               lr=2e-5, weight_decay=1e-4, batch_size=32, epochs=15, patience=4, seed=42,
               results_dir="results/fusion", run_name="fusion_cross_attention")
    cfg.update(over)
    return cfg


def parse_args():
    p = argparse.ArgumentParser()
    c = default_cfg()
    for key, val in c.items():
        if isinstance(val, bool):
            p.add_argument(f"--{key}", action="store_true")
        elif val is None:
            p.add_argument(f"--{key}", type=str, default=None)
        else:
            p.add_argument(f"--{key}", type=type(val), default=val)
    return vars(p.parse_args())


if __name__ == "__main__":
    train(parse_args())