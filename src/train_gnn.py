import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from dataset import make_datasets, make_loaders
from gnn_model import GNNClassifier, count_params
from metrics import multilabel_metrics, singlelabel_metrics, tune_thresholds


def run_epoch(model, loader, criterion, device, optimizer=None):
    model.train() if optimizer else model.eval()
    total, probs, ys = 0.0, [], []
    for batch in loader:
        batch = batch.to(device)
        target = batch.y if criterion.__class__ is nn.BCEWithLogitsLoss else batch.y.view(-1)
        with torch.set_grad_enabled(optimizer is not None):
            logits, _ = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(logits, target)
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += loss.item() * batch.num_graphs
        p = torch.sigmoid(logits) if criterion.__class__ is nn.BCEWithLogitsLoss else logits.softmax(1)
        probs.append(p.detach().cpu().numpy())
        ys.append(target.detach().cpu().numpy())
    return total / len(loader.dataset), np.concatenate(probs), np.concatenate(ys)


def score(y, probs, mode, thresholds=None):
    return multilabel_metrics(y, probs, thresholds) if mode == "multi" else singlelabel_metrics(y, probs)


def train(cfg, datasets=None, vocab=None, verbose=True, save=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    if datasets is None:
        datasets, vocab = make_datasets(
            labels=cfg["labels"], top_k=cfg["top_k"], graph=cfg["graph"],
            n_nodes=cfg["n_nodes"], feat=cfg["feat"], with_std=cfg["with_std"],
            edge_mode=cfg["edge_mode"], k=cfg["k"], tau=cfg["tau"])
    loaders = make_loaders(datasets, cfg["batch_size"])

    in_dim = datasets["training"][0].x.shape[1]
    n_classes = len(vocab)
    model = GNNClassifier(in_dim, n_classes, cfg["hidden"], cfg["out_dim"], cfg["layers"],
                          cfg["dropout"], cfg["kind"], cfg["readout"]).to(device)
    criterion = nn.BCEWithLogitsLoss() if cfg["labels"] == "multi" else nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])

    if verbose:
        print(f"{cfg['kind']} | in_dim {in_dim} | {n_classes} classes | "
              f"{count_params(model):,} params | nodes {datasets['training'][0].num_nodes} | "
              f"edges {datasets['training'][0].edge_index.shape[1]}", flush=True)

    history, best, best_state, patience = [], -1.0, None, 0
    t0 = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss, tr_p, tr_y = run_epoch(model, loaders["training"], criterion, device, opt)
        with torch.no_grad():
            va_loss, va_p, va_y = run_epoch(model, loaders["validation"], criterion, device)
        sched.step()

        th = tune_thresholds(va_y, va_p) if cfg["labels"] == "multi" else None
        va_m = score(va_y, va_p, cfg["labels"], th)
        tr_m = score(tr_y, tr_p, cfg["labels"])
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
                        "train_macro_f1": tr_m["macro_f1"], "val_macro_f1": va_m["macro_f1"],
                        "val_micro_f1": va_m["micro_f1"], "val_auc_pr": va_m["auc_pr"]})

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  ep {epoch:3d} | loss {tr_loss:.4f}/{va_loss:.4f} | "
                  f"val macroF1 {va_m['macro_f1']:.4f} microF1 {va_m['micro_f1']:.4f} "
                  f"AUC-PR {va_m['auc_pr']:.4f}", flush=True)

        if va_m["auc_pr"] > best:
            best, patience = va_m["auc_pr"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_th = th
        else:
            patience += 1
            if patience >= cfg["patience"]:
                if verbose:
                    print(f"  early stop at epoch {epoch}", flush=True)
                break

    model.load_state_dict(best_state)
    with torch.no_grad():
        _, te_p, te_y = run_epoch(model, loaders["test"], criterion, device)
        _, va_p, va_y = run_epoch(model, loaders["validation"], criterion, device)
    th = tune_thresholds(va_y, va_p) if cfg["labels"] == "multi" else None
    test_m = score(te_y, te_p, cfg["labels"], th)

    result = {"config": cfg, "vocab": vocab, "test": {k: v for k, v in test_m.items() if k != "per_class_ap"},
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
                    "thresholds": th}, f"{cfg['results_dir']}/{name}.pt")
        json.dump(result, open(f"{cfg['results_dir']}/{name}.json", "w"), indent=2)
    return result, model


def default_cfg(**over):
    cfg = dict(labels="multi", top_k=20, graph="segment", n_nodes=30, feat="mel+chroma",
               with_std=False, edge_mode="knn", k=3, tau=0.7, kind="GraphSAGE", readout="mean",
               hidden=256, out_dim=256, layers=3, dropout=0.3, lr=1e-3, weight_decay=1e-4,
               batch_size=256, epochs=80, patience=15, seed=42, results_dir="results",
               run_name="gnn")
    cfg.update(over)
    return cfg


def parse_args():
    p = argparse.ArgumentParser()
    c = default_cfg()
    for key, val in c.items():
        if isinstance(val, bool):
            p.add_argument(f"--{key}", action="store_true")
        else:
            p.add_argument(f"--{key}", type=type(val), default=val)
    return vars(p.parse_args())


if __name__ == "__main__":
    train(parse_args())
