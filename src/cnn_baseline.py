import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(__file__))
import features_store
from cnn_model import MelSpectrogramCNN
from labels import build_vocab, encode_labels, load_split
from metrics import multilabel_metrics, singlelabel_metrics, tune_thresholds

SEGS = features_store.SEGS


class SegmentDataset(Dataset):
    def __init__(self, mel, rows, y):
        self.mel, self.rows, self.y = mel, rows, y

    def __len__(self):
        return len(self.rows) * SEGS

    def __getitem__(self, i):
        t, s = divmod(i, SEGS)
        x = torch.from_numpy(self.mel[self.rows[t], s].astype(np.float32)).unsqueeze(0)
        return x, torch.tensor(self.y[t]), t


def collect(model, loader, device, n_tracks, n_classes, multi):
    model.eval()
    acc = np.zeros((n_tracks, n_classes), dtype=np.float64)
    with torch.no_grad():
        for x, _, t in loader:
            logits = model(x.to(device))
            p = torch.sigmoid(logits) if multi else logits.softmax(1)
            np.add.at(acc, t.numpy(), p.cpu().numpy())
    return acc / SEGS


def main(labels="multi", top_k=20, epochs=12, batch_size=256, lr=1e-3, seed=42,
         results_dir="results", workers=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    store = features_store.load()
    pos, mel = store["pos"], store["mel"]
    usable = set(features_store.usable_ids())

    vocab = build_vocab("top" if labels == "top" else "multi", top_k)
    data = {}
    for name in ("training", "validation", "test"):
        tids, y = encode_labels(load_split(name), vocab, labels)
        keep = [i for i, t in enumerate(tids) if t in usable]
        rows = np.array([pos[tids[i]] for i in keep])
        data[name] = (rows, y[keep].astype(np.float32 if labels == "multi" else np.int64))

    loaders = {n: DataLoader(SegmentDataset(mel, r, y), batch_size=batch_size,
                             shuffle=(n == "training"), num_workers=workers, pin_memory=True)
               for n, (r, y) in data.items()}

    multi = labels == "multi"
    model = MelSpectrogramCNN(len(vocab)).to(device)
    criterion = nn.BCEWithLogitsLoss() if multi else nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    print(f"CNN baseline (Person 1 architecture) | {labels} | {len(vocab)} classes | "
          f"{len(data['training'][0])} train tracks", flush=True)

    best, best_state, history = -1.0, None, []
    for epoch in range(1, epochs + 1):
        model.train()
        total, t0 = 0.0, time.time()
        for x, y, _ in loaders["training"]:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
        sched.step()

        train_loss = total / (len(data["training"][0]) * SEGS)
        v_rows, y_val = data["validation"]
        probs = collect(model, loaders["validation"], device, len(v_rows), len(vocab), multi)
        m = (multilabel_metrics(y_val, probs, tune_thresholds(y_val, probs)) if multi
             else singlelabel_metrics(y_val.astype(int), probs))
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_macro_f1": m["macro_f1"], "val_auc_pr": m["auc_pr"]})
        print(f"  ep {epoch:2d} | loss {train_loss:.4f} | val macroF1 {m['macro_f1']:.4f} "
              f"AUC-PR {m['auc_pr']:.4f} | {(time.time()-t0)/60:.1f} min", flush=True)
        if m["auc_pr"] > best:
            best = m["auc_pr"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    rows_v, y_val = data["validation"]
    val_probs = collect(model, loaders["validation"], device, len(rows_v), len(vocab), multi)
    th = tune_thresholds(y_val, val_probs) if multi else None
    rows_t, y_test = data["test"]
    test_probs = collect(model, loaders["test"], device, len(rows_t), len(vocab), multi)
    test_m = (multilabel_metrics(y_test, test_probs, th) if multi
              else singlelabel_metrics(y_test.astype(int), test_probs))

    print(f"  TEST macroF1 {test_m['macro_f1']:.4f} | microF1 {test_m['micro_f1']:.4f} | "
          f"AUC-PR {test_m['auc_pr']:.4f}", flush=True)

    Path(results_dir).mkdir(exist_ok=True)
    out = {"model": "CNN mel-spectrogram (Person 1 architecture)", "labels": labels,
           "test": {k: v for k, v in test_m.items() if k != "per_class_ap"},
           "per_class_ap": test_m["per_class_ap"], "history": history, "vocab": vocab}
    json.dump(out, open(f"{results_dir}/cnn_baseline_{labels}.json", "w"), indent=2)
    torch.save({"model_state": best_state, "vocab": vocab},
               f"{results_dir}/cnn_baseline_{labels}.pt")
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--labels", default="multi")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--top_k", type=int, default=20)
    a = p.parse_args()
    main(labels=a.labels, epochs=a.epochs, top_k=a.top_k)
