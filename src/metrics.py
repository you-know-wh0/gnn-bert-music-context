import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score


def auc_pr(y_true, probs):
    """Per-class average precision; None for classes with no positive test example."""
    per = [float(average_precision_score(y_true[:, k], probs[:, k]))
           if y_true[:, k].sum() > 0 else None
           for k in range(y_true.shape[1])]
    seen = [v for v in per if v is not None]
    return (float(np.mean(seen)) if seen else 0.0), per


def tune_thresholds(y_true, probs, grid=np.arange(0.05, 0.95, 0.025)):
    th = np.full(probs.shape[1], 0.5)
    for k in range(probs.shape[1]):
        if y_true[:, k].sum() == 0:
            continue
        scores = [f1_score(y_true[:, k], probs[:, k] > t, zero_division=0) for t in grid]
        th[k] = grid[int(np.argmax(scores))]
    return th


def multilabel_metrics(y_true, probs, thresholds=None):
    th = 0.5 if thresholds is None else thresholds
    pred = (probs > th).astype(int)
    mean_ap, per_class = auc_pr(y_true, probs)
    return {
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, pred, average="micro", zero_division=0)),
        "auc_pr": mean_ap,
        "per_class_ap": per_class,
    }


def onehot(y, n):
    m = np.zeros((len(y), n), dtype=int)
    m[np.arange(len(y)), y] = 1
    return m


def singlelabel_metrics(y_true, probs):
    pred = probs.argmax(1)
    mean_ap, per_class = auc_pr(onehot(y_true, probs.shape[1]), probs)
    return {
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, pred, average="micro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "auc_pr": mean_ap,
        "per_class_ap": per_class,
    }


def random_baseline(y_true, mode, seed=0, n_classes=None):
    rng = np.random.default_rng(seed)
    if mode == "multi":
        prior = y_true.mean(0)
        probs = rng.random((len(y_true), y_true.shape[1])) * prior
        return multilabel_metrics(y_true, probs, tune_thresholds(y_true, probs))
    probs = rng.random((len(y_true), n_classes))
    return singlelabel_metrics(y_true, probs)
