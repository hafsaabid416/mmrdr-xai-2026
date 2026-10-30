#evaluation metrics 


from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, precision_recall_fscore_support

from . import GRADES

K = len(GRADES)


def predict(probs: np.ndarray, rule: str = "argmax") -> np.ndarray:
    if rule == "argmax":
        return probs.argmax(1)
    if rule == "median":
        return (np.cumsum(probs, 1) < 0.5).sum(1).clip(0, K - 1)
    raise ValueError(rule)


def p_at_least(probs: np.ndarray, k: int) -> np.ndarray:
    return probs[:, k:].sum(1)


def binary_rates(y_bin: np.ndarray, p_bin: np.ndarray) -> dict[str, float]:
    tp = int(((y_bin == 1) & (p_bin == 1)).sum()); fn = int(((y_bin == 1) & (p_bin == 0)).sum())
    tn = int(((y_bin == 0) & (p_bin == 0)).sum()); fp = int(((y_bin == 0) & (p_bin == 1)).sum())
    return {"sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
            "ppv": tp / max(tp + fp, 1), "npv": tn / max(tn + fn, 1)}


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:

    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(ece)


def grading_metrics(y: np.ndarray, probs: np.ndarray, rule: str = "argmax",
                    thresholds: dict[str, float] | None = None) -> dict:
    
    pred = predict(probs, rule)
    err = pred - y
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=range(K), zero_division=0)
    out = {
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=range(K), zero_division=0)),
        "qwk": float(cohen_kappa_score(y, pred, weights="quadratic", labels=range(K))),
        "mae": float(np.abs(err).mean()),
        "under_grading_rate": float((err < 0).mean()),
        "distant_error_rate": float((np.abs(err) >= 2).mean()),
        "missed_sight_threatening": int(((y >= 3) & (pred <= 1)).sum()),
        "ece": expected_calibration_error(probs, y),
        "per_class": {GRADES[k]: {"precision": float(p[k]), "recall": float(r[k]), "f1": float(f[k]),
                                  "support": int(s[k])} for k in range(K)},
        "confusion_matrix": confusion_matrix(y, pred, labels=range(K)).tolist(),
    }
    thresholds = thresholds or {}
    for name, k in (("referable", 2), ("sight_threatening", 3)):
        t = thresholds.get(name)
        p_bin = (p_at_least(probs, k) >= t).astype(int) if t is not None else (pred >= k).astype(int)
        out[name] = binary_rates((y >= k).astype(int), p_bin) | {"threshold": t}
    return out

     #operating threshold fitted on validation only
def threshold_for_sensitivity(y_bin: np.ndarray, score: np.ndarray, target: float) -> float:
    
    if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
        return 0.5
    u = np.unique(score)
    cands = np.concatenate([[u[0] - 1e-6], (u[:-1] + u[1:]) / 2])   
    pos, neg = score[y_bin == 1], score[y_bin == 0]
    sens = np.array([(pos >= t).mean() for t in cands])
    spec = np.array([(neg < t).mean() for t in cands])
    ok = sens >= target
    if not ok.any():
        return float(cands.min())
    j = np.where(ok, sens + spec, -np.inf)
    return float(cands[int(np.argmax(j))])                            


def bootstrap_ci(metric: Callable[[np.ndarray, np.ndarray], float], y: np.ndarray, probs: np.ndarray,
                 n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    
    rng = np.random.default_rng(seed)
    stats = [metric(y[idx], probs[idx]) for idx in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))]
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(metric(y, probs)), float(lo), float(hi)
