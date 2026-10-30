#Calibration and conformal prediction 

#Everything here is fitted on the validation set and only applied to the test set.

from __future__ import annotations

import numpy as np
import torch

from .models import grade_probs


def nll(logits: np.ndarray, y: np.ndarray, head: str, t: float) -> float:
    p = grade_probs(torch.as_tensor(logits, dtype=torch.float32), head, t).numpy()
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())


def fit_temperature(logits: np.ndarray, y: np.ndarray, head: str) -> float:
    
    grid = np.exp(np.linspace(np.log(0.25), np.log(8.0), 121))
    best = min(grid, key=lambda t: nll(logits, y, head, t))
    fine = np.linspace(best / 1.06, best * 1.06, 41)
    return float(min(fine, key=lambda t: nll(logits, y, head, t)))

      #Finite-sample-corrected conformal quantile.
def _quantile(scores: np.ndarray, alpha: float) -> float:
    
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def fit_conformal(probs_val: np.ndarray, y_val: np.ndarray, alpha: float = 0.1,
                  class_conditional: bool = True) -> np.ndarray:
    
    scores = 1 - probs_val[np.arange(len(y_val)), y_val]
    k = probs_val.shape[1]
    if not class_conditional:
        return np.full(k, _quantile(scores, alpha))
    return np.array([_quantile(scores[y_val == c], alpha) for c in range(k)])


def prediction_sets(probs: np.ndarray, qhat: np.ndarray, point: np.ndarray | None = None) -> np.ndarray:
    
    sets = (1 - probs) <= qhat[None, :]
    empty = ~sets.any(1)
    sets[empty, probs[empty].argmax(1)] = True
    if point is not None:
        sets[np.arange(len(point)), point] = True
    return sets


def set_summary(sets: np.ndarray, y: np.ndarray) -> dict:
    covered = sets[np.arange(len(y)), y]
    size = sets.sum(1)
    contiguous = np.array([np.all(np.diff(np.flatnonzero(s)) == 1) for s in sets])
    return {
        "coverage": float(covered.mean()),
        "coverage_per_grade": {int(c): float(covered[y == c].mean()) for c in np.unique(y)},
        "mean_set_size": float(size.mean()),
        "set_size_distribution": {int(s): int((size == s).sum()) for s in np.unique(size)},
        "singleton_rate": float((size == 1).mean()),
        "contiguous_rate": float(contiguous.mean()),
    }


def reliability_curve(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> dict[str, list]:
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, n_bins + 1)
    out = {"conf": [], "acc": [], "count": []}
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out["conf"].append(float(conf[m].mean())); out["acc"].append(float((pred[m] == y[m]).mean()))
            out["count"].append(int(m.sum()))
    return out
