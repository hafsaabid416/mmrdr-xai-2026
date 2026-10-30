
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

GRADE_SHORT = ["0 No DR", "1 Mild", "2 Mod", "3 Sev", "4 PDR"]
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


def save(fig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path); plt.close(fig)


def confusion(cm: np.ndarray, path: str | Path, title: str) -> None:
    cm = np.asarray(cm); rown = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    im = ax.imshow(rown, cmap="Blues", vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v}\n{rown[i, j]:.0%}", ha="center", va="center", fontsize=7,
                color="white" if rown[i, j] > .55 else "black")
        if j < i and v:                                   
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="#C44E52", lw=1.2))
    ax.set_xticks(range(5), GRADE_SHORT, rotation=30); ax.set_yticks(range(5), GRADE_SHORT)
    ax.set_xlabel("predicted"); ax.set_ylabel("reference grade"); ax.set_title(title)
    fig.colorbar(im, fraction=.046, label="row %")
    save(fig, path)


def reliability(before: dict, after: dict, ece_b: float, ece_a: float, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(3.8, 3.6))
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    ax.plot(before["conf"], before["acc"], "o-", c="#DD8452", label=f"uncalibrated (ECE {ece_b:.3f})")
    ax.plot(after["conf"], after["acc"], "o-", c="#4C72B0", label=f"temperature-scaled (ECE {ece_a:.3f})")
    ax.set_xlabel("confidence (max probability)"); ax.set_ylabel("accuracy"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("Reliability diagram (test)"); save(fig, path)


def selective(cov: np.ndarray, acc: np.ndarray, qwk: np.ndarray, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.plot(cov * 100, acc, c="#4C72B0", label="accuracy"); ax.plot(cov * 100, qwk, c="#55A868", label="QWK")
    ax.invert_xaxis(); ax.set_xlabel("% of cases graded automatically (rest referred to clinician)")
    ax.set_ylabel("score on retained cases"); ax.legend(frameon=False); ax.set_title("Selective prediction (test)")
    save(fig, path)
