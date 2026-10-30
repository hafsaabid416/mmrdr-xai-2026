#Image-quality robustness and the blur-shortcut test.
#evaluate.py needs to run before this 

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mmrdr.calibration import prediction_sets  
from mmrdr.data import FundusDataset  
from mmrdr.inference import load_run, logits_of, predict_frame  
from mmrdr.metrics import expected_calibration_error, grading_metrics, p_at_least, predict  # noqa: E402
from mmrdr.models import grade_probs  
from mmrdr.plots import plt, save  
from mmrdr.preprocessing import NORMALISERS  
from mmrdr.utils import save_json  

## Degradations simulating poor capture: defocus (Gaussian sigma), under-exposure (gamma > 1), compression (JPEG quality)
def _blur(s):
    return lambda im: cv2.GaussianBlur(im, (0, 0), s)


def _dark(g):          
    lut = np.clip(255 * (np.arange(256) / 255) ** g, 0, 255).astype(np.uint8)
    return lambda im: cv2.LUT(im, lut)


def _jpeg(q):
    return lambda im: cv2.imdecode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)


CONDITIONS = {"clean": lambda im: im,
              "blur σ=1": _blur(1), "blur σ=2": _blur(2), "blur σ=4": _blur(4),
              "dark γ=1.5": _dark(1.5), "dark γ=2.5": _dark(2.5),
              "JPEG q=30": _jpeg(30), "JPEG q=10": _jpeg(10)}


class DegradedDataset(FundusDataset):
    """Applies a degradation to the cached image *before* the run's normaliser, as a camera would."""

    def __init__(self, df, image_dir, normaliser, degrade):
        super().__init__(df, image_dir, "none")
        self.degrade, self.norm = degrade, NORMALISERS[normaliser]

    def load(self, i):
        return self.norm(self.degrade(super().load(i)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    run = Path(a.run); out = run / "robustness"; out.mkdir(exist_ok=True)
    fitted = json.loads((run / "eval" / "test_metrics.json").read_text())["fitted_on_validation"]
    T, rule, thr, qhat = fitted["temperature"], fitted["decision_rule"], fitted["thresholds"], np.array(fitted["conformal_qhat_per_grade"])
    model, cfg = load_run(run)
    head = cfg["model"]["grade_head"]
    test = pd.read_csv(a.splits).query("split == 'test'")

    rows, per_image = [], {}
    for name, fn in CONDITIONS.items():
        ds = DegradedDataset(test, a.images, cfg["data"]["normaliser"], fn)
        pred = predict_frame(model, cfg, test, a.images, num_workers=a.workers, dataset=ds)
        y = pred["grade"].to_numpy()
        p = grade_probs(torch.as_tensor(logits_of(pred), dtype=torch.float32), head, T).numpy()
        m = grading_metrics(y, p, rule, thr)
        sets = prediction_sets(p, qhat, predict(p, rule))
        g0 = y == 0
        ent = -(p * np.log(np.clip(p, 1e-12, 1))).sum(1)
        rows.append({"condition": name, "qwk": m["qwk"], "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                     "referable_sens": m["referable"]["sensitivity"], "referable_spec": m["referable"]["specificity"],
                     "ece": expected_calibration_error(p, y), "mean_entropy": float(ent.mean()),
                     "set_coverage": float(sets[np.arange(len(y)), y].mean()), "mean_set_size": float(sets.sum(1).mean()),
                     "flag_rate_referral_uncertain": float((sets[:, :2].any(1) & sets[:, 2:].any(1)).mean()),
                     # D7 shortcut test: does degradation push healthy eyes towards disease
                     "grade0_mean_expected_grade": float((p[g0] * np.arange(5)).sum(1).mean()),
                     "grade0_false_referral_rate": float((p_at_least(p[g0], 2) >= thr["referable"]).mean())})
        per_image[name] = predict(p, rule)
        print(f"{name:12s} QWK {rows[-1]['qwk']:.3f} | ref sens {rows[-1]['referable_sens']:.3f} spec "
              f"{rows[-1]['referable_spec']:.3f} | set size {rows[-1]['mean_set_size']:.2f} | "
              f"grade-0 false referral {rows[-1]['grade0_false_referral_rate']:.3f}")

    df = pd.DataFrame(rows).set_index("condition")
    df.round(4).to_csv(out / "robustness.csv")
    save_json(df.to_dict(orient="index"), out / "robustness.json")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
    x = np.arange(len(df))
    axes[0].bar(x, df["qwk"], color="#4C72B0"); axes[0].set_ylim(max(0, df["qwk"].min() - .1), 1); axes[0].set_title("QWK")
    axes[1].bar(x, df["mean_set_size"], color="#DD8452"); axes[1].set_title("mean conformal set size\n(does uncertainty rise?)")
    axes[2].bar(x, df["grade0_false_referral_rate"] * 100, color="#C44E52")
    axes[2].set_title("grade-0 eyes referred (%)\n(blur → 'diseased' shortcut?)")
    for ax in axes:
        ax.set_xticks(x, df.index, rotation=40, ha="right")
    fig.suptitle(f"Robustness to synthetic image degradation — {run.name}", y=1.04)
    save(fig, out / "robustness.png")


if __name__ == "__main__":
    main()
