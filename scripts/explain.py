#Grad-CAM explanations, clinician panels and a saliency sanity check for selected test cases

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mmrdr import LESION_NAMES, LESIONS  
from mmrdr.data import FundusDataset, to_input  
from mmrdr.explain import GradCAM, map_similarity, overlay, randomised_copy  
from mmrdr.inference import load_run  
from mmrdr.plots import plt, save  
from mmrdr.utils import save_json  

CASES = [  # (key, title, row filter, ranking column, descending)
    ("correct_pdr", "Correct: PDR", lambda d: (d.grade == 4) & (d.pred == 4), "p_cal_4", True),
    ("correct_moderate", "Correct: Moderate NPDR", lambda d: (d.grade == 2) & (d.pred == 2), "p_cal_2", True),
    ("correct_nodr", "Correct: No DR", lambda d: (d.grade == 0) & (d.pred == 0), "p_cal_0", True),
    ("missed_sight_threatening", "Error: sight-threatening graded ≤ Mild", lambda d: (d.grade >= 3) & (d.pred <= 1), "conf", True),
    ("confident_error", "Error: most confident wrong", lambda d: d.pred != d.grade, "conf", True),
    ("inconsistency_flag", "Flag: lesion findings disagree with grade",
     lambda d: d.get("pdr_signs_but_graded_below_4", False) | d.get("lesions_but_graded_0", False), "conf", True),
    ("referral_uncertain", "Flag: plausible grades span the referral boundary",
     lambda d: d.get("referral_uncertain", False), "conf", False),
]


def tensor(ds: FundusDataset, i: int, dev):
    img = ds.load(i)
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None]
    import torch
    return img, to_input(torch.from_numpy(np.ascontiguousarray(x)), dev)


def panel(img, cam, row, fitted, has_lesions, path, title):
    """Clinician-facing summary: separates model prediction, explanation and clinical judgement (brief §3.3)."""
    probs = np.array([row[f"p_cal_{k}"] for k in range(5)])
    in_set = [str(k) in str(row["set"]) for k in range(5)]
    thr = fitted["thresholds"]["referable"]
    p_ref = probs[2:].sum()
    fig = plt.figure(figsize=(13, 4.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, .9, 1.1], wspace=.5)
    ax = fig.add_subplot(gs[0]); ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax.axis("off")
    ax.set_title("Fundus image", fontsize=10)
    ax = fig.add_subplot(gs[1]); ax.imshow(overlay(img, cam)); ax.axis("off")
    ax.set_title("EXPLANATION — regions raising\nthe severity estimate (Grad-CAM)", fontsize=9)
    ax = fig.add_subplot(gs[2])
    ax.barh(range(5), probs, color=["#4C72B0" if s else "#C9D3E3" for s in in_set])
    ax.set_yticks(range(5), ["0 No DR", "1 Mild", "2 Moderate", "3 Severe", "4 PDR"], fontsize=8); ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.set_title(f"PREDICTION — grade {int(row['pred'])}\n90 % set: {{{', '.join(str(row['set']))}}}", fontsize=9)
    ax.set_xlabel("calibrated probability", fontsize=8)
    ax = fig.add_subplot(gs[3])
    if has_lesions:
        lp = np.array([row[f"lesion_{l}"] for l in LESIONS])
        ax.barh(range(7), lp, color=["#C44E52" if v > .5 else "#E8B4B5" for v in lp])
        ax.axvline(.5, c="grey", ls="--", lw=.8); ax.set_xlim(0, 1)
        ax.set_yticks(range(7), LESION_NAMES, fontsize=7.5); ax.invert_yaxis()
        ax.set_title("FINDINGS — model-detected lesions", fontsize=9); ax.set_xlabel("probability", fontsize=8)
    else:
        ax.axis("off")
    flags = [c.replace("_", " ") for c in ("referral_uncertain", "urgency_uncertain", "pdr_signs_but_graded_below_4",
                                           "lesions_but_graded_0", "vb_irma_but_graded_below_3") if bool(row.get(c, False))]
    decision = "REFER (P(grade ≥ 2) = %.3f ≥ threshold %.3f)" % (p_ref, thr) if p_ref >= thr else \
               "No referral (P(grade ≥ 2) = %.3f < threshold %.3f)" % (p_ref, thr)
    fig.text(.5, -.04, f"SUGGESTION: {decision}    |    REVIEW FLAGS: {', '.join(flags) if flags else 'none'}",
             ha="center", fontsize=9.5, weight="bold")
    fig.text(.5, -.11, "CLINICAL JUDGEMENT REQUIRED — the heatmap shows image regions that influenced the model, not confirmed "
             "lesions; a confident prediction or plausible map can still be wrong. Research prototype, not for clinical use.",
             ha="center", fontsize=8, style="italic", color="#444")
    fig.suptitle(f"{title}   ·   reference grade {int(row['grade'])}", fontsize=11, y=1.03)
    save(fig, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--n-sanity", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    run = Path(a.run); out = run / "explain"; out.mkdir(exist_ok=True)
    model, cfg = load_run(run)
    dev = next(model.parameters()).device
    fitted = json.loads((run / "eval" / "test_metrics.json").read_text())["fitted_on_validation"]
    pred = pd.read_csv(run / "eval" / "test_predictions.csv")
    pred["conf"] = pred[[f"p_cal_{k}" for k in range(5)]].max(axis=1)
    truth = pd.read_csv(a.splits).set_index("image")
    has_lesions = model.lesion_head is not None
    ds = FundusDataset(truth.loc[pred["image"]].reset_index(), a.images, cfg["data"]["normaliser"])  # same row order as pred
    cam = GradCAM(model)

    #representative cases 
    chosen = []
    for key, title, filt, col, desc in CASES:
        sel = pred[filt(pred)]
        if len(sel):
            chosen.append((key, title, sel.sort_values(col, ascending=not desc).index[0]))
    n_cols = 4 if has_lesions else 2
    fig, axes = plt.subplots(len(chosen), n_cols, figsize=(3.1 * n_cols, 3.2 * len(chosen)), squeeze=False)
    for r, (key, title, i) in enumerate(chosen):
        row = pred.loc[i]
        img, x = tensor(ds, i, dev)
        g = cam(x, "grade")
        true_les = [l for l in LESIONS if truth.at[row["image"], l]]
        axes[r, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[r, 0].set_title(f"{title}\nref {int(row.grade)} · pred {int(row.pred)} · conf {row.conf:.2f} · set {{{row.set}}}\n"
                             f"annotated: {', '.join(true_les) or 'none'}", fontsize=7.5)
        axes[r, 1].imshow(overlay(img, g)); axes[r, 1].set_title("Grad-CAM: severity", fontsize=8)
        if has_lesions:
            top = sorted(LESIONS, key=lambda l: -row[f"lesion_{l}"])[:2]
            for c, l in enumerate(top, start=2):
                axes[r, c].imshow(overlay(img, cam(x, l)))
                axes[r, c].set_title(f"Grad-CAM: {l} (p={row[f'lesion_{l}']:.2f})", fontsize=8)
        panel(img, g, row, fitted, has_lesions, out / f"panel_{key}.png", title)
        for ax in axes[r]:
            ax.axis("off")
    fig.tight_layout(); save(fig, out / "gallery.png")

    #sanity check
    cam.remove()
    rnd = randomised_copy(model, a.seed)
    cam, cam_r = GradCAM(model), GradCAM(rnd)
    idx = np.random.default_rng(a.seed).choice(len(pred), min(a.n_sanity, len(pred)), replace=False)
    sims, examples = [], []
    for i in idx:
        img, x = tensor(ds, int(i), dev)
        m1, m2 = cam(x, "grade"), cam_r(x, "grade")
        sims.append(map_similarity(m1, m2))
        if len(examples) < 4:
            examples.append((img, m1, m2))
    fig, axes = plt.subplots(3, len(examples), figsize=(3 * len(examples), 9), squeeze=False)
    for c, (img, m1, m2) in enumerate(examples):
        axes[0, c].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); axes[1, c].imshow(overlay(img, m1)); axes[2, c].imshow(overlay(img, m2))
    for r_, t in enumerate(["image", "trained model", "top layers randomised"]):
        axes[r_, 0].set_ylabel(t)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Saliency sanity check: mean Spearman ρ (trained vs randomised) = {np.mean(sims):.2f}")
    save(fig, out / "sanity_check.png")
    save_json({"n_images": len(sims), "spearman_mean": float(np.mean(sims)), "spearman_sd": float(np.std(sims)),
               "interpretation": "low ρ ⇒ maps depend on learned weights (passes); high ρ ⇒ maps reflect input/architecture only"},
              out / "sanity_check.json")
    print(f"cases: {[k for k, _, _ in chosen]} | sanity ρ = {np.mean(sims):.2f} ± {np.std(sims):.2f}")


if __name__ == "__main__":
    main()
