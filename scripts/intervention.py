
#simulate a clinician correcting model d's lesion findings outputs go to <run>/intervention/


from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mmrdr import LESIONS  
from mmrdr.metrics import grading_metrics, p_at_least, predict  
from mmrdr.models import grade_probs  
from mmrdr.plots import plt, save  
from mmrdr.utils import save_json  


class ConceptHead:
    #The bottleneck's linear grade head

    def __init__(self, ckpt: Path, head: str, temperature: float):
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
        self.W = sd["grade_head.weight"].float().numpy()            
        self.b = sd["grade_head.bias"].float().numpy()
        self.head, self.T = head, temperature

    def logits(self, concepts: np.ndarray) -> np.ndarray:
        return concepts @ self.W.T + self.b

    def probs(self, concepts: np.ndarray) -> np.ndarray:
        return grade_probs(torch.as_tensor(self.logits(concepts), dtype=torch.float32), self.head, self.T).numpy()


def correct(c: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = c.copy(); out[mask] = truth[mask]; return out


def uncertain_mask(c: np.ndarray, k: int, rows: np.ndarray | None = None) -> np.ndarray:
    """Select the k lesions per image whose probability is closest to 0.5 (most uncertain)."""
    m = np.zeros_like(c, dtype=bool)
    if k == 0:
        return m
    idx = np.argsort(np.abs(c - 0.5), axis=1)[:, :k]
    np.put_along_axis(m, idx, True, axis=1)
    if rows is not None:
        m[~rows] = False
    return m


def summarise(name: str, k: int, y, p, rule, thr, n_edits: int, n_images: int) -> dict:
    """Two distinct clinical decisions are reported and named explicitly:
      *_graded_le_mild   — point prediction ≤ 1 for a reference grade ≥ 2 (or ≥ 3): grading error
      *_missed_at_thr    — P(grade ≥ k) below the validation-fitted referral threshold: referral decision
    `missed_referable` / `missed_sight_threatening` keep their original (point-prediction) meaning."""
    m = grading_metrics(y, p, rule, thr)
    pred = predict(p, rule)
    ref_thr = p_at_least(p, 2) >= thr["referable"]
    stdr_thr = p_at_least(p, 3) >= thr["sight_threatening"]
    return {"policy": name, "k": k, "qwk": m["qwk"], "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
            "referable_sens": m["referable"]["sensitivity"], "referable_spec": m["referable"]["specificity"],
            "missed_referable": int(((y >= 2) & (pred <= 1)).sum()), "missed_sight_threatening": int(((y >= 3) & (pred <= 1)).sum()),
            "referable_missed_at_thr": int(((y >= 2) & ~ref_thr).sum()),
            "sight_threatening_missed_at_thr": int(((y >= 3) & ~stdr_thr).sum()),
            "lesion_edits": n_edits, "images_reviewed": n_images}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a bottleneck run (model.bottleneck: true)")
    ap.add_argument("--splits", required=True)
    ap.add_argument("--seeds", type=int, default=20)
    a = ap.parse_args()

    run = Path(a.run); out = run / "intervention"; out.mkdir(exist_ok=True)
    import yaml
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    if not cfg["model"].get("bottleneck"):
        raise SystemExit("intervention requires a concept-bottleneck run (model.bottleneck: true)")
    fitted = json.loads((run / "eval" / "test_metrics.json").read_text())["fitted_on_validation"]
    rule, thr = fitted["decision_rule"], fitted["thresholds"]
    head = ConceptHead(run / "best.pt", cfg["model"]["grade_head"], fitted["temperature"])

    pred = pd.read_csv(run / "eval" / "test_predictions.csv")
    truth = pd.read_csv(a.splits).set_index("image").loc[pred["image"], list(LESIONS)].to_numpy(np.float32)
    c = pred[[f"lesion_{l}" for l in LESIONS]].to_numpy(np.float32)
    y = pred["grade"].to_numpy()
    flagged = pred["any_flag"].to_numpy(bool) if "any_flag" in pred else np.zeros(len(y), bool)

    # Consistency check
    gap = np.abs(head.logits(c) - pred.filter(like="logit_").to_numpy()).max()
    print(f"max |logit| reconstruction gap: {gap:.3f}")

    rows = [summarise("model only", 0, y, head.probs(c), rule, thr, 0, 0)]
    for k in range(1, len(LESIONS) + 1):
        m = uncertain_mask(c, k)
        rows.append(summarise("uncertainty-guided", k, y, head.probs(correct(c, truth, m)), rule, thr, int(m.sum()), len(y)))
        rnd = []
        for s in range(a.seeds):
            r = np.random.default_rng(s).random(c.shape)
            mr = np.zeros_like(m); np.put_along_axis(mr, np.argsort(r, 1)[:, :k], True, axis=1)
            rnd.append(summarise("random", k, y, head.probs(correct(c, truth, mr)), rule, thr, int(mr.sum()), len(y)))
        rows.append({**{key: float(np.mean([d[key] for d in rnd])) for key in rnd[0] if key not in ("policy",)},
                     "policy": "random"})
        mf = uncertain_mask(c, k, flagged)
        rows.append(summarise("flagged-only", k, y, head.probs(correct(c, truth, mf)), rule, thr, int(mf.sum()), int(flagged.sum())))
    # Directional corrections (all 7 findings), on every image and on flagged images only
    add_all, rem_all = np.maximum(c, truth), np.minimum(c, truth)
    add_flag, rem_flag = c.copy(), c.copy()
    add_flag[flagged], rem_flag[flagged] = add_all[flagged], rem_all[flagged]
    for name, cc, n_img in [("add-only (all images)", add_all, len(y)), ("remove-only (all images)", rem_all, len(y)),
                            ("add-only (flagged)", add_flag, int(flagged.sum())),
                            ("remove-only (flagged)", rem_flag, int(flagged.sum()))]:
        rows.append(summarise(name, len(LESIONS), y, head.probs(cc), rule, thr, int((np.abs(cc - c) > 1e-6).sum()), n_img))
    df = pd.DataFrame(rows)
    df.loc[df.policy == "uncertainty-guided", "policy"] = "uncertainty-guided"
    df.round(4).to_csv(out / "intervention.csv", index=False)

    base = df.iloc[0]
    allfix = df[(df.policy == "uncertainty-guided") & (df.k == len(LESIONS))].iloc[0]
    flag2 = df[(df.policy == "flagged-only") & (df.k == 2)].iloc[0]
    directional = {r_["policy"]: r_ for r_ in df[df.policy.str.contains("only \\(", regex=True)].to_dict("records")}
    summary = {"model_only": base.to_dict(), "all_lesions_corrected": allfix.to_dict(), "directional": directional,
               "flagged_only_top2": flag2.to_dict(), "flag_rate": float(flagged.mean()),
               "reconstruction_gap": float(gap)}
    save_json(summary, out / "intervention.json")

    #figure 1: QWK, grading errors and referral-decision misses vs number of corrected lesions 
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    styles = {"uncertainty-guided": ("#4C72B0", "-"), "random": ("#8C8C8C", "--"), "flagged-only": ("#C44E52", "-")}
    panels = [("qwk", "test QWK"),
              ("missed_referable", "grading error: referable\n(ref ≥ 2) graded ≤ Mild"),
              ("referable_missed_at_thr", "referral decision: referable\nbelow the validation referral threshold")]
    for pol, (col, ls) in styles.items():
        d = pd.concat([df.iloc[[0]], df[df.policy == pol]])
        for ax, (key, _) in zip(axes, panels):
            ax.plot(d.k, d[key], ls, marker="o", c=col, label=pol)
    for ax, (_, label) in zip(axes, panels):
        ax.set_ylabel(label); ax.set_xlabel("lesion findings corrected per image (k)"); ax.set_xticks(range(len(LESIONS) + 1))
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Clinician intervention on concept-bottleneck model (flagged-only: {flagged.mean():.0%} of images reviewed)")
    fig.tight_layout()
    save(fig, out / "intervention_curve.png")

    #figure 2: the learned concept to grade rules (fully transparent model) 
    W = head.W
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    lim = np.abs(W).max()
    im = ax.imshow(W, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    for (i, j), v in np.ndenumerate(W):
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(LESIONS)), LESIONS)
    labels = [f"grade > {k}" for k in range(W.shape[0])] if cfg["model"]["grade_head"] == "corn" else [f"grade {k}" for k in range(W.shape[0])]
    ax.set_yticks(range(W.shape[0]), labels)
    ax.set_title("Model D: weight of each lesion on each severity decision"); fig.colorbar(im, fraction=.04)
    save(fig, out / "concept_weights.png")

    #figure 3: worked examples worst under-graded images. the clinician reviews all 7 findings 
    p0 = head.probs(c); pred0 = predict(p0, rule)
    cand = np.flatnonzero((y >= 2) & (pred0 <= 1))
    cand = cand[np.argsort(-(y[cand] - pred0[cand]))][:3]
    for n, i in enumerate(cand):
        c1 = truth[i].copy()                                         # clinician confirms / corrects every finding
        m = np.abs(c1 - c[i]) > 0.5                                  
        p1 = head.probs(c1[None])[0]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"width_ratios": [1.4, 1]})
        xs = np.arange(len(LESIONS)); w = .38
        axes[0].bar(xs - w / 2, c[i], w, color="#C9D3E3", label="model")
        axes[0].bar(xs + w / 2, c1, w, color=["#C44E52" if m[j] else "#4C72B0" for j in range(len(LESIONS))],
                    label="after clinician review (red = changed)")
        axes[0].set_xticks(xs, LESIONS, fontsize=8)
        axes[0].set_ylim(0, 1.05); axes[0].set_ylabel("lesion probability"); axes[0].legend(frameon=False, fontsize=8)
        axes[1].bar(np.arange(5) - w / 2, p0[i], w, color="#C9D3E3", label=f"before: grade {pred0[i]}")
        axes[1].bar(np.arange(5) + w / 2, p1, w, color="#4C72B0", label=f"after: grade {predict(p1[None], rule)[0]}")
        axes[1].set_xticks(range(5), ["0", "1", "2", "3", "4"]); axes[1].set_xlabel("grade"); axes[1].legend(frameon=False, fontsize=8)
        fig.suptitle(f"{pred.image[i]} · reference grade {y[i]} · annotated: "
                     f"{', '.join(l for l, t in zip(LESIONS, truth[i]) if t) or 'none'}", fontsize=9)
        save(fig, out / f"example_{n}.png")

    print("missed referable / STDR = point prediction ≤ Mild; 'at threshold' = below the validation referral threshold")
    print(f"model only:       QWK {base.qwk:.3f} acc {base.accuracy:.3f} missed referable {int(base.missed_referable)} "
          f"missed STDR {int(base.missed_sight_threatening)} | at threshold: referable {int(base.referable_missed_at_thr)}, "
          f"STDR {int(base.sight_threatening_missed_at_thr)}")
    print(f"flagged-only k=2: QWK {flag2.qwk:.3f} acc {flag2.accuracy:.3f} missed referable {int(flag2.missed_referable)} "
          f"missed STDR {int(flag2.missed_sight_threatening)} | {int(flag2.lesion_edits)} edits on {int(flag2.images_reviewed)} images")
    print(f"all corrected:    QWK {allfix.qwk:.3f} acc {allfix.accuracy:.3f} missed referable {int(allfix.missed_referable)} "
          f"missed STDR {int(allfix.missed_sight_threatening)}")
    for name, r_ in directional.items():
        print(f"{name:25s} QWK {r_['qwk']:.3f} acc {r_['accuracy']:.3f} missed referable {int(r_['missed_referable'])} "
              f"missed STDR {int(r_['missed_sight_threatening'])} | at threshold: referable {int(r_['referable_missed_at_thr'])}, "
              f"STDR {int(r_['sight_threatening_missed_at_thr'])} | {int(r_['lesion_edits'])} edits")


if __name__ == "__main__":
    main()
