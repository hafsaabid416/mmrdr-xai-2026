from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mmrdr import LESIONS  
from mmrdr import plots  
from mmrdr.calibration import fit_conformal, fit_temperature, prediction_sets, reliability_curve, set_summary  
from mmrdr.flags import flag_report, review_flags  
from mmrdr.inference import lesions_of, load_run, logits_of, predict_frame  
from mmrdr.metrics import (bootstrap_ci, expected_calibration_error, grading_metrics, p_at_least,  
                           predict, threshold_for_sensitivity)
from mmrdr.models import grade_probs  
from mmrdr.utils import save_json  


def probs_at(logits: np.ndarray, head: str, t: float) -> np.ndarray:
    return grade_probs(torch.as_tensor(logits, dtype=torch.float32), head, t).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--alpha", type=float, default=0.10, help="conformal miscoverage (0.10 → 90 %% sets)")
    ap.add_argument("--sens-target", type=float, default=0.90, help="validation sensitivity for operating points")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    run = Path(a.run); out = run / "eval"; out.mkdir(exist_ok=True)
    model, cfg = load_run(run)
    head = cfg["model"]["grade_head"]
    splits = pd.read_csv(a.splits)

    
    val = pd.read_csv(run / "val_predictions.csv")
    yv, lv = val["grade"].to_numpy(), logits_of(val)
    T = fit_temperature(lv, yv, head)
    pv = probs_at(lv, head, T)
    rule = max(("argmax", "median"), key=lambda r: cohen_kappa_score(yv, predict(pv, r), weights="quadratic"))
    thr = {"referable": threshold_for_sensitivity((yv >= 2).astype(int), p_at_least(pv, 2), a.sens_target),
           "sight_threatening": threshold_for_sensitivity((yv >= 3).astype(int), p_at_least(pv, 3), a.sens_target)}
    qhat = fit_conformal(pv, yv, a.alpha, class_conditional=True)
    fitted = {"temperature": T, "decision_rule": rule, "rule_val_qwk": {
                  r: float(cohen_kappa_score(yv, predict(pv, r), weights="quadratic")) for r in ("argmax", "median")},
              "thresholds": thr, "sens_target": a.sens_target, "conformal_alpha": a.alpha,
              "conformal_qhat_per_grade": qhat.tolist()}

   
    test_df = splits[splits.split == "test"]
    pred = predict_frame(model, cfg, test_df, a.images, num_workers=a.workers)
    y, lt, les = pred["grade"].to_numpy(), logits_of(pred), lesions_of(pred)
    p_raw, p = probs_at(lt, head, 1.0), probs_at(lt, head, T)
    yhat = predict(p, rule)

    metrics = grading_metrics(y, p, rule, thr)
    metrics["default_operating_point"] = {k: grading_metrics(y, p, rule)[k] for k in ("referable", "sight_threatening")}
    metrics["ece_uncalibrated"] = expected_calibration_error(p_raw, y)
    metrics["ece_calibrated"] = expected_calibration_error(p, y)

    def m_qwk(yy, pp): return cohen_kappa_score(yy, predict(pp, rule), weights="quadratic")
    def m_f1(yy, pp): return f1_score(yy, predict(pp, rule), average="macro", labels=range(5), zero_division=0)
    def m_acc(yy, pp): return float((predict(pp, rule) == yy).mean())
    def sens(k, t): return lambda yy, pp: float((p_at_least(pp, k)[yy >= k] >= t).mean()) if (yy >= k).any() else np.nan
    def spec(k, t): return lambda yy, pp: float((p_at_least(pp, k)[yy < k] < t).mean())
    ci = {"qwk": m_qwk, "macro_f1": m_f1, "accuracy": m_acc,
          "referable_sensitivity": sens(2, thr["referable"]), "referable_specificity": spec(2, thr["referable"]),
          "sight_threatening_sensitivity": sens(3, thr["sight_threatening"]),
          "sight_threatening_specificity": spec(3, thr["sight_threatening"])}
    metrics["ci95"] = {k: bootstrap_ci(f, y, p, a.n_boot) for k, f in ci.items()}

    sets = prediction_sets(p, qhat, yhat)
    metrics["conformal"] = set_summary(sets, y)
    flags = review_flags(yhat, sets, les)
    metrics["review_flags"] = flag_report(flags, y, yhat)

    order = np.argsort(-p.max(1)); cov, accs, qwks = [], [], []
    for frac in np.linspace(1.0, 0.5, 26):                      
        keep = order[: int(round(frac * len(y)))]
        cov.append(frac); accs.append(float((yhat[keep] == y[keep]).mean()))
        qwks.append(float(cohen_kappa_score(y[keep], yhat[keep], weights="quadratic")))
    metrics["selective"] = {"coverage": cov, "accuracy": accs, "qwk": qwks}

    if les is not None:                                           
        truth = test_df.set_index("image").loc[pred["image"], list(LESIONS)].to_numpy()
        metrics["lesion_auroc"] = {l: float(roc_auc_score(truth[:, j], les[:, j])) if 0 < truth[:, j].sum() < len(truth)
                                   else None for j, l in enumerate(LESIONS)}

    #save
    for k in range(5):
        pred[f"p_cal_{k}"] = p[:, k]
    pred["pred"] = yhat
    pred["set"] = ["".join(str(k) for k in np.flatnonzero(s)) for s in sets]
    pred = pd.concat([pred, flags.reset_index(drop=True)], axis=1)
    pred.to_csv(out / "test_predictions.csv", index=False)
    pred[pred.pred < pred.grade].sort_values(["grade", "pred"], ascending=[False, True]).to_csv(
        out / "under_graded.csv", index=False)
    save_json({"fitted_on_validation": fitted, "test": metrics}, out / "test_metrics.json")

    plots.confusion(metrics["confusion_matrix"], out / "confusion_matrix.png", f"{run.name} — test")
    plots.reliability(reliability_curve(p_raw, y), reliability_curve(p, y),
                      metrics["ece_uncalibrated"], metrics["ece_calibrated"], out / "reliability.png")
    plots.selective(np.array(cov), np.array(accs), np.array(qwks), out / "selective_prediction.png")

    c = metrics["ci95"]
    print(f"{run.name}: T={T:.2f} rule={rule} | test QWK {c['qwk'][0]:.3f} [{c['qwk'][1]:.3f}, {c['qwk'][2]:.3f}] "
          f"macro-F1 {c['macro_f1'][0]:.3f} acc {c['accuracy'][0]:.3f} | referable sens/spec "
          f"{c['referable_sensitivity'][0]:.3f}/{c['referable_specificity'][0]:.3f} | ECE {metrics['ece_uncalibrated']:.3f}"
          f"→{metrics['ece_calibrated']:.3f} | 90% sets: coverage {metrics['conformal']['coverage']:.3f}, "
          f"size {metrics['conformal']['mean_set_size']:.2f} | flagged {metrics['review_flags']['any_flag']['rate']:.1%}")


if __name__ == "__main__":
    main()
