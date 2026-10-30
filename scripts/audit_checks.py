#sanity checks on saved predictions  #run after evaluate.py and intervention.py. Outputs go to <runs>/audit/


from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mmrdr.calibration import prediction_sets  
from mmrdr.metrics import p_at_least, predict  
from mmrdr.utils import save_json  

UNCERTAINTY_FLAGS = ["referral_uncertain", "urgency_uncertain"]
LESION_FLAGS = ["pdr_signs_but_graded_below_4", "lesions_but_graded_0", "vb_irma_but_graded_below_3"]


def load(run: Path):
    fitted = json.loads((run / "eval" / "test_metrics.json").read_text())
    pred = pd.read_csv(run / "eval" / "test_predictions.csv")
    p = pred[[f"p_cal_{k}" for k in range(5)]].to_numpy()
    return fitted["fitted_on_validation"], fitted["test"], pred, p


def check_consistency(run: Path, tol: int = 1) -> dict:
    fit, test, pred, p = load(run)
    y, thr, rule = pred["grade"].to_numpy(), fit["thresholds"], fit["decision_rule"]
    fn_ref = int(((y >= 2) & (p_at_least(p, 2) < thr["referable"])).sum())
    fn_stdr = int(((y >= 3) & (p_at_least(p, 3) < thr["sight_threatening"])).sum())
    # the same counts implied by the sensitivities evaluate.py reported
    fn_ref_eval = int(round((y >= 2).sum() * (1 - test["referable"]["sensitivity"])))
    fn_stdr_eval = int(round((y >= 3).sum() * (1 - test["sight_threatening"]["sensitivity"])))
    out = {"eval_qwk": test["qwk"], "recomputed_qwk": float(cohen_kappa_score(y, predict(p, rule), weights="quadratic")),
           "referable_fn_recomputed": fn_ref, "referable_fn_from_eval_sensitivity": fn_ref_eval,
           "stdr_fn_recomputed": fn_stdr, "stdr_fn_from_eval_sensitivity": fn_stdr_eval}
    iv = run / "intervention" / "intervention.csv"
    if iv.exists():
        base = pd.read_csv(iv).iloc[0]
        out |= {"intervention_qwk": float(base.qwk), "intervention_referable_missed_at_thr": int(base.referable_missed_at_thr),
                "intervention_stdr_missed_at_thr": int(base.sight_threatening_missed_at_thr)}
        
        assert abs(out["intervention_referable_missed_at_thr"] - fn_ref) <= tol, out
        assert abs(out["intervention_stdr_missed_at_thr"] - fn_stdr) <= tol, out
        assert abs(out["intervention_qwk"] - test["qwk"]) < 0.005, out
    assert fn_ref == fn_ref_eval and fn_stdr == fn_stdr_eval, out
    assert abs(out["recomputed_qwk"] - out["eval_qwk"]) < 1e-6, out
    out["status"] = "consistent"
    return out


def coverage_per_grade(run: Path) -> pd.DataFrame:
    fit, _, pred, p = load(run)
    y = pred["grade"].to_numpy()
    q = np.array(fit["conformal_qhat_per_grade"])
    yhat = predict(p, fit["decision_rule"])
    std, shown = prediction_sets(p, q), prediction_sets(p, q, yhat)
    rows = []
    for g in range(5):
        m = y == g
        rows.append({"run": run.name, "grade": g, "n_test": int(m.sum()),
                     "coverage_standard": float(std[m, g].mean()), "coverage_displayed": float(shown[m, g].mean()),
                     "mean_set_size_displayed": float(shown[m].sum(1).mean())})
    rows.append({"run": run.name, "grade": "all", "n_test": len(y), "coverage_standard": float(std[np.arange(len(y)), y].mean()),
                 "coverage_displayed": float(shown[np.arange(len(y)), y].mean()), "mean_set_size_displayed": float(shown.sum(1).mean())})
    return pd.DataFrame(rows)


def flag_contributions(run: Path) -> pd.DataFrame:
    fit, _, pred, p = load(run)
    y, thr = pred["grade"].to_numpy(), fit["thresholds"]
    yhat = pred["pred"].to_numpy()
    targets = {"any error": yhat != y,
               "referral missed at threshold": (y >= 2) & (p_at_least(p, 2) < thr["referable"]),
               "urgent missed at threshold": (y >= 3) & (p_at_least(p, 3) < thr["sight_threatening"]),
               "referable graded ≤ Mild": (y >= 2) & (yhat <= 1)}
    lesion_cols = [c for c in LESION_FLAGS if c in pred]
    groups = {c: pred[c].to_numpy(bool) for c in UNCERTAINTY_FLAGS + lesion_cols}
    if lesion_cols:
        groups["lesion–grade inconsistency (any)"] = pred[lesion_cols].to_numpy(bool).any(1)
    groups["all flags combined"] = pred["any_flag"].to_numpy(bool)
    unc = pred[UNCERTAINTY_FLAGS].to_numpy(bool).any(1)
    les = pred[lesion_cols].to_numpy(bool).any(1) if lesion_cols else np.zeros(len(y), bool)
    n_review = int(groups["all flags combined"].sum())
    conf_only = np.zeros(len(y), bool)
    conf_only[np.argsort(p.max(1))[:n_review]] = True          
    groups["confidence only (same workload)"] = conf_only
    rows = []
    for name, m in groups.items():
        r = {"run": run.name, "flag": name, "images_reviewed": int(m.sum()), "review_rate": float(m.mean())}
        for t, hit in targets.items():
            r[f"{t}: caught"] = int((hit & m).sum())
            r[f"{t}: total"] = int(hit.sum())
        if name == "lesion–grade inconsistency (any)":
            r["errors caught only by this type"] = int(((yhat != y) & les & ~unc).sum())
        if name in UNCERTAINTY_FLAGS:
            r["errors caught only by this type"] = int(((yhat != y) & m & ~les & ~pred[[c for c in UNCERTAINTY_FLAGS if c != name]].to_numpy(bool).any(1)).sum())
        rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--bottleneck", default=None, help="run with intervention outputs to cross-check (A)")
    a = ap.parse_args()
    runs = Path(a.runs); out = runs / "audit"; out.mkdir(exist_ok=True)

    cons = {m: check_consistency(runs / m) for m in a.models}
    save_json(cons, out / "consistency.json")
    print("A · consistency:", {m: c["status"] for m, c in cons.items()})
    if a.bottleneck:
        c = cons[a.bottleneck]
        print(f"    {a.bottleneck}: referral FN eval/recomputed/intervention = {c['referable_fn_from_eval_sensitivity']}/"
              f"{c['referable_fn_recomputed']}/{c.get('intervention_referable_missed_at_thr')}; urgent FN = "
              f"{c['stdr_fn_from_eval_sensitivity']}/{c['stdr_fn_recomputed']}/{c.get('intervention_stdr_missed_at_thr')}; "
              f"QWK eval/intervention = {c['eval_qwk']:.3f}/{c.get('intervention_qwk', float('nan')):.3f}")

    cov = pd.concat([coverage_per_grade(runs / m) for m in a.models], ignore_index=True)
    cov.round(3).to_csv(out / "coverage_per_grade.csv", index=False)
    print("\nB · conformal coverage per grade (target 0.90):")
    print(cov.pivot(index="grade", columns="run", values="coverage_standard").round(3).to_string())

    fl = pd.concat([flag_contributions(runs / m) for m in a.models], ignore_index=True)
    fl.to_csv(out / "flag_contributions.csv", index=False)
    print("\nC · flag contributions (exploratory — rules refined during the study):")
    show = ["flag", "images_reviewed", "any error: caught", "referral missed at threshold: caught",
            "urgent missed at threshold: caught", "errors caught only by this type"]
    for m in a.models:
        d = fl[fl.run == m]
        tot = d.iloc[0]
        print(f"\n{m}  (totals: errors {tot['any error: total']}, referral misses {tot['referral missed at threshold: total']}, "
              f"urgent misses {tot['urgent missed at threshold: total']})")
        print(d[show].to_string(index=False))


if __name__ == "__main__":
    main()
