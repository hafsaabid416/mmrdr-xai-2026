#review flags that route cases to a clinician 


from __future__ import annotations

import numpy as np
import pandas as pd

from . import LESIONS

_L = {name: i for i, name in enumerate(LESIONS)}


def review_flags(pred: np.ndarray, sets: np.ndarray, lesion_probs: np.ndarray | None,
                 lesion_threshold: float = 0.5) -> pd.DataFrame:
    
    f = pd.DataFrame({
        "referral_uncertain": sets[:, :2].any(1) & sets[:, 2:].any(1),
        "urgency_uncertain": sets[:, :3].any(1) & sets[:, 3:].any(1),
    })
    if lesion_probs is not None:
        pdr_signs = lesion_probs[:, [_L["NV"], _L["VH"], _L["RD"]]].max(1) > lesion_threshold
        any_lesion = lesion_probs.max(1) > lesion_threshold
        severe_signs = lesion_probs[:, _L["VB_IRMA"]] > lesion_threshold
        f["pdr_signs_but_graded_below_4"] = pdr_signs & (pred < 4)
        f["lesions_but_graded_0"] = any_lesion & (pred == 0)
        f["vb_irma_but_graded_below_3"] = severe_signs & (pred < 3)
    f["any_flag"] = f.any(axis=1)
    f["multi_grade_set"] = sets.sum(1) > 1                         
    return f


def flag_report(flags: pd.DataFrame, y: np.ndarray, pred: np.ndarray) -> dict:
    wrong = pred != y
    missed_stdr = (y >= 3) & (pred <= 1)
    under = pred < y
    out = {}
    for col in flags.columns:
        m = flags[col].to_numpy()
        out[col] = {"rate": float(m.mean()), "n": int(m.sum()),
                    "error_rate_flagged": float(wrong[m].mean()) if m.any() else None,
                    "error_rate_unflagged": float(wrong[~m].mean()) if (~m).any() else None,
                    "under_graded_caught": int((under & m).sum()), "under_graded_total": int(under.sum()),
                    "missed_sight_threatening_caught": int((missed_stdr & m).sum()),
                    "missed_sight_threatening_total": int(missed_stdr.sum())}
    return out
