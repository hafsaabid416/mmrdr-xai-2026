#fast unit tests:preprocessing, CORN maths, metrics and leakage-safe splitting.


import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

from mmrdr import LESIONS  
from mmrdr.losses import corn_loss  
from mmrdr.metrics import grading_metrics, predict  
from mmrdr.models import DRNet, grade_probs 
from mmrdr.preprocessing import fov_mask, preprocess 
from make_splits import check_no_leakage, make_splits  


def test_preprocess_shape_and_black_background():
    img = np.full((300, 400, 3), 30, np.uint8)            
    out = preprocess(img, 128)
    assert out.shape == (128, 128, 3)
    assert out[0, 0].sum() == 0 and out[-1, -1].sum() == 0


def test_fov_mask_is_centred_circle():
    m = fov_mask(np.zeros((200, 200, 3), np.uint8))
    assert m[100, 100] and not m[0, 0]
    assert abs(m.mean() - np.pi / 4) < 0.02


def test_corn_probs_valid_and_rank_consistent():
    p = grade_probs(torch.randn(64, 4) * 3, "corn")
    assert torch.allclose(p.sum(1), torch.ones(64), atol=1e-5) and (p >= 0).all()


def test_corn_loss_decreases_for_correct_logits():
    y = torch.tensor([0, 1, 2, 3, 4])
    good = torch.tensor([[-9, -9, -9, -9], [9, -9, -9, -9], [9, 9, -9, -9], [9, 9, 9, -9], [9, 9, 9, 9]], dtype=torch.float)
    assert corn_loss(good, y) < 1e-3 < corn_loss(-good, y)


@pytest.mark.parametrize("head,lesion,bottleneck", [("softmax", False, False), ("corn", False, False),
                                                     ("corn", True, False), ("corn", True, True)])
def test_model_variants_forward(head, lesion, bottleneck):
    m = DRNet("resnet50", pretrained=False, grade_head=head, lesion_head=lesion, bottleneck=bottleneck).eval()
    out = m(torch.randn(2, 3, 64, 64))
    assert out["grade_logits"].shape == (2, 5 if head == "softmax" else 4)
    assert ("lesion_logits" in out) == lesion


def test_metrics_perfect_and_median_rule():
    y = np.array([0, 1, 2, 3, 4]); probs = np.eye(5)
    m = grading_metrics(y, probs)
    assert m["accuracy"] == 1 and m["qwk"] == pytest.approx(1) and m["referable"]["sensitivity"] == 1
    assert predict(np.array([[0.3, 0.3, 0.4, 0, 0]]), "median")[0] == 1


def test_splits_have_no_leakage():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({"image": [f"img/{'tr' if i < 320 else 'ts'}{i:06d}.jpg" for i in range(n)],
                       "split": ["train"] * 320 + ["test"] * 80, "grade": rng.integers(0, 5, n),
                       "lr": rng.integers(0, 2, n), "dup_group": np.arange(n)})
    for l in LESIONS:
        df[l] = rng.integers(0, 2, n)
    df.loc[[0, 330], "dup_group"] = 330                                 
    df.loc[[1, 2], "dup_group"] = 2; df.loc[[1, 2], "grade"] = 3          
    df.loc[[3, 4], "dup_group"] = 4; df.loc[3, "grade"] = 0; df.loc[4, "grade"] = 4   # conflicting grades
    splits, excl = make_splits(df)
    check_no_leakage(splits)
    assert "img/tr000000.jpg" not in set(splits.image)
    assert set(excl.reason) == {"duplicate of a test image", "redundant copy within training",
                                "duplicate group with conflicting grades"}
    assert (splits.split == "test").sum() == 80


def test_temperature_scaling_reduces_nll_for_overconfident_logits():
    from mmrdr.calibration import fit_temperature, nll
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 500)
    logits = rng.normal(0, 1, (500, 5)); logits[np.arange(500), y] += 1.0
    logits *= 6                                                    
    t = fit_temperature(logits, y, "softmax")
    assert t > 1 and nll(logits, y, "softmax", t) < nll(logits, y, "softmax", 1.0)


def test_class_conditional_conformal_covers_each_grade():
    from mmrdr.calibration import fit_conformal, prediction_sets
    rng = np.random.default_rng(1)
    y = rng.integers(0, 5, 4000)
    p = rng.dirichlet(np.ones(5), 4000); p[np.arange(4000), y] += 0.5; p /= p.sum(1, keepdims=True)
    q = fit_conformal(p[:2000], y[:2000], alpha=0.1, class_conditional=True)
    sets = prediction_sets(p[2000:], q)
    yt = y[2000:]
    for c in range(5):
        assert sets[yt == c, c].mean() > 0.85                       #finite-sample noise


def test_threshold_meets_sensitivity_target():
    from mmrdr.metrics import threshold_for_sensitivity
    rng = np.random.default_rng(2)
    yb = rng.integers(0, 2, 1000); s = np.clip(yb * 0.3 + rng.random(1000) * 0.7, 0, 1)
    t = threshold_for_sensitivity(yb, s, 0.9)
    assert (s[yb == 1] >= t).mean() >= 0.9


def test_sets_include_point_prediction_and_flags_use_management_boundaries():
    from mmrdr.calibration import prediction_sets
    from mmrdr.flags import review_flags
    probs = np.array([[0.6, 0.4, 0, 0, 0], [0.1, 0.1, 0.8, 0, 0], [0.3, 0.3, 0.4, 0, 0]])
    qhat = np.full(5, 0.5)                                          
    sets = prediction_sets(probs, qhat, point=np.array([1, 2, 1]))
    assert sets[0, 1] and sets[2, 1]                                
    f = review_flags(np.array([1, 2, 1]), sets, None)
    assert not f.referral_uncertain[0]                               
    assert not f.referral_uncertain[1]                               
