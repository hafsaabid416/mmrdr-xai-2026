#shared inference utilities. load a trained run and predict a DataFrame of images

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from . import LESIONS, NUM_GRADES
from .data import FundusDataset, make_loader, to_input
from .models import DRNet, build_model, grade_probs


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_run(run_dir: str | Path, dev: torch.device | None = None) -> tuple[DRNet, dict]:
    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    cfg["model"]["pretrained"] = False                      # weights from checkpoint
    model = build_model(cfg)
    ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    dev = dev or device()
    return model.to(dev).to(memory_format=torch.channels_last).eval(), cfg


@torch.no_grad()
def predict_loader(model, loader, head: str, dev: torch.device):
    
    model.eval()
    logits, lesions, idx = [], [], []
    for x, _, _, i in loader:
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            out = model(to_input(x, dev))
        logits.append(out["grade_logits"].float().cpu())
        if "lesion_logits" in out:
            lesions.append(torch.sigmoid(out["lesion_logits"].float()).cpu())
        idx.append(i)
    gl = torch.cat(logits)
    return (gl.numpy(), grade_probs(gl, head).numpy(),
            torch.cat(lesions).numpy() if lesions else None, torch.cat(idx).numpy())


def predict_frame(model, cfg: dict, df: pd.DataFrame, image_dir: str | Path, batch_size: int = 32,
                  num_workers: int = 2, dataset: FundusDataset | None = None) -> pd.DataFrame:
    """Predictions for every row of `df`, in the original row order."""
    dev = next(model.parameters()).device
    ds = dataset or FundusDataset(df, image_dir, cfg["data"]["normaliser"])
    dl = make_loader(ds, batch_size, num_workers, shuffle=False)
    logits, probs, les, idx = predict_loader(model, dl, cfg["model"]["grade_head"], dev)
    order = np.argsort(idx)
    out = pd.DataFrame({"image": ds.df["image"].values, "grade": ds.grades})
    for k in range(logits.shape[1]):
        out[f"logit_{k}"] = logits[order, k]
    for k in range(NUM_GRADES):
        out[f"p_{k}"] = probs[order, k]
    if les is not None:
        for j, l in enumerate(LESIONS):
            out[f"lesion_{l}"] = les[order, j]
    return out


def logits_of(pred: pd.DataFrame) -> np.ndarray:
    return pred[[c for c in pred.columns if c.startswith("logit_")]].to_numpy()


def lesions_of(pred: pd.DataFrame) -> np.ndarray | None:
    cols = [f"lesion_{l}" for l in LESIONS]
    return pred[cols].to_numpy() if cols[0] in pred.columns else None
