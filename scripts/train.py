"""Train one experiment from a YAML config.

The best checkpoint is chosen on validation QWK; this script never reads the test split.

Usage:
    python scripts/train.py --config configs/C_multitask.yaml --splits data/splits.csv \
        --images cache512 --out outputs/runs/C_multitask [train.epochs=20 ...]

Writes best.pt, history.csv, val_metrics.json and train.log to --out.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mmrdr import LESIONS, NUM_GRADES  
from mmrdr.data import FundusDataset, make_loader, to_input  
from mmrdr.inference import predict_loader 
from mmrdr.losses import DRLoss, effective_number_weights  
from mmrdr.metrics import grading_metrics  
from mmrdr.models import build_model  
from mmrdr.utils import get_logger, load_config, save_json, seed_everything  


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--images", required=True, help="preprocessed 512px image cache")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force", action="store_true", help="retrain even if the run already finished")
    ap.add_argument("overrides", nargs="*", help="key=value config overrides")
    a = ap.parse_args()

    cfg = load_config(a.config, a.overrides)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if (out / "val_metrics.json").exists() and not a.force:
        print(f"{out.name}: already finished — skipping (use --force to retrain)")
        return
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    log = get_logger("train", out / "train.log")
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tc, head = cfg["train"], cfg["model"]["grade_head"]
    rule = cfg["eval"]["decision"]

    splits = pd.read_csv(a.splits)
    tr_df, va_df = splits[splits.split == "train"], splits[splits.split == "val"]
    log.info(f"torch {torch.__version__} | GPU {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'none'}")
    log.info(f"device={device} | train={len(tr_df)} val={len(va_df)} | normaliser={cfg['data']['normaliser']}")
    tr_ds = FundusDataset(tr_df, a.images, cfg["data"]["normaliser"], train=True, aug_cfg=cfg["augment"], seed=cfg["seed"])
    va_ds = FundusDataset(va_df, a.images, cfg["data"]["normaliser"])
    tr_dl = make_loader(tr_ds, tc["batch_size"], tc["num_workers"], shuffle=True, balanced_sampler=tc.get("balanced_sampler", False))
    va_dl = make_loader(va_ds, tc["batch_size"], tc["num_workers"], shuffle=False)

    counts = np.bincount(tr_ds.grades, minlength=NUM_GRADES)
    cw = effective_number_weights(counts, tc["cb_beta"]) if tc.get("class_weighting", True) else None
    pos = tr_ds.lesions.mean(0)
    pw = torch.tensor(np.clip((1 - pos) / np.maximum(pos, 1e-6), 1, tc.get("max_lesion_pos_weight", 10)) ** 0.5,
                      dtype=torch.float32)
    log.info(f"class counts={counts.tolist()} weights={None if cw is None else cw.numpy().round(3).tolist()}")

    model = build_model(cfg).to(device).to(memory_format=torch.channels_last)
    crit = DRLoss(head, cw, cfg["loss"]["lesion_weight"], pw).to(device)
    head_params = [p for n, p in model.named_parameters() if not n.startswith("features.")]
    opt = torch.optim.AdamW([{"params": model.features.parameters(), "lr": tc["lr_backbone"]},
                             {"params": head_params, "lr": tc["lr_head"]}], weight_decay=tc["weight_decay"])
    steps = tc["epochs"] * len(tr_dl); warm = tc["warmup_epochs"] * len(tr_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(warm, 1) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(steps - warm, 1))))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    start, best, bad, history = 0, -np.inf, 0, []
    if a.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"]); start, best, bad, history = ck["epoch"] + 1, ck["best"], ck["bad"], ck["history"]
        log.info(f"resumed at epoch {start}")

    for epoch in range(start, tc["epochs"]):
        model.train(); tr_ds.set_epoch(epoch)
        t0, run, n = time.time(), {}, 0
        for x, y, les, _ in tr_dl:
            x = to_input(x, device)
            y, les = y.to(device), les.to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                outp = model(x)
            parts = crit({k: v.float() for k, v in outp.items()}, y, les)
            opt.zero_grad(set_to_none=True)
            scaler.scale(parts["loss"]).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            scaler.step(opt); scaler.update(); sched.step()
            for k, v in parts.items():
                run[k] = run.get(k, 0.0) + v.item() * len(y)
            n += len(y)
        _, probs, _, _ = predict_loader(model, va_dl, head, device)
        m = grading_metrics(va_ds.grades, probs, rule)
        row = {"epoch": epoch, **{f"train_{k}": v / n for k, v in run.items()},
               "val_qwk": m["qwk"], "val_macro_f1": m["macro_f1"], "val_accuracy": m["accuracy"],
               "val_ece": m["ece"], "val_under_grading": m["under_grading_rate"],
               "lr": opt.param_groups[0]["lr"], "sec": time.time() - t0, "img_per_sec": n / (time.time() - t0)}
        history.append(row)
        score = m["qwk"] + 1e-3 * m["macro_f1"]                      
        improved = score > best
        if improved:
            best, bad = score, 0
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch}, out / "best.pt")
        else:
            bad += 1
        log.info(f"ep {epoch:02d} loss {row['train_loss']:.4f} | val QWK {m['qwk']:.4f} F1 {m['macro_f1']:.4f} "
                 f"acc {m['accuracy']:.4f} ECE {m['ece']:.3f} | {row['sec']:.0f}s {row['img_per_sec']:.0f} img/s"
                 + (" *" if improved else ""))
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best": best, "bad": bad, "history": history},
                   out / "last.pt")
        if bad >= tc["patience"]:
            log.info(f"early stopping (patience {tc['patience']})"); break

    # Validation predictions of the selected model 
    model.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=False)["model"])
    logits, probs, les_p, idx = predict_loader(model, va_dl, head, device)
    pred = pd.DataFrame({"image": va_ds.df["image"].values[idx], "grade": va_ds.grades[idx]})
    for k in range(logits.shape[1]):
        pred[f"logit_{k}"] = logits[:, k]
    for k in range(NUM_GRADES):
        pred[f"p_{k}"] = probs[:, k]
    if les_p is not None:
        for j, l in enumerate(LESIONS):
            pred[f"lesion_{l}"] = les_p[:, j]
    pred.to_csv(out / "val_predictions.csv", index=False)
    save_json(grading_metrics(va_ds.grades[idx], probs, rule), out / "val_metrics.json")
    (out / "last.pt").unlink(missing_ok=True)                        
    log.info(f"done. best val QWK {max(h['val_qwk'] for h in history):.4f}")


if __name__ == "__main__":
    main()
