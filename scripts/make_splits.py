"""Build the train/val/test split used by all experiments (saved as data/splits.csv).

Input: cfp_eda.csv from notebooks/01_data_exploration.ipynb (per-image labels and duplicate groups).

- The official test set is kept as is.
- Training images that duplicate a test image are removed.
- Within training, one image per duplicate group is kept; groups with conflicting grades are dropped.
- Validation is a stratified, group-aware 1/7 of the remaining training images.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mmrdr import LESIONS  
from mmrdr.utils import save_json  


def make_splits(eda: pd.DataFrame, n_folds: int = 7, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = eda.copy()
    group_has_test = df.groupby("dup_group")["split"].transform(lambda s: (s == "test").any())
    is_train = df["split"] == "train"

    excluded = []
    leak = is_train & group_has_test
    excluded.append(df[leak].assign(reason="duplicate of a test image"))

    pool = df[is_train & ~leak]
    conflict = pool.groupby("dup_group")["grade"].transform("nunique") > 1
    excluded.append(pool[conflict].assign(reason="duplicate group with conflicting grades"))
    pool = pool[~conflict]
    redundant = pool.duplicated("dup_group", keep="first")
    excluded.append(pool[redundant].assign(reason="redundant copy within training"))
    pool = pool[~redundant]

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    _, val_idx = next(sgkf.split(pool, pool["grade"], groups=pool["dup_group"]))
    pool = pool.assign(split="train")
    pool.iloc[val_idx, pool.columns.get_loc("split")] = "val"

    test = df[df["split"] == "test"].copy()
    test["test_dup_within_test"] = test.duplicated("dup_group", keep=False)
    test["test_had_train_copy"] = test["dup_group"].isin(df.loc[leak, "dup_group"])

    cols = ["image", "split", "grade", "lr", *LESIONS, "dup_group"]
    splits = pd.concat([pool[cols], test[cols + ["test_dup_within_test", "test_had_train_copy"]]],
                       ignore_index=True)
    splits[["test_dup_within_test", "test_had_train_copy"]] = (
        splits[["test_dup_within_test", "test_had_train_copy"]].fillna(False).astype(bool))
    return splits, pd.concat(excluded, ignore_index=True)[["image", "split", "grade", "dup_group", "reason"]]


def check_no_leakage(splits: pd.DataFrame) -> None:
    spans = splits.groupby("dup_group")["split"].nunique()
    bad = spans[spans > 1]
    assert bad.empty, f"duplicate groups span splits: {bad.index.tolist()[:10]}"
    assert splits["image"].is_unique


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eda", required=True, help="cfp_eda.csv from the EDA notebook")
    ap.add_argument("--out", required=True, help="output splits.csv")
    ap.add_argument("--folds", type=int, default=7)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    splits, excluded = make_splits(pd.read_csv(a.eda), a.folds, a.seed)
    check_no_leakage(splits)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(out, index=False)
    excluded.to_csv(out.with_name("excluded_images.csv"), index=False)

    table = pd.crosstab(splits["grade"], splits["split"])[["train", "val", "test"]]
    summary = {
        "counts": {s: int((splits.split == s).sum()) for s in ("train", "val", "test")},
        "grade_by_split": table.to_dict(),
        "excluded": excluded["reason"].value_counts().to_dict(),
        "test_images_with_train_copy_removed": int(splits["test_had_train_copy"].sum()),
        "test_images_duplicated_within_test": int(splits["test_dup_within_test"].sum()),
        "val_fraction_of_pool": round(float((splits.split == "val").sum() /
                                           splits.split.isin(["train", "val"]).sum()), 4),
        "seed": a.seed,
    }
    save_json(summary, out.with_name("splits_summary.json"))
    print(table.assign(total=table.sum(axis=1)))
    print({k: v for k, v in summary.items() if k != "grade_by_split"})


if __name__ == "__main__":
    main()
