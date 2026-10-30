# MMRDR-XAI: Explainable diabetic retinopathy grading

Research prototype for five-grade diabetic retinopathy (DR) classification from colour fundus photographs (CFP), comparing conventional classification, ordinal learning, lesion-supervised grading and an editable concept bottleneck. It also evaluates calibration, review flags, image degradation and Grad-CAM explanations. **Research only: not clinically validated and not for patient care.**

This README documents the code in this repository. Details of design decisions and references are in [`docs/DECISIONS.md`](docs/DECISIONS.md).

## 1. What is included

| Location | Contents |
|---|---|
| `configs/` | YAML settings for the shared training configuration and Models A–D |
| `src/mmrdr/` | Model, data, preprocessing, calibration, metrics and explanation code |
| `scripts/` | Data preparation, splitting, training, evaluation, robustness, explanation and intervention entry points |
| `notebooks/mmrdr_eda.ipynb` | Data exploration and duplicate analysis |
| `notebooks/mmrdr_colab_runner.ipynb` | Colab workflow using the scripts |
| `data/splits.csv` | Frozen split identifiers and labels; **no retinal images** |
| `data/excluded_images.csv`, `data/splits_summary.json` | Exclusions and split summary |
| `tests/test_core.py` | Basic implementation tests |
| `docs/DECISIONS.md` | Experiment rationale and sources |

**Not included in this ZIP:** the licensed MMRDR images, preprocessed image cache, trained `.pt` checkpoints, full run histories or explanation plots. Code for generating these outputs is included. Without trained checkpoints, testing a new retinal image requires running training first. If distributing checkpoints separately, document their download location and associated configuration before claiming a no-training test path.

## 2. Dataset

Source: Tang et al. (2026), MMRDR, *Scientific Data*. Obtain the dataset from the authors' data deposit: <https://doi.org/10.6084/m9.figshare.29423747.v2>. Check the dataset's licence and access conditions. **Do not commit the images to GitHub.**

Only the CFP subset is used (11,118 images and seven image-level lesion indicators). After extraction, the data-preparation script expects:

```text
<DATA_ROOT>/MMRDR-CFP/
  FP.csv
  img/
    ... original CFP images ...
```

`notebooks/mmrdr_eda.ipynb` documents the archive extraction and exploratory analysis. Dataset download and extraction may require substantial space and time in Colab. The model scripts do not download the dataset automatically.

The repository already includes the **frozen analysis split** at `data/splits.csv`. You do **not** need to rerun EDA to use that split. If you wish to regenerate it, run EDA to obtain `cfp_eda.csv` and use `scripts/make_splits.py`; this may take longer than the main preparation step.

## 3. Environment and installation

The experiments were developed using Google Colab with an NVIDIA T4 GPU and Python 3.12. GPU access is recommended for training; basic tests run on CPU.

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests/
```

`requirements.txt` specifies minimum package versions, **not a fully pinned environment**. Reproduction on a different PyTorch/CUDA version may yield some variation. Main settings, including `seed: 42`, batch size, optimisers and augmentation settings, are defined in `configs/base.yaml`.

## 4. Prepare image cache (no model training)

Run this from the repository root, replacing `/path/to/MMRDR-CFP` with the actual extracted CFP directory:

```bash
python scripts/prepare_images.py \
  --src /path/to/MMRDR-CFP \
  --out cache512 \
  --size 512
```

This creates the padded, resized 512×512 image cache used by training and evaluation. The original data is not modified. The cache directory is ignored by Git.

Optional split regeneration (not necessary when using the committed split):

```bash
python scripts/make_splits.py \
  --eda /path/to/cfp_eda.csv \
  --out data/splits_regenerated.csv \
  --seed 42
```

The experiment split preserves the published test partition while excluding identified duplicate or conflicting training images. Patient-level independence cannot be guaranteed because patient identifiers are unavailable.

## 5. Train the models

From the repository root, using `cache512` and the included `data/splits.csv`:

```bash
# Model A: standard preprocessing
python scripts/train.py --config configs/A_baseline.yaml \
  --splits data/splits.csv --images cache512 --out outputs/runs/A_baseline

# Model A preprocessing comparisons
python scripts/train.py --config configs/A_graham.yaml \
  --splits data/splits.csv --images cache512 --out outputs/runs/A_graham
python scripts/train.py --config configs/A_clahe.yaml \
  --splits data/splits.csv --images cache512 --out outputs/runs/A_clahe

# Model B: ordinal CORN
python scripts/train.py --config configs/B_ordinal.yaml \
  --splits data/splits.csv --images cache512 --out outputs/runs/B_ordinal_none

# Model C: ordinal grade and auxiliary lesion predictions
python scripts/train.py --config configs/C_multitask.yaml \
  --splits data/splits.csv --images cache512 --out outputs/runs/C_multitask_none

# Model D: lesion-to-grade concept bottleneck
python scripts/train.py --config configs/D_bottleneck.yaml \
  --splits data/splits.csv --images cache512 --out outputs/runs/D_bottleneck_none
```

The `_none` run names indicate standard preprocessing, which was selected from the validation preprocessing comparison. Run names are chosen by the user; the scripts do not require these exact names. The training script saves `best.pt`, `val_predictions.csv`, `val_metrics.json`, `history.csv` and a training log in the chosen run directory. Best checkpoint selection uses **validation QWK**, not test performance. The shared settings include seed 42, up to 20 epochs, AdamW and a cosine learning-rate schedule. Run `python scripts/train.py --help` for optional overrides and `--resume`.

## 6. Evaluate an existing checkpoint (no retraining)

**If `outputs/runs/C_multitask_none/best.pt` and the other training-run files are already available**, skip Section 5 and run:

```bash
python scripts/evaluate.py \
  --run outputs/runs/C_multitask_none \
  --splits data/splits.csv --images cache512
```

This fits temperature scaling, decision thresholds and conformal thresholds on the **validation** predictions and assesses performance on the held-out **test** images. Outputs are saved under `<run>/eval/`, including `test_metrics.json`, `test_predictions.csv` and evaluation figures. The default bootstrap count is 2,000; run `python scripts/evaluate.py --help` for options.

To check the complete model comparison, repeat the evaluation command with the run directories for A, B and D. Only compare runs that used the documented frozen split.

## 7. Explainability, robustness and lesion corrections

After running `evaluate.py` for Model C:

```bash
python scripts/explain.py \
  --run outputs/runs/C_multitask_none \
  --splits data/splits.csv --images cache512

python scripts/robustness.py \
  --run outputs/runs/C_multitask_none \
  --splits data/splits.csv --images cache512
```

The explanation script generates representative Grad-CAM panels and a model-randomisation sanity check under `<run>/explain/`. The robustness script tests blur, darkening and JPEG compression under `<run>/robustness/`. Explanations reflect model influence, **not verified lesion localisation**.

To reproduce the **exploratory** Model D lesion-correction experiment, first evaluate Model D, then run:

```bash
python scripts/evaluate.py \
  --run outputs/runs/D_bottleneck_none \
  --splits data/splits.csv --images cache512

python scripts/intervention.py \
  --run outputs/runs/D_bottleneck_none \
  --splits data/splits.csv
```

These interventions substitute recorded dataset lesion annotations for predicted lesions. They are **not actual clinician edits** and were explored using test results; their effectiveness requires independent confirmation.

## 8. Results to verify

Reported held-out performance (2,225 test images):

| Model | QWK | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| A | 0.873 | 0.782 | 0.673 |
| B | 0.881 | 0.760 | 0.634 |
| C | 0.883 | 0.812 | 0.685 |
| D | 0.879 | 0.791 | 0.643 |

Model C: Grade ≥2 sensitivity 0.877, Grade ≥3 sensitivity 0.953; test calibration error 0.137 before and 0.022 after temperature scaling. Model D exploratory correction of the three most uncertain lesion predictions: QWK 0.879 to 0.956, Grade ≥2 threshold misses 67 to 14, and Grade ≥3 threshold misses 23 to 61. See the technical report for confidence intervals, grading errors, explanations and limitations.

**Outputs are not bundled in the uploaded GitHub ZIP.** Regenerate them with Sections 5–7 or supply a separately authorised verification archive. Do not claim the results can be reproduced without training unless the corresponding checkpoints have been provided.

## 9. Reproducibility and limitations

- Frozen image-level train/validation/test split, stored in `data/splits.csv`; patient identity cannot be verified.
- Training seed 42; only one seed in the reported main experiments.
- Dataset contains no demographic or device metadata for subgroup evaluation.
- Calibration and threshold choices are fitted using validation data; review-flag refinements and lesion-correction analyses on test data are exploratory.
- Synthetic image degradation does not replace validation on real clinical devices.
- The work is a research prototype, **not a validated medical device**.

For the experiment rationale, consult [`docs/DECISIONS.md`](docs/DECISIONS.md). For users of the Colab workflow, see [`notebooks/mmrdr_colab_runner.ipynb`](notebooks/mmrdr_colab_runner.ipynb).
