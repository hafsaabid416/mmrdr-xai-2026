# MMRDR-XAI: clinically explainable diabetic retinopathy grading

A **research prototype** that grades colour fundus photographs (CFP) into the five ICDR severity levels
(0 No DR · 1 Mild NPDR · 2 Moderate NPDR · 3 Severe NPDR · 4 PDR) and reports calibrated uncertainty,
lesion-level findings and visual explanations for clinician review. It is **not** a clinically validated
diagnostic tool.

Every design choice is recorded with its evidence and literature in [`docs/DECISIONS.md`](docs/DECISIONS.md).

## Repository layout

```
configs/            experiment configs (YAML, inherit from base.yaml)
data/splits.csv     frozen leakage-safe split (committed; contains labels only, no images)
docs/               decision register, model card
notebooks/
  01_data_exploration.ipynb   EDA → preprocessing decisions
  00_colab_runner.ipynb       thin Colab wrapper around the scripts
scripts/            command-line entry points (make_splits, prepare_images, train, …)
src/mmrdr/          package: preprocessing, data, models, losses, metrics, …
tests/              unit tests (preprocessing, CORN maths, metrics, split leakage)
```

## Data

MMRDR (Tang et al., *Scientific Data* 2026) is available from figshare:
<https://doi.org/10.6084/m9.figshare.29423747.v2>. **The dataset is not redistributed here.**

1. Download all nine parts `MMRDR.zip.001`–`.009` (≈17.8 GB).
2. The split archive's central directory is unreadable by standard tools after joining; extract it with the
   streaming extractor in `notebooks/01_data_exploration.ipynb` (or any tool that reads local headers).
   Only `MMRDR-CFP/` (11,118 images + `FP.csv`) is used.

## Setup

```bash
pip install -r requirements.txt
python -m pytest -q tests          # 10 fast tests, CPU only
```

## Reproducing the experiments

```bash
# 1. leakage-safe split (already committed as data/splits.csv)
python scripts/make_splits.py --eda <eda_dir>/cfp_eda.csv --out data/splits.csv

# 2. preprocess once to a 512 px cache (pad to square, black background, resize)
python scripts/prepare_images.py --src <path>/MMRDR-CFP --out cache512

# 3. illumination ablation on the baseline (select by validation QWK)
python scripts/train.py --config configs/A_baseline.yaml --splits data/splits.csv --images cache512 --out outputs/runs/A_baseline
python scripts/train.py --config configs/A_graham.yaml   --splits data/splits.csv --images cache512 --out outputs/runs/A_graham
python scripts/train.py --config configs/A_clahe.yaml    --splits data/splits.csv --images cache512 --out outputs/runs/A_clahe

# 4. model ladder with the selected normaliser (one change at a time)
python scripts/train.py --config configs/B_ordinal.yaml    ... data.normaliser=<winner>
python scripts/train.py --config configs/C_multitask.yaml  ... data.normaliser=<winner>
python scripts/train.py --config configs/D_bottleneck.yaml ... data.normaliser=<winner>
```

Any config value can be overridden from the command line (`train.epochs=10`). Runs resume after a
disconnect with `--resume`. Seeds are fixed (`seed: 42`); library versions and GPU are logged in `train.log`.

| Config | Change | Decision |
|---|---|---|
| `A_baseline` | ResNet-50 (ImageNet), softmax, class-balanced CE | A1, T1 |
| `A_graham` / `A_clahe` | + illumination normalisation | P3 |
| `B_ordinal` | ordinal CORN head | A3 |
| `C_multitask` | + 7-lesion auxiliary head | A4 |
| `D_bottleneck` | grade predicted only from lesion probabilities | A5 |

**Experimental protocol.** The official test split is kept intact; training images that duplicate a test
image are removed (exact MD5 or retinal-vessel similarity ≥ 0.98); validation is a stratified, group-aware
1/7 of the remaining training data. Model selection, temperature scaling, conformal calibration and
operating thresholds use the validation set only. The test set is evaluated once.

## External resources

- ImageNet-pretrained ResNet-50 / ConvNeXt-T weights from `torchvision`.
- CORN ordinal loss implemented from Shi, Cao & Raschka (2023).
- All other code written for this project.

## Evaluation, reliability and explanations

Run after training, per model (the test set is scored once; everything fitted uses validation):

```bash
python scripts/evaluate.py   --run outputs/runs/C_multitask_none --splits data/splits.csv --images cache512
python scripts/robustness.py --run outputs/runs/C_multitask_none --splits data/splits.csv --images cache512
python scripts/explain.py    --run outputs/runs/C_multitask_none --splits data/splits.csv --images cache512
```

| Script | Fitted on validation | Reported on test |
|---|---|---|
| `evaluate.py` | temperature (calibration), decision rule (argmax vs median), referable / sight-threatening thresholds for ≥ 90 % sensitivity, class-conditional conformal thresholds (α = 0.1) | accuracy, macro-F1, QWK with bootstrap 95 % CIs; per-class metrics; confusion matrix; under-grading and distant-error rates; sensitivity/specificity at the operating points; ECE before/after calibration; conformal coverage and set size; review-flag performance; selective-prediction curve; lesion AUROC |
| `robustness.py` | — (re-uses the above unchanged) | the same metrics under blur, under-exposure and JPEG compression, plus the **blur-shortcut test**: do degraded healthy eyes get referred? |
| `explain.py` | — | Grad-CAM for the severity estimate and for individual lesions on representative correct, erroneous and flagged cases; clinician-facing summary panels; saliency sanity check (model randomisation) |

Outputs are written to `<run>/eval/`, `<run>/robustness/` and `<run>/explain/`.

## Validation results (model selection)

| Run | Val QWK | Macro-F1 | Under-grading | Referable sens. (default rule) |
|---|---|---|---|---|
| A baseline (softmax) | 0.893 | 0.719 | 11.5 % | 0.749 |
| A + Graham / + CLAHE | 0.894 / 0.889 | 0.692 / 0.701 | 12.1 % / 10.9 % | 0.816 / 0.808 |
| B ordinal | 0.895 | 0.676 | 8.2 % | 0.860 |
| **C ordinal + lesions (main)** | **0.910** | 0.706 | 9.6 % | **0.878** |
| D concept bottleneck | 0.899 | 0.680 | 12.1 % | 0.838 |

Test-set results are produced by `evaluate.py` and summarised in the report.
