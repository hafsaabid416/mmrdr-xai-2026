# Design Decision Register

Every design choice in this project is recorded here with **what drove it** and **what supports it**.
The report's justifications are drawn from this register.

**Drivers**
- **D** · Dataset: an observed property of MMRDR-CFP (see `notebooks/01_data_exploration.ipynb`)
- **B** · Brief: a requirement of the assessment brief
- **U** · End user: optometrists/ophthalmologists reviewing outputs in a Boots Opticians workflow (job description)
- **E** · Environment: deployment context: UK primary eye care, high-street hardware, NHS/MHRA governance

**Status:** ✅ decided · 🔬 to be confirmed on the validation set · ⏳ pending EDA result · 💡 proposed future work (not demonstrated)

---

## 1. Data and problem framing

| ID | Decision | Driver | Motivation / evidence | Literature | Status |
|---|---|---|---|---|---|
| D1 | Use CFP subset only; task = 5-grade ICDR classification | B, E | Brief §2; CFP is the screening modality in community/primary care (MMRDR paper) — matches the optometric setting | Wilkinson 2003; Tang 2026 | ✅ |
| D2 | Report clinically framed binary endpoints in addition to 5-class metrics: **referable DR (grade ≥2)** and **sight-threatening DR (grade ≥3)** | U, E | Clinicians act on referral decisions, not 5-class accuracy; screening programmes judge systems on referable-disease sensitivity/specificity | Wong 2018 (ICO guidelines); Gulshan 2016; Abràmoff 2018 | ✅ |
| D3 | Treat **under-grading** as the most harmful error and analyse it separately | B, U | Brief §2 error analysis; a missed PDR delays urgent treatment, over-grading costs an unnecessary referral | Wong 2018; Gulshan 2016 | ✅ |
| D4 | Treat the 537 Mild images with hard exudates as a **grading-convention ambiguity**, not label errors to remove | D | EDA: 41 % of grade 1 carry HE, contradicting "MA only" (ICDR) but consistent with schemes admitting exudates at background level (UK NSC R1). Removing them would bias the class | Wilkinson 2003; NHS DESP grading definitions; Krause 2018 | ✅ |
| D5 | Expect adjacent-grade label noise; interpret 1↔2 and 2↔3 confusions accordingly | D | Source graders reached κ ≥ 0.75 (substantial, not perfect); grader variability in DR is well documented | Tang 2026; Krause 2018 | ✅ |
| D6 | Record the weak laterality–grade association as a finding, not a correction target | D | EDA §2: χ² p = 0.042; PDR 44.6 % right-eye vs ≈50 % elsewhere (≤ 5 pp) — small, plausibly chance across multiple tests | — | ✅ |
| D7 | Treat image sharpness as a **potential shortcut** | D, E | EDA §5: sharpness ρ = −0.23, file size ρ = −0.21 with grade; gradient already present from grade 0→2. Partly genuine pathology (vitreous haemorrhage haze), partly risk of "blurry ⇒ diseased" ⇒ over-grading of poorly focused healthy eyes in stores | Geirhos 2020; DeGrave 2021 | ✅ |

## 2. Splitting and leakage

| ID | Decision | Driver | Motivation / evidence | Literature | Status |
|---|---|---|---|---|---|
| S1 | **Keep the official test set intact** | B, D | Comparability with the dataset paper's baselines (ResNet-50 Acc 0.823, F1 0.702) | Tang 2026 | ✅ |
| S2 | **Remove training images that are (near-)duplicates of test images**; also report test metrics on a de-duplicated test subset | D, B | EDA: 197 exact duplicates, 34 groups spanning train and test → proven leakage. Brief: "avoid data leakage" | Kapoor & Narayanan 2023 | ✅ (threshold ⏳) |
| S2a | Detect duplicates by **exact MD5 + retinal-vessel-pattern similarity ≥ 0.98**, not perceptual hashing | D | EDA §6: 64-bit pHash flagged ~87 000 pairs incl. grade conflicts at Hamming 0 — fundus images share global layout. Vessel patterns are eye-specific; all 104 exact pairs score 1.0. **Threshold justified by grade-conflict rate**: random pairs conflict 59.9 % of the time; pairs at 0.80–0.98 conflict 71–75 % (≈ random ⇒ different eyes); 0.98–0.995 24 %; > 0.995 0 % (copies). Confirmed visually. Result: 263 images in 127 groups, 45 spanning train/test, 4 with conflicting grades; 52 train images dropped | Zauner 2010; Kapoor & Narayanan 2023 | ✅ |
| S3 | Validation = one stratified, **group-aware** fold (1/7 ≈ 14 %) of the training pool; within training keep one image per duplicate group and drop groups with conflicting grades | B, D | Brief: no model selection on test. Result: 7,507 train / 1,252 val / 2,225 test; excluded 52 (copies of test), 76 (redundant copies), 6 (conflicting grades) | Kapoor & Narayanan 2023 | ✅ |
| S4 | Acknowledge residual patient-level leakage risk | D | CFP split is image-level; no patient IDs exist in the source (OIA-DDR) — fellow eyes may cross splits | Tang 2026; Li 2019 | ✅ (limitation) |
| S5 | Freeze the split in `splits.csv`, versioned in the repository, fixed seed | B | Reproducibility requirement | Pineau 2021 (ML reproducibility checklist) | ✅ |

## 3. Preprocessing and augmentation

| ID | Decision | Driver | Motivation / evidence | Literature | Status |
|---|---|---|---|---|---|
| P1 | **Pad to square (black); no FOV crop; never stretch** | D | EDA §4: median FOV fraction 0.785 ≈ π/4 ⇒ already cropped to an inscribed circle; 1,290 (11.6 %) non-square images (aspect ≤ 1.33) would be distorted by stretching; 1,857 distinct resolutions | Graham 2015 | ✅ |
| P1b | **Black out everything outside the retinal circle** (geometric mask: circle centred in frame, diameter = longer side) | D | EDA §8: some cameras leave a grey background (device signature ⇒ shortcut risk). A brightness-threshold mask wrongly removed dark retina, so the mask uses the inscribed-circle geometry shown in §4 (FOV fraction ≈ π/4) | Geirhos 2020 | ✅ |
| P2 | Input resolution **512 × 512** | D, B | EDA §4: median width 1,659 px ⇒ ×3.2 downscale keeps microaneurysms (~10–20 px native) at ~3–6 px; at 224 they fall to ~1–3 px. Matches paper baselines | Sahlsten 2019; Tang 2026 | ✅ |
| P3 | Illumination normalisation: **none** (ablation on baseline, validation) | D, E | Val QWK none 0.893 / Graham 0.894 / CLAHE 0.889 (ΔQWK ≤ 0.005, within noise); none had the best macro-F1 (0.719) and calibration (ECE 0.051) and keeps natural colour for clinicians. Pre-set rule: switch only for a clear gain | Graham 2015; Zuiderveld 1994 | ✅ |
| P4 | Augmentation: flips, rotation (±180°), brightness/contrast/gamma jitter, small scale jitter; **no aggressive cropping**; background re-masked | D, E | EDA §4/§7: brightness 26–210, colour casts, many cameras; no canonical orientation; laterality association weak (D6); large crops may remove the only defining lesion | Graham 2015; Gulshan 2016 | ✅ |
| P4a | **Mild blur + JPEG-compression augmentation on all grades** | D, E | Decorrelates sharpness from grade (D7); simulates store image quality | Geirhos 2020; Hendrycks & Dietterich 2019 | ✅ |
| P6 | **Keep low-quality images** in training and test | D | EDA §7: darkest / least-sharp images are disproportionately PDR — exclusion would remove severe cases; quality studied via R5 instead | Hendrycks & Dietterich 2019 | ✅ |
| P5 | ImageNet mean/std normalisation | — | Required by ImageNet-pretrained weights | He 2016 | ✅ |

## 4. Model architecture

| ID | Decision | Driver | Motivation / evidence | Literature | Status |
|---|---|---|---|---|---|
| A1 | **Baseline: ResNet-50, ImageNet-pretrained, softmax** | B | Brief asks for a baseline; same backbone as dataset paper → direct comparability | He 2016; Tang 2026 | ✅ |
| A2 | CNN backbone (not ViT) for the main model | D, B, U | Small lesions favour high resolution and local inductive bias; Grad-CAM is native and better studied on CNNs; dataset paper shows RETFound (ViT-L) ≈ ResNet-50 on CFP (F1 0.714 vs 0.702) | Zhou 2023; Tang 2026; Selvaraju 2017 | ✅ |
| A3 | **Ordinal grade head (CORN)** | D, B, U | Grades are ordered. Validation: QWK unchanged vs A (0.895 vs 0.893) but **under-grading fell 11.5 % → 8.2 %** and referable sensitivity at the default rule rose 0.75 → 0.86 — a clinical benefit QWK hides | Shi 2023; Cao 2020 | ✅ |
| A4 | **Auxiliary 7-lesion multi-label head** (multi-task) — **main model (C)** | D, B, U | EDA: lesions determine grade. Validation: best QWK **0.910**, accuracy 0.821, referable sensitivity 0.878; kept improving to epoch 14 while A/B overfit by epoch 3 ⇒ lesion supervision acts as a regulariser. Gives clinician-readable findings | Caruana 1997; Wang 2017 | ✅ |
| A5 | **Concept-bottleneck variant (D)**: grade predicted only from the 7 lesion probabilities | U, B | Fully transparent, overridable reasoning. Validation QWK 0.899 — **interpretability costs only 0.012 QWK** vs C | Koh 2020 | ✅ |
| A6 | Moderate model size (~25 M params) | E | Point-of-care inference on store hardware; cost and latency | Liu 2022 (ConvNeXt); He 2016 | ✅ |
| A7 | Optional backbone comparison (ConvNeXt-T or RETFound) only if compute allows | B | Justified comparison, not a requirement ("state of the art not expected") | Liu 2022; Zhou 2023 | 💡/🔬 |

## 5. Training protocol

| ID | Decision | Driver | Motivation / evidence | Literature | Status |
|---|---|---|---|---|---|
| T1 | Class-balanced loss weighting (effective number of samples, β = 0.999), applied from the baseline onwards | D | EDA: imbalance ratio 10.9 : 1; minority classes are the clinically critical ones | Cui 2019; Johnson & Khoshgoftaar 2019 | ✅ |
| T2 | AdamW, cosine schedule with warm-up, mixed precision | E | Standard, stable fine-tuning; fits a single Colab GPU | Loshchilov & Hutter 2019; 2017 | ✅ |
| T3 | Model selection and early stopping on **validation QWK** (patience 6, max 20 epochs) | B, D | QWK is the standard ordinal agreement metric; test never used for selection. Curves justify the schedule: A/B peak at epoch 2–3 then overfit (train loss → 0, ECE ↑); C peaks at 14 | Cohen 1968 | ✅ |
| T4 | Fixed seeds; config files; logged hyperparameters | B | Reproducibility requirement | Pineau 2021 | ✅ |

## 6. Evaluation

| ID | Decision | Driver | Motivation / evidence | Literature | Status |
|---|---|---|---|---|---|
| V1 | Metrics: accuracy, macro-F1, QWK, per-class P/R/F1, confusion matrix | B, D | Brief requirement; macro-F1 and QWK robust to the 59 % majority class | Cohen 1968 | ✅ |
| V2 | Referable (≥2) and sight-threatening (≥3) operating points: threshold on P(grade ≥ k) fitted on **validation** for **≥ 90 % sensitivity**, maximising sensitivity + specificity among qualifying thresholds (midpoints between scores) | U, E | Validation showed the default argmax rule missed 25 % of referable cases (baseline) — screening needs an explicit, sensitivity-first operating point. 90 % exceeds the commonly cited 80 % minimum for DR screening | Gulshan 2016; Abràmoff 2018; Scanlon 2017 | ✅ |
| V3 | **Bootstrap 95 % CIs** for all test metrics | D | EDA: only 100 Severe / 134 PDR test images → high variance per class | Efron & Tibshirani 1993 | ✅ |
| V4 | Under-grading rate and "distant error" rate (|error| ≥ 2) | B, U | Brief: underestimation and clinically different grades | Gulshan 2016 | ✅ |
| V5 | Error analysis cross-referenced with lesion labels (e.g. PDR with NV predicted < 4) | B, D | Ties errors to clinical findings rather than pixels | Krause 2018 | ✅ |
| V6 | Laterality as the only subgroup analysis; full bias audit as future work | D, E | No demographic/device metadata | Obermeyer 2019 | ✅ / 💡 |

## 7. Reliability and uncertainty

| ID | Decision | Driver | Motivation / evidence | Literature | Status |
|---|---|---|---|---|---|
| R1 | Calibration: reliability diagram, ECE; **temperature scaling** fitted on validation (grid search on NLL; works for softmax and CORN) | B, U | Validation ECE of the main model was 0.13 — confidence must be recalibrated before being shown to clinicians | Guo 2017 | ✅ |
| R2 | **Class-conditional (Mondrian) conformal prediction sets**, LAC score, α = 0.10, fitted on validation | U, B, D | Guarantees ≈ 90 % coverage **for each grade**, not just on average — marginal coverage would be dominated by the 59 % No-DR class and under-cover Severe/PDR, the clinically critical grades | Angelopoulos & Bates 2021; Sadinle 2019; Vovk 2005 | ✅ |
| R3 | **Selective prediction** (accuracy/QWK vs fraction graded automatically) and **management-boundary uncertainty flags**: refer for review only when the conformal set spans the referral boundary (0–1 vs ≥ 2) or the urgency boundary (≤ 2 vs ≥ 3); the predicted grade is always included in the set | U, B | First test run flagged 52.6 % of cases when *any* multi-grade set triggered review — impractical workload, and most of that uncertainty (e.g. "0 or 1") does not change management. Including the point prediction keeps the set consistent with the headline grade and can only increase coverage | Leibig 2017; Geifman & El-Yaniv 2017; Angelopoulos & Bates 2021 | ✅ |
| R4 | **Rule-based review flags** from the grading protocol: PDR signs (NV/VH/RD) but graded < 4; lesions but graded 0; VB/IRMA but graded < 3 | U, D | Auditable safety net — the clinician sees *why* a case was flagged; derived from the ICDR criteria the labels follow (EDA §3) | Wilkinson 2003 | ✅ |
| R5 | Robustness to **synthetic image-quality degradation** (blur, under-exposure, JPEG compression) — does accuracy fall, does uncertainty rise, and **does blurring grade-0 images raise the predicted grade** (tests the D7 shortcut)? | D, E | EDA §5: KS train vs test ≤ 0.033 ⇒ official test set has no quality shift; real stores vary in cameras and operators | Hendrycks & Dietterich 2019; Beede 2020 | ✅ |

## 8. Explainability

| ID | Decision | Driver | Motivation / evidence | Literature | Status |
|---|---|---|---|---|---|
| X1 | **Grad-CAM** on the final convolutional block for the predicted grade | B | Brief §3.1; native to CNNs, class-discriminative | Selvaraju 2017 | ✅ |
| X2 | **Sanity check**: cascading randomisation of the last conv stage + heads; Spearman ρ between trained and randomised Grad-CAMs (30 test images) | B | Low ρ ⇒ maps depend on learned weights; high ρ ⇒ they reflect the input only | Adebayo 2018 | ✅ |
| X3 | Lesion-head outputs as **concept-level explanations**, with lesion-specific Grad-CAM | B, U, D | Brief §3.2; clinicians reason in findings, not pixels | Koh 2020; Selvaraju 2017 | ✅ |
| X4 | Quantitative check of explanations against lesion labels is **image-level only** (no lesion masks) — state this limitation explicitly | D, B | MMRDR provides presence flags, not locations; saliency often localises poorly in medical images | Arun 2021; Saporta 2022 | ✅ |
| X5 | Position explanations as aids to review, not evidence of correctness | U, B | Brief §3.3; explanations can increase unwarranted trust | Ghassemi 2021 | ✅ |

## 9. Communication to clinicians

| ID | Decision | Driver | Motivation / evidence | Literature | Status |
|---|---|---|---|---|---|
| C1 | One **clinician-facing summary panel** per case: image, heatmap, grade probabilities, conformal set, lesion findings, referral suggestion, review flags | U | JD: co-design clinician-facing interfaces showing confidence, supporting override | Beede 2020; Goddard 2012 | ✅ (prototype figure) |
| C2 | Separate **prediction**, **explanation** and **clinical judgement** explicitly in the panel and report | B, U | Brief §3.3; automation bias risk | Goddard 2012; Ghassemi 2021 | ✅ |

## 10. Governance and reproducibility

| ID | Decision | Driver | Motivation / evidence | Literature | Status |
|---|---|---|---|---|---|
| G1 | **Model card** and **dataset notes** in the repository | E, U | JD: governance-ready outputs, traceability | Mitchell 2019; Gebru 2021 | ✅ |
| G2 | Report structured against **TRIPOD+AI** items where applicable | E, B | Reporting standard for clinical prediction models using AI | Collins 2024 | ✅ |
| G3 | Present as research prototype; no clinical claims | B, E | Brief §6; UK medical-device regulation applies to clinical use | MHRA SaMD/AIaMD programme | ✅ |

## 11. Anticipated challenge (Section 4) — proposed, not demonstrated

| ID | Decision | Driver | Motivation / evidence | Literature | Status |
|---|---|---|---|---|---|
| F1 | Challenge: **safe translation from offline performance to real optometric workflows** (dataset shift across stores/devices, human–AI interaction, governance) | E, U | Real deployments of DR AI have failed on image quality and workflow fit despite good offline accuracy | Beede 2020; Kelly 2019 | 💡 |
| F2 | Strategy: staged evaluation — silent (shadow) mode → supervised pilot with override logging → monitored rollout | E | Recommended staged early clinical evaluation | Vasey 2022 (DECIDE-AI) | 💡 |
| F3 | Site/device subgroup audit, recalibration per site, drift monitoring | E | Shift degrades calibration and performance | Ruamviboonsuk 2022; Guo 2017 | 💡 |
| F4 | Clinical safety case and DPIA from the outset | E | NHS clinical risk management standards DCB0129 (manufacturer) / DCB0160 (deployer); UK GDPR DPIA | NHS England DCB0129/DCB0160; ICO | 💡 |

---

## References (verify every entry against the source before submission)

- Abràmoff MD et al. (2018). Pivotal trial of an autonomous AI-based diagnostic system for detection of diabetic retinopathy in primary care offices. *npj Digital Medicine* 1:39.
- Adebayo J et al. (2018). Sanity checks for saliency maps. *NeurIPS*.
- Angelopoulos AN, Bates S (2021). A gentle introduction to conformal prediction and distribution-free uncertainty quantification. *arXiv:2107.07511*.
- Arun N et al. (2021). Assessing the trustworthiness of saliency maps for localizing abnormalities in medical imaging. *Radiology: Artificial Intelligence* 3(6).
- Beede E et al. (2020). A human-centered evaluation of a deep learning system deployed in clinics for the detection of diabetic retinopathy. *CHI*.
- Cao W, Mirjalili V, Raschka S (2020). Rank consistent ordinal regression for neural networks (CORAL). *Pattern Recognition Letters* 140.
- Caruana R (1997). Multitask learning. *Machine Learning* 28.
- Cohen J (1968). Weighted kappa. *Psychological Bulletin* 70(4).
- Collins GS et al. (2024). TRIPOD+AI statement. *BMJ* 385.
- Cui Y et al. (2019). Class-balanced loss based on effective number of samples. *CVPR*.
- Efron B, Tibshirani RJ (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Gebru T et al. (2021). Datasheets for datasets. *Communications of the ACM* 64(12).
- Geifman Y, El-Yaniv R (2017). Selective classification for deep neural networks. *NeurIPS*.
- Geirhos R et al. (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence* 2.
- Ghassemi M, Oakden-Rayner L, Beam AL (2021). The false hope of current approaches to explainable AI in health care. *Lancet Digital Health* 3(11).
- Goddard K, Roudsari A, Wyatt JC (2012). Automation bias: a systematic review. *JAMIA* 19(1).
- Graham B (2015). Kaggle Diabetic Retinopathy Detection competition report.
- Gulshan V et al. (2016). Development and validation of a deep learning algorithm for detection of diabetic retinopathy in retinal fundus photographs. *JAMA* 316(22).
- Guo C et al. (2017). On calibration of modern neural networks. *ICML*.
- He K et al. (2016). Deep residual learning for image recognition. *CVPR*.
- Hendrycks D, Dietterich T (2019). Benchmarking neural network robustness to common corruptions and perturbations. *ICLR*.
- Johnson JM, Khoshgoftaar TM (2019). Survey on deep learning with class imbalance. *Journal of Big Data* 6.
- Kapoor S, Narayanan A (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns* 4(9).
- Kelly CJ et al. (2019). Key challenges for delivering clinical impact with artificial intelligence. *BMC Medicine* 17.
- Koh PW et al. (2020). Concept bottleneck models. *ICML*.
- Krause J et al. (2018). Grader variability and the importance of reference standards for evaluating machine learning models for diabetic retinopathy. *Ophthalmology* 125(8).
- Leibig C et al. (2017). Leveraging uncertainty information from deep neural networks for disease detection. *Scientific Reports* 7.
- Li T et al. (2019). Diagnostic assessment of deep learning algorithms for diabetic retinopathy screening (OIA-DDR). *Information Sciences* 501.
- Liu Z et al. (2022). A ConvNet for the 2020s. *CVPR*.
- Loshchilov I, Hutter F (2017). SGDR: stochastic gradient descent with warm restarts. *ICLR*. — (2019). Decoupled weight decay regularization. *ICLR*.
- Mitchell M et al. (2019). Model cards for model reporting. *FAT\**.
- Obermeyer Z et al. (2019). Dissecting racial bias in an algorithm used to manage the health of populations. *Science* 366.
- Pineau J et al. (2021). Improving reproducibility in machine learning research. *JMLR* 22.
- Romano Y, Sesia M, Candès E (2020). Classification with valid and adaptive coverage. *NeurIPS*.
- DeGrave AJ, Janizek JD, Lee S-I (2021). AI for radiographic COVID-19 detection selects shortcuts over signal. *Nature Machine Intelligence* 3.
- Ruamviboonsuk P et al. (2022). Real-time diabetic retinopathy screening by deep learning in a multisite national screening programme: a prospective interventional cohort study. *Lancet Digital Health* 4(4).
- Sadinle M, Lei J, Wasserman L (2019). Least ambiguous set-valued classifiers with bounded error levels. *JASA* 114(525).
- Sahlsten J et al. (2019). Deep learning fundus image analysis for diabetic retinopathy and macular edema grading. *Scientific Reports* 9.
- Saporta A et al. (2022). Benchmarking saliency methods for chest X-ray interpretation. *Nature Machine Intelligence* 4.
- Scanlon PH (2017). The English National Screening Programme for diabetic retinopathy 2003–2016. *Acta Diabetologica* 54.
- Selvaraju RR et al. (2017). Grad-CAM: visual explanations from deep networks via gradient-based localization. *ICCV*.
- Shi X, Cao W, Raschka S (2023). Deep neural networks for rank-consistent ordinal regression based on conditional probabilities (CORN). *Pattern Analysis and Applications* 26.
- Tang Z et al. (2026). A multimodal retinal image dataset for diabetic retinopathy detection using foundation models (MMRDR). *Scientific Data* 13:639.
- Vasey B et al. (2022). Reporting guideline for the early-stage clinical evaluation of decision support systems driven by AI: DECIDE-AI. *Nature Medicine* 28.
- Vovk V, Gammerman A, Shafer G (2005). *Algorithmic Learning in a Random World*. Springer.
- Wang Z et al. (2017). Zoom-in-Net: deep mining lesions for diabetic retinopathy detection. *MICCAI*.
- Wilkinson CP et al. (2003). Proposed international clinical diabetic retinopathy and diabetic macular edema disease severity scales. *Ophthalmology* 110(9).
- Wong TY et al. (2018). Guidelines on diabetic eye care: the ICO recommendations for screening, follow-up, referral, and treatment. *Ophthalmology* 125(10).
- Zhou Y et al. (2023). A foundation model for generalizable disease detection from retinal images (RETFound). *Nature* 622.
- Zauner C (2010). *Implementation and benchmarking of perceptual image hash functions*. MSc thesis, Upper Austria University of Applied Sciences.
- Zuiderveld K (1994). Contrast limited adaptive histogram equalization. *Graphics Gems IV*.
