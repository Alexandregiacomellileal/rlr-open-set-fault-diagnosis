# Robust Local Reference Calibration for Open-Set Electrical Fault Diagnosis under Cross-Site Domain Shift

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21904236.svg)](https://doi.org/10.5281/zenodo.21904236)

Reproducibility repository for the paper by **Alexandre Giacomelli Leal** (Independent Researcher, Brazil).

The repository implements **Robust Local Reference (RLR)** calibration for reference-assisted cross-domain open-set electrical fault diagnosis. The main question is whether an unlabeled local reference can reduce the risk that benign site-dependent operating shifts are mistaken for genuinely unknown faults.

> **Key interpretation:** RLR is a domain-deconfounding strategy, not a universal novelty detector.

## Repository contents

```text
.
├── scripts/
│   ├── 12b_unified_leakage_free_evaluation.py
│   ├── 13_reference_duration_rlr_entropy.py
│   ├── 14_reference_estimator_ablation.py
│   ├── 15_reference_contamination_ablation.py
│   └── 16_classifier_robustness_rlr_entropy.py
├── results/
│   ├── manuscript_reference_metrics.csv
│   └── README.md
├── docs/
│   └── PROTOCOL.md
├── DATA.md
├── CITATION.cff
├── .zenodo.json
├── requirements.txt
├── environment.yml
└── LICENSE
```

## Data

The dataset is **not redistributed** in this repository. Download the public three-station controlled-fault benchmark from Zenodo:

**Dataset DOI:** https://doi.org/10.5281/zenodo.20565892

See [`DATA.md`](DATA.md) for details.

## Software archive and DOI

The versioned reproducibility package is archived on Zenodo:

**Software DOI (v1.0.0):** https://doi.org/10.5281/zenodo.21904236

## Installation

### pip

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### conda

```bash
conda env create -f environment.yml
conda activate rlr-open-set
```

## Primary leakage-free experiment

Run the primary evaluation separately for each unseen station. The scripts expect the original dataset ZIP downloaded from Zenodo.

```bash
python scripts/12b_unified_leakage_free_evaluation.py --zip /path/to/ZENODO_UPLOAD_v1.0.zip --out results/generated/primary --station S1
python scripts/12b_unified_leakage_free_evaluation.py --zip /path/to/ZENODO_UPLOAD_v1.0.zip --out results/generated/primary --station S2
python scripts/12b_unified_leakage_free_evaluation.py --zip /path/to/ZENODO_UPLOAD_v1.0.zip --out results/generated/primary --station S3
```

The protocol uses Day 1 only for local reference statistics and Days 2--7 for model development/external evaluation. It combines leave-one-station-out deployment with leave-one-fault-type-out rejection across 13 primary fault types, yielding 39 matched scenarios.

## Sensitivity studies

### Reference duration

```bash
python scripts/13_reference_duration_rlr_entropy.py --zip /path/to/ZENODO_UPLOAD_v1.0.zip --out results/generated/reference_duration --station S1
```

Repeat for `S2` and `S3`.

### Reference estimator

```bash
python scripts/14_reference_estimator_ablation.py --zip /path/to/ZENODO_UPLOAD_v1.0.zip --out results/generated/reference_estimator --station S1
```

Repeat for `S2` and `S3`.

### Reference contamination

```bash
python scripts/15_reference_contamination_ablation.py --zip /path/to/ZENODO_UPLOAD_v1.0.zip --out results/generated/reference_contamination --station S1
```

Repeat for `S2` and `S3`.

### Classifier-family robustness

Example:

```bash
python scripts/16_classifier_robustness_rlr_entropy.py --zip /path/to/ZENODO_UPLOAD_v1.0.zip --out results/generated/classifiers --station S1 --representation RLR --model RF
```

Repeat the desired combinations of station (`S1`, `S2`, `S3`), representation (`RAW`, `RLR`), and classifier (`LR`, `RF`, `LGBM`).

## Reference results

The primary manuscript-level aggregate values are provided in [`results/manuscript_reference_metrics.csv`](results/manuscript_reference_metrics.csv). In the primary controlled experiment, RLR + Entropy achieved the highest mean H-score among the controlled core variants, while RLR + PUF achieved the highest mean unknown AUROC, known-event acceptance, and open-set macro-F1.

The strongest mechanism-specific evidence occurs at the most shifted station (S3): raw geometric rejection accepted only about 3.1% of known events, whereas RLR increased known-event acceptance to approximately 78--80%.

## Reproducibility notes

- The target-site Day-1 reference is unlabeled for the method; labels are used only for post-hoc contamination auditing in the relevant sensitivity study.
- The 39 station/fault scenarios are matched scenarios from **three physical stations**, not 39 independent installations.
- The secondary studies are reduced-complexity sensitivity analyses and should not replace the primary experiment.
- The code uses fixed random seeds internally where specified by the experiment scripts.

## Citation

If you use this code, please cite the archived Zenodo software release:

**Leal, A. G. (2026). Robust Local Reference Calibration for Open-Set Electrical Fault Diagnosis under Cross-Site Domain Shift (v1.0.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.21904236**

GitHub also renders a citation suggestion from [`CITATION.cff`](CITATION.cff).

## License

The code in this repository is released under the [MIT License](LICENSE). The external benchmark dataset is **not** covered by this repository's MIT license and remains under the terms specified by its original Zenodo record.

## Author

**Alexandre Giacomelli Leal**  
Independent Researcher, Brazil  
Corresponding e-mail: alexgiacomelli@yahoo.com
