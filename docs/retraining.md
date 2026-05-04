# Retraining Instructions

This document describes the organizer-facing retraining path for the CV4CHL code package.
The fast `--skip-train` path is only an integrity check of frozen artifacts; use the steps
below to rerun the training pipeline from the released training set.

## 1. Prepare Environment

```bash
cd cvpr2026-cv4chl
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.10. The TCN stage can run on CPU but is intended for a CUDA
GPU such as T4, A10, or A100.

## 2. Prepare Data

Download the CGPS competition data into `1_data/raw/` as described in
`docs/data_access.md`. Expected minimum layout:

```text
1_data/raw/
├── track1_train.json
├── track2_train.json
└── dataset/
    └── {patient_id}/{video_pose}/frame_*.json
```

The repository does not redistribute these files.

## 3. Run Full Pipeline

```bash
python reproduce.py
```

`reproduce.py` executes the notebooks in order:

1. `notebooks/01_preprocess/CV4CHL_01_preprocess_features.ipynb`
2. `notebooks/02_train_pipeline/CV4CHL_02_train_xgb_baseline.ipynb`
3. `notebooks/02_train_pipeline/CV4CHL_03_train_track2_clinical_classifier.ipynb`
   - writes `5_outputs/submissions/track2_clinical_predictions.csv`
   - uses released Track-2 training labels, released cross-track EVGS labels where
     available, and Track-1 model-predicted EVGS for the remaining Track-2 test patients
   - applies the in-notebook `classify_clinical()` Rodda & Graham exclusion +
     positive-evidence classifier plus L/R symmetry post-processing
4. `notebooks/02_train_pipeline/CV4CHL_04_train_evgs_rule_sources.ipynb`
5. `notebooks/02_train_pipeline/CV4CHL_05_train_model_diversity_item12.ipynb`
6. `notebooks/02_train_pipeline/CV4CHL_06_train_item05_classifier_ensemble.ipynb`
7. `notebooks/02_train_pipeline/CV4CHL_07_train_tcn_item14.ipynb`
8. `src/cv4chl/ml_merge.py` - write `preclinical_ml_merge_track1.csv`
9. `src/cv4chl/clinical_consistency.py` - apply the clinical-knowledge second-pass filter
10. `src/cv4chl/final_assembly.py` - write the final CSV

Runtime outputs are written under:

```text
1_data/processed/
5_outputs/submissions/
submissions/final/final_submission.csv
```

## 4. Useful Resume Commands

```bash
python reproduce.py --from preprocess --to preprocess
python reproduce.py --from xgb_baseline --to xgb_baseline
python reproduce.py --from evgs_rule_sources --to evgs_rule_sources
python reproduce.py --from clinical_consistency --to clinical_consistency
python reproduce.py --skip-tcn
python reproduce.py --dry-run-tcn
```

`--dry-run-tcn` is only a smoke test and is not expected to match the submitted final CSV.

## 5. Fast Artifact Check

For a quick check that the committed intermediate artifacts reconstruct the submitted CSV:

```bash
pip install -r requirements-reproduce.txt
python reproduce.py --skip-train
```

The fast check still runs the deterministic post-training stages: pre-clinical ML merge,
clinical consistency, final assembly, and SHA-256 verification. It skips only data
preprocessing and model/rule training notebooks.

Expected SHA-256:

```text
cfebf119712f900288156a8b409442a792aef677b49c66358d52ec3399905a26
```
