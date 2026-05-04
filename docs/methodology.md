# Methodology - CVPR 2026 CV4CHL Solution (Pipeline Overview)

> **This file is a brief pipeline overview for reviewers reading the
> reproduction package.** The full method description (feature engineering,
> per-item source selection, clinical post-processing rationale, ablations,
> references) is in the accompanying technical report. The code under
> `src/cv4chl/` and `notebooks/02_train_pipeline/` is the source of truth.

## Pipeline

```text
raw data (Kaggle CGPS)
  |
  |  notebooks/01_preprocess/CV4CHL_01_preprocess_features.ipynb
  v
1_data/processed/{keypoints.pkl, features.pkl, labels.pkl, train_ready.pkl}
  |
  |  notebooks/02_train_pipeline/CV4CHL_{02..07}*.ipynb
  v
5_outputs/submissions/                 runtime per-item sources
  |
  |  src/cv4chl/ml_merge.py
  v
5_outputs/submissions/preclinical_ml_merge_track1.csv
  |
  |  src/cv4chl/clinical_consistency.py
  |  (calls clinical_rules.apply_track1_second_pass)
  v
5_outputs/submissions/clinical_consistency_track1.csv
  |
  |  src/cv4chl/final_assembly.py
  v
submissions/final/final_submission.csv
```

## Stages

| Stage | Code | Purpose |
|---|---|---|
| Preprocess | `notebooks/01_preprocess/CV4CHL_01_*.ipynb` | Extract gait keypoints + handcrafted features from raw pose JSON |
| Per-item training | `notebooks/02_train_pipeline/CV4CHL_{02..07}*.ipynb` | XGBoost baseline + Track-2 classifier + per-item ML / rule sources |
| Pre-clinical merge | `src/cv4chl/ml_merge.py` | Overlay selected per-item sources onto the XGBoost baseline |
| Clinical consistency | `src/cv4chl/clinical_consistency.py` | Patient-agnostic second-pass rules over the merged Track-1 row |
| Final assembly | `src/cv4chl/final_assembly.py` | Combine Track-1 + Track-2 + recompute Track-1 `Total` |

`python reproduce.py --skip-train` runs only the deterministic post-training stages
using the committed reference artifacts (~5 seconds, byte-exact integrity check).
`python reproduce.py` reruns preprocessing and training before the deterministic
post-training stages.
