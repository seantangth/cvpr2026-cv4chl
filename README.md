# CV4CHL 2026 Competition Code Package

> Cerebral-palsy gait phenotyping from 2D pose keypoints.
> Training code, deterministic assembly code, and frozen final-submission artifacts.
> Byte-exact reproduction is verifiable via `--skip-train`; full retrain is functionally
> equivalent (see "Byte-exact reproduction scope" below).

**Author:** Tze-Hsiang Tang (Kaggle: `seantangth`)
**Repository:** https://github.com/seantangth/cvpr2026-cv4chl
**Competition:** [[CVPR 2026] The First AI for Children Challenge - Kaggle page](https://www.kaggle.com/competitions/cvpr-2026-the-first-ai-children-challenge)
**License:** CC BY-NC-SA 4.0 (see `LICENSE`; per Challenge Rules section 4).

---

## Submission Scope

This repository is the public code package for the CVPR 2026 AI for Children Challenge
submission. It includes the training notebooks and support code needed to retrain the
model/rule sources from the released training set, plus compact frozen artifacts for a fast
byte-exact check of the submitted CSV.

Included:

- preprocessing code for the competition pose/keypoint data;
- training notebooks for the XGBoost baseline, Track-2 clinical classifier, EVGS rule
  sources, item-5 classifier ensemble, item-12 model-diversity source, and item-14 TCN;
- deterministic final assembly code under `src/cv4chl/`;
- frozen intermediate CSV artifacts under `submissions/` for quick integrity checks;
- documentation for data access, methodology, and retraining.

Not included:

- raw CGPS data, because it must be downloaded from Kaggle under the competition rules.

This repository is intended to be submitted as the public GitHub URL for code review.

---

## Full Retraining

> **Python 3.10 is only required for full retrain.** `requirements.txt` pins
> training-time versions matched to the release environment (numpy 1.26.4, xgboost 2.0.3,
> catboost 1.2.5, ...). Newer Python versions ship newer binary wheels with incompatible
> ABI; `pip install -r requirements.txt` will succeed but kernel imports fail at runtime
> with `numpy.dtype size changed`. The fast integrity check (`--skip-train`) uses only
> the lightweight pins in `requirements-reproduce.txt` and works on any Python 3.9+
> (verified on 3.9.6, 3.10, 3.11).

Full retraining requires the CGPS data under `1_data/raw/`:

```bash
cd cvpr2026-cv4chl
python3.10 -m venv .venv         # MUST be Python 3.10
source .venv/bin/activate
pip install -r requirements.txt

# Download CGPS data into 1_data/raw/ first; see docs/data_access.md.
python reproduce.py
```

The full pipeline is:

```text
raw CGPS data
  |
  |  preprocessing notebook
  v
1_data/processed/{keypoints, train_ready}.pkl
  |
  |  training notebooks
  v
5_outputs/submissions/                 runtime outputs, not committed
  |  includes track2_clinical_predictions.csv from the Track-2 notebook
  |
  |  src/cv4chl/ml_merge.py
  v
5_outputs/submissions/preclinical_ml_merge_track1.csv
  |
  |  src/cv4chl/clinical_consistency.py
  v
5_outputs/submissions/clinical_consistency_track1.csv
  |
  |  src/cv4chl/final_assembly.py
  v
submissions/final/final_submission.csv
  |
  |  SHA-256 check
  v
byte-exact reproduction confirmed
```

Retraining stages are executed by `reproduce.py` via `nbconvert`:

| Stage | Notebook |
|---|---|
| Preprocess | `notebooks/01_preprocess/CV4CHL_01_preprocess_features.ipynb` |
| XGBoost baseline | `notebooks/02_train_pipeline/CV4CHL_02_train_xgb_baseline.ipynb` |
| Track-2 clinical classifier | `notebooks/02_train_pipeline/CV4CHL_03_train_track2_clinical_classifier.ipynb` |
| EVGS rule sources | `notebooks/02_train_pipeline/CV4CHL_04_train_evgs_rule_sources.ipynb` |
| Item-12 model diversity | `notebooks/02_train_pipeline/CV4CHL_05_train_model_diversity_item12.ipynb` |
| Item-5 classifier ensemble | `notebooks/02_train_pipeline/CV4CHL_06_train_item05_classifier_ensemble.ipynb` |
| Item-14 TCN | `notebooks/02_train_pipeline/CV4CHL_07_train_tcn_item14.ipynb` |

The deterministic stages after training are plain Python modules:

| Stage | Code |
|---|---|
| Pre-clinical ML merge | `src/cv4chl/ml_merge.py` |
| Clinical consistency | `src/cv4chl/clinical_consistency.py` |
| Final assembly | `src/cv4chl/final_assembly.py` |

Useful controls:

```bash
python reproduce.py --skip-tcn
python reproduce.py --from evgs_rule_sources --to evgs_rule_sources
python reproduce.py --from clinical_consistency --to clinical_consistency
python reproduce.py --from assemble
python reproduce.py --dry-run-tcn
```

If cross-GPU TCN drift is observed, `python reproduce.py --skip-tcn` uses the committed
item-14 reference slice and keeps the final byte-exact.

### Build-chain self-consistency check

Every `reproduce.py` invocation begins with a closure check on the committed release
artifacts: the Track-1 pre-clinical merge is rebuilt from
`submissions/reference/xgb_baseline.csv` + `submissions/item_sources/*.csv` via
`src/cv4chl/ml_merge.py`, and asserted byte-equal to
`submissions/intermediate/preclinical_ml_merge_track1.csv`. If those artifacts are not
derivable from each other under the documented merge code, the run aborts before any
training or assembly happens (exit code 2). This guards against the failure mode where
the committed baseline and preclinical drift apart under the documented merge code.

### Byte-exact reproduction scope

`--skip-train` is the byte-exact path: the deterministic post-training code is rerun
over the committed reference artifacts, the build-chain self-check confirms internal
closure, and the final SHA-256 matches `cfebf119…05a26`.

Full retrain (`python reproduce.py` with no flags) reruns every training notebook from
the raw CGPS data. Whether its final SHA-256 also lands at `cfebf119…05a26` depends on
training-time determinism — XGBoost / LightGBM / CatBoost / 1D-TCN seeded with
`random_state=42` and pinned library versions can drift across hardware (e.g. CUDA
TCN reduction order on a different GPU, sklearn / XGBoost minor versions). Treat full
retrain as **functionally equivalent**, not byte-exact: the committee receives both a
verifiable byte-exact reproduction (`--skip-train`) and a from-scratch retrain path
(`python reproduce.py`) for source-level review of every model.

### Full-retrain provenance contract

`python reproduce.py` (no flags) is **strict full retrain**: every assembly stage reads
only freshly produced runtime artifacts under `5_outputs/`. If any expected runtime
artifact is missing, assembly raises `FileNotFoundError` naming the responsible notebook,
ensuring the final CSV genuinely depends on the retrain.

The clinical-consistency stage's `verify_reference` assertion (which compares the
regenerated intermediate against `submissions/item_sources/clinical_consistency_track1.csv`)
is enabled only under `--skip-train`, where post-training stages are deterministic over
committed inputs. Under full retrain the freshly trained sources may produce a
legitimately different intermediate, so byte-exactness is enforced exactly once at the
final SHA-256 check rather than redundantly mid-pipeline.

The only runtime path scaffolded from a committed file in full retrain is
`5_outputs/submissions/reference_baseline.csv`, which training notebooks 04-07 read as
their base table to overlay onto. Notebook 02 overwrites it with its freshly trained
output, so on a default full-retrain run this scaffolding has no effect on the final CSV.

`--skip-tcn` is a documented exception: it skips the item-14 TCN notebook and stages the
committed item-14 slice into the canonical runtime path before assembly runs. This is
the one place where a committed slice is intentionally promoted to a runtime output;
the `staged ... (--skip-tcn item-14 reuse)` log line marks it explicitly.

Each per-item source has exactly one canonical fresh path. The training notebook at the
right is the unique producer of that path; the file naming and the per-item selection
rule (CV-driven or pre-defined) are encoded directly in that notebook:

| Item | Canonical fresh path | Producer |
|---|---|---|
| Track-1 XGB baseline | `5_outputs/submissions/xgb_baseline.csv` | `CV4CHL_02_train_xgb_baseline.ipynb` |
| Track-2 classifier   | `5_outputs/submissions/track2_clinical_predictions.csv` | `CV4CHL_03_train_track2_clinical_classifier.ipynb` |
| EVGS rule, item 03   | `5_outputs/submissions/evgs_rules_individual_item_03.csv` | `CV4CHL_04_train_evgs_rule_sources.ipynb` |
| EVGS rule, item 09   | `5_outputs/submissions/evgs_rules_individual_item_09.csv` | `CV4CHL_04_train_evgs_rule_sources.ipynb` |
| EVGS rule, item 11   | `5_outputs/submissions/evgs_rules_individual_item_11.csv` | `CV4CHL_04_train_evgs_rule_sources.ipynb` |
| EVGS rule, item 15   | `5_outputs/submissions/evgs_rules_individual_item_15.csv` | `CV4CHL_04_train_evgs_rule_sources.ipynb` |
| EVGS rule, item 16   | `5_outputs/submissions/evgs_rules_individual_item_16.csv` | `CV4CHL_04_train_evgs_rule_sources.ipynb` |
| EVGS rule, item 17   | `5_outputs/submissions/evgs_rules_individual_item_17.csv` | `CV4CHL_04_train_evgs_rule_sources.ipynb` |
| Item-12 model diversity | `5_outputs/submissions/item12_model_diversity.csv` | `CV4CHL_05_train_model_diversity_item12.ipynb` (canonical-emit cell) |
| Item-5 classifier ensemble | `5_outputs/submissions/item05_classifier_ensemble_full.csv` | `CV4CHL_06_train_item05_classifier_ensemble.ipynb` |
| Item-14 TCN | `5_outputs/submissions/tcn_item14/item14_tcn.csv` | `CV4CHL_07_train_tcn_item14.ipynb` (canonical-emit cell) |

The training notebooks also produce exploratory candidate CSVs (per-item rule
applications for items not in the canonical set, top-N model-diversity combos,
TCN per-item / combo / top-N variants) for offline review. These are written to
`5_outputs/diagnostics/<source>/` and are explicitly **not** consumed by
`final_assembly.py`. Only the artifacts listed in the table above feed the final
CSV. The diagnostic CSVs exist to let a reviewer inspect the model selection
process; they are byproducts of LOSO CV, not candidate submissions.

### Clean-room reproducibility check

The strongest reproducibility test, recommended for organizer review:

```bash
# 1. Wipe runtime outputs and (temporarily) hide committed item-source slices.
rm -rf 5_outputs/
mv submissions/item_sources /tmp/item_sources.bak
mv submissions/intermediate /tmp/intermediate.bak

# 2. Full retrain end-to-end from raw data. --skip-release-self-check tells
#    reproduce.py not to run the build-chain closure check (which would fail
#    here because we just hid the files it inspects).
python reproduce.py --skip-release-self-check

# 3. Restore the hidden directories afterwards (they back the --skip-train path).
mv /tmp/item_sources.bak submissions/item_sources
mv /tmp/intermediate.bak submissions/intermediate
```

The build-chain self-check (run by default at Stage 0) verifies that the committed
release artifacts are internally consistent — i.e. that the documented merge code
walks from `submissions/reference/xgb_baseline.csv` + `submissions/item_sources/*.csv`
to `submissions/intermediate/preclinical_ml_merge_track1.csv` byte-equal. The
clean-room test deliberately removes those committed files to verify that retraining
is not secretly reading them, so the closure check has to be opt-out for that scenario.

If step 2 finishes successfully, the final CSV truly depends only on the retrained
runtime artifacts. If any training notebook fails to produce its canonical artifact,
assembly fails immediately rather than papering over the gap with the
hidden committed slices.

---

## Fast Integrity Check

This package is intended to let a reviewer regenerate:

```text
submissions/final/final_submission.csv
```

from the committed release artifacts and verify its SHA-256 hash. The fast path does
not require the raw Kaggle dataset or GPU, but it is not a substitute for full retraining.

```bash
cd cvpr2026-cv4chl
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-reproduce.txt
python reproduce.py --skip-train
```

Expected result:

```text
final csv sha256: cfebf119712f900288156a8b409442a792aef677b49c66358d52ec3399905a26
byte-exact reproduction confirmed
```

`reproduce.py --skip-train` runs the deterministic post-training stages only:

1. copy the committed pre-clinical ML-merge checkpoint into `5_outputs/submissions/`;
2. regenerate `clinical_consistency_track1.csv` by applying patient-agnostic
   clinical-knowledge second-pass rules from `src/cv4chl/clinical_rules.py` (37 cells);
3. assemble the final CSV from release artifacts plus the regenerated clinical source;
4. check the expected hash.

The skip-train path reads only the committed release artifacts under `submissions/`;
runtime model outputs under `5_outputs/` are ignored on this path. The fallback policy
is mode-specific: full retrain (`python reproduce.py`) fails fast if any runtime
artifact is missing, while `--skip-train` falls back to the committed slices by
design.

---

## Package Scope

This is a final-submission reproduction and review package.

- The filename `final_submission.csv` is the submitted artifact name and
  is treated by the reproduction code only as a filename.
- The committed CSVs in `submissions/reference/`, `submissions/intermediate/`, and
  `submissions/item_sources/` are frozen intermediate prediction artifacts from the
  documented model/rule pipeline.
- The clinical-consistency stage applies rules that inspect only the ML first-pass EVGS
  row: item pattern, side totals, row total, and L/R symmetry. It writes
  `5_outputs/reports/clinical_second_pass_log.csv` so each changed cell has a rule ID and
  evidence string.
- The deterministic assembly code reads only local CSV artifacts and recomputes Track-1
  `Total`.
- Full retraining code is included in `notebooks/` and is the primary route for organizer
  retraining. The `--skip-train` path is only a quick integrity check.
- The Track-2 retraining notebook writes `5_outputs/submissions/track2_clinical_predictions.csv`
  from released Track-2 training labels, released cross-track EVGS labels where available,
  and Track-1 model-predicted EVGS for the remaining Track-2 test patients.
- The committed `submissions/intermediate/track2_classifier_output.csv` is a frozen
  snapshot of `5_outputs/submissions/track2_clinical_predictions.csv` produced by the
  Track-2 retraining notebook's `classify_clinical()` function (Rodda & Graham
  exclusion + positive-evidence logic with L/R symmetry post-processing). It is used
  by the `--skip-train` integrity check and is reproducible from the released code +
  training data by rerunning notebook 03.

The post-training entry points are `src/cv4chl/ml_merge.py`,
`src/cv4chl/clinical_consistency.py`, and `src/cv4chl/final_assembly.py`.

See [docs/methodology.md](docs/methodology.md) for the pipeline overview.
Detailed methodology (feature engineering, per-item source selection,
clinical post-processing rationale, references) is in the accompanying
technical report.

---

## Repository Layout

```text
cvpr2026-cv4chl/
├── reproduce.py
├── README.md
├── LICENSE
├── requirements-reproduce.txt
├── requirements.txt
├── src/cv4chl/
│   ├── __init__.py
│   ├── clinical_consistency.py
│   ├── clinical_rules.py
│   ├── final_assembly.py
│   ├── ml_merge.py
│   └── verify_byte_exact.py
├── notebooks/
│   ├── 00_data_review/
│   ├── 01_preprocess/
│   ├── 02_train_pipeline/
│   └── 03_assembly/        # optional interactive wrapper around src/cv4chl/*; not executed by reproduce.py
├── submissions/
│   ├── final/final_submission.csv
│   ├── intermediate/       # pre-clinical ML-merge checkpoint + Track-2 classifier output
│   ├── item_sources/       # minimal per-item and clinical-consistency source slices
│   └── reference/          # XGBoost baseline reference
├── references/
└── docs/
    ├── methodology.md
    ├── data_access.md
    └── retraining.md
```

Runtime-only directories are created by `reproduce.py` and are not committed:

- `1_data/raw/`
- `1_data/processed/`
- `5_outputs/submissions/`

---

## Data Access

The CGPS dataset is redistributed only via Kaggle in line with the competition licensing.

```bash
mkdir -p 1_data/raw && cd 1_data/raw
kaggle competitions download -c cvpr-2026-the-first-ai-children-challenge
unzip -q cvpr-2026-the-first-ai-children-challenge.zip
tar -xjf dataset.tar.bz2
```

See [docs/data_access.md](docs/data_access.md) for expected layout and offline
reproduction notes.

---

## Compute Requirements

| Stage | CPU | GPU | Wall time |
|---|---|---|---|
| Fast reviewer reproduction (`--skip-train`) | yes | no | <5 seconds |
| Preprocessing | yes | no | ~10 min |
| xgb_baseline / track2_clinical / evgs_rule_sources / model_diversity_item12 / item05_classifier_ensemble | yes | no | 2-7 min each |
| tcn_item14 (1D-TCN) | yes | recommended | ~30-60 min on T4/A10/A100; ~2 hours on CPU |

Tested on Python 3.10, macOS 14 CPU, and Ubuntu 22.04 with NVIDIA T4.

---

## License

This code and non-dataset artifacts are released under **CC BY-NC-SA 4.0** (see `LICENSE`),
consistent with Challenge Rules section 4. The Edinburgh Visual Gait Score Reference
Guide PDF in `references/` is copyright of its original authors and is included for
reproducibility / evaluation purposes only under fair use; it is **not** relicensed by
this repository.
