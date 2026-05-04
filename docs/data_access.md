# Data Access Instructions

This repository does **not** redistribute the raw dataset. Follow the instructions below to
obtain the data directly from Kaggle.

## Primary: Kaggle

1. Create a Kaggle account and accept the Competition Rules at:
   https://www.kaggle.com/competitions/cvpr-2026-the-first-ai-children-challenge
2. Install the Kaggle API: `pip install kaggle` and place your API token at
   `~/.kaggle/kaggle.json` (see https://www.kaggle.com/docs/api).
3. From the repository root:
   ```bash
   mkdir -p 1_data/raw && cd 1_data/raw
   kaggle competitions download -c cvpr-2026-the-first-ai-children-challenge
   unzip -q cvpr-2026-the-first-ai-children-challenge.zip
   tar -xjf dataset.tar.bz2     # extracts dataset/ folder (~1 GB, 110 patients)
   cd ../..
   ```

This yields:
- `1_data/raw/track1_train.json` (Track 1 EVGS labels)
- `1_data/raw/track2_train.json` (Track 2 CP-subtype labels)
- `1_data/raw/dataset/{patient_id}/{video_pose}/frame_*.json` (per-frame Sapiens-2B keypoints)
- `1_data/raw/dataset.tar.bz2` (original tarball, preserved)

Note: the path `1_data/raw/` is the layout the training notebooks expect — do not rename it
(`reproduce.py` reads from this exact location via `CV4CHL_REPO_ROOT`).

## Regenerating processed pkl from raw

The four processed pkl files (`keypoints.pkl`, `features.pkl`, `labels.pkl`,
`train_ready.pkl`) are produced by the preprocessing notebook and saved to
`1_data/processed/`. The orchestrator runs this automatically as the first stage:

```bash
python reproduce.py --from preprocess --to preprocess
```

Or manually:

```bash
CV4CHL_REPO_ROOT=$(pwd) python -m nbconvert \
    --to notebook --execute --inplace \
    notebooks/01_preprocess/CV4CHL_01_preprocess_features.ipynb
```

Expected runtime: ~5–10 minutes on a modern laptop (CPU-only, no GPU needed).
Output: `1_data/processed/{keypoints.pkl, features.pkl, labels.pkl, train_ready.pkl}`.

## Licensing

The Children Gait Pose Sequence (CGPS) dataset is licensed under CC BY-NC-SA 4.0 by
PediaMed AI / UIUC HCESC. By downloading the data you accept the Kaggle Competition Rules
§2.4 and the dataset CC BY-NC-SA 4.0 license. This repository's code is also CC BY-NC-SA 4.0
(see `LICENSE`) to remain compatible.

## Privacy notice

The CGPS dataset contains pose keypoints from paediatric subjects, with no raw video and no
identifiable metadata beyond anonymized patient IDs (per Kaggle dataset description). Do
not attempt to re-identify subjects. Commercial use is prohibited under the dataset license.
