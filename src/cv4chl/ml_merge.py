"""Per-item ensemble merge utility.

Assembles the per-item ensemble baseline by overlaying four item-specific
candidate sources on top of the xgb_baseline XGBoost base:

* ``evgs_rules`` — threshold-tuned EVGS rules for items
  ``{3, 9, 11, 15, 16, 17}`` (both L and R sides).
* ``item05_classifier_ensemble`` — classifier ensemble (XGB + LightGBM + CatBoost)
  for item ``5`` (both sides).
* ``model_diversity_item12`` — model diversity (CatBoost / LightGBM, trained inside
  ``notebooks/02_train_pipeline/CV4CHL_05_train_model_diversity_item12.ipynb``)
  for item ``12``, R-side only.
* ``tcn_item14`` — 1D-TCN for item ``14`` (both sides).

The per-item ensemble result feeds into the EVGS-pattern-based clinical
consistency checks applied in ``src/cv4chl/clinical_consistency.py``,
which then feeds the final assembly module.

This module exposes:

* :func:`merge_ml_sources` — the main merge function (pure ``pandas``).
* :func:`merge_ml_sources_from_default_layout` — convenience wrapper
  that resolves source paths under a given repo root.
* :func:`write_preclinical_ml_merge` — writes the Track-1 pre-clinical
  checkpoint consumed by ``clinical_consistency.py``.
* :data:`ITEM_SOURCE_MAP` — the (column → source name) dictionary.
"""

from __future__ import annotations

import pathlib
from typing import Mapping

import pandas as pd


# Per-cell source map: column name → logical source key. The merge
# function expects a path for each unique key (`evgs_rules_item3`,
# `evgs_rules_item9`, …, `item05_classifier_ensemble`,
# `model_diversity_item12`, `tcn_item14`).
#
# Track-2 columns and Track-1 columns NOT listed here are taken from
# the base CSV unchanged.
ITEM_SOURCE_MAP: dict[str, str] = {
    "L3": "evgs_rules_item3",  "R3": "evgs_rules_item3",
    "L5": "item05_classifier_ensemble",        "R5": "item05_classifier_ensemble",
    "L9": "evgs_rules_item9",  "R9": "evgs_rules_item9",
    "L11": "evgs_rules_item11", "R11": "evgs_rules_item11",
    "R12": "model_diversity_item12",       # R-side only; L12 stays as base
    "L14": "tcn_item14",        "R14": "tcn_item14",
    "L15": "evgs_rules_item15", "R15": "evgs_rules_item15",
    "L16": "evgs_rules_item16", "R16": "evgs_rules_item16",
    "L17": "evgs_rules_item17", "R17": "evgs_rules_item17",
}


# --------------------------------------------------------------------- #
# Default repo layout                                                   #
# --------------------------------------------------------------------- #

DEFAULT_BASE_REL = pathlib.Path("submissions/reference/xgb_baseline.csv")
DEFAULT_ML_SOURCES_REL = pathlib.Path("submissions/item_sources")
PRECLINICAL_ML_MERGE_REF = pathlib.Path("submissions/intermediate/preclinical_ml_merge_track1.csv")
RUNTIME_PRECLINICAL_ML_MERGE = pathlib.Path("5_outputs/submissions/preclinical_ml_merge_track1.csv")

# Runtime outputs created when the training notebooks are rerun.
# These paths must match what the notebooks emit; assembly fails fast
# with an actionable hint if any runtime source is missing under
# full-retrain mode.
RUNTIME_BASE_REL = pathlib.Path("5_outputs/submissions/xgb_baseline.csv")
RUNTIME_ML_SOURCE_FILES = {
    "evgs_rules_item3":  pathlib.Path("5_outputs/submissions/evgs_rules_individual_item_03.csv"),
    "evgs_rules_item9":  pathlib.Path("5_outputs/submissions/evgs_rules_individual_item_09.csv"),
    "evgs_rules_item11": pathlib.Path("5_outputs/submissions/evgs_rules_individual_item_11.csv"),
    "evgs_rules_item15": pathlib.Path("5_outputs/submissions/evgs_rules_individual_item_15.csv"),
    "evgs_rules_item16": pathlib.Path("5_outputs/submissions/evgs_rules_individual_item_16.csv"),
    "evgs_rules_item17": pathlib.Path("5_outputs/submissions/evgs_rules_individual_item_17.csv"),
    "item05_classifier_ensemble": pathlib.Path("5_outputs/submissions/item05_classifier_ensemble_full.csv"),
    "model_diversity_item12":     pathlib.Path("5_outputs/submissions/item12_model_diversity.csv"),
    "tcn_item14":                 pathlib.Path("5_outputs/submissions/tcn_item14/item14_tcn.csv"),
}

# Maps each runtime source key to the notebook responsible for producing
# it; used to surface actionable error messages on full-retrain failures.
RUNTIME_ML_SOURCE_NOTEBOOK = {
    "evgs_rules_item3":           "02_train_pipeline/CV4CHL_04_train_evgs_rule_sources.ipynb",
    "evgs_rules_item9":           "02_train_pipeline/CV4CHL_04_train_evgs_rule_sources.ipynb",
    "evgs_rules_item11":          "02_train_pipeline/CV4CHL_04_train_evgs_rule_sources.ipynb",
    "evgs_rules_item15":          "02_train_pipeline/CV4CHL_04_train_evgs_rule_sources.ipynb",
    "evgs_rules_item16":          "02_train_pipeline/CV4CHL_04_train_evgs_rule_sources.ipynb",
    "evgs_rules_item17":          "02_train_pipeline/CV4CHL_04_train_evgs_rule_sources.ipynb",
    "item05_classifier_ensemble": "02_train_pipeline/CV4CHL_06_train_item05_classifier_ensemble.ipynb",
    "model_diversity_item12":     "02_train_pipeline/CV4CHL_05_train_model_diversity_item12.ipynb",
    "tcn_item14":                 "02_train_pipeline/CV4CHL_07_train_tcn_item14.ipynb",
}


# --------------------------------------------------------------------- #
# Core merge                                                            #
# --------------------------------------------------------------------- #

def merge_ml_sources(
    base_csv: str | pathlib.Path,
    source_csvs: Mapping[str, str | pathlib.Path],
) -> pd.DataFrame:
    """Merge per-item ML predictions on top of the baseline table.

    Parameters
    ----------
    base_csv
        Path to the baseline CSV. Provides every cell that is not listed
        in :data:`ITEM_SOURCE_MAP`.
    source_csvs
        Mapping ``{logical_source_key: csv_path}`` covering every unique
        key in :data:`ITEM_SOURCE_MAP`. The keys are:

        * ``evgs_rules_item3``, ``evgs_rules_item9``, ``evgs_rules_item11``,
          ``evgs_rules_item15``, ``evgs_rules_item16``, ``evgs_rules_item17`` —
          one CSV per item.
        * ``item05_classifier_ensemble`` — item-5 classifier ensemble CSV.
        * ``model_diversity_item12`` — item-12 model-diversity CSV.
        * ``tcn_item14`` — item-14 TCN CSV.

    Returns
    -------
    pandas.DataFrame
        Merged DataFrame with the same rows / columns as the base, with
        Track-1 ``Total`` recomputed as the sum of items L1..L17 + R1..R17.
        Track-2 rows are unchanged (``Total`` stays at the ``-1`` sentinel).

    Notes
    -----
    The function is pure: no global state, no I/O outside the listed
    paths, and the same inputs always produce the same DataFrame.
    """
    required_keys = set(ITEM_SOURCE_MAP.values())
    missing = required_keys - set(source_csvs)
    if missing:
        raise ValueError(
            f"merge_ml_sources: missing source CSVs for keys {sorted(missing)}. "
            f"Required keys: {sorted(required_keys)}."
        )

    df = pd.read_csv(base_csv).copy()
    t1_mask = df["ID"].str.startswith("track1")
    base_t1_ids = set(df.loc[t1_mask, "ID"])

    # Index each source by ID and verify it covers the same Track-1 cohort.
    # ID-based lookup makes the merge robust to row order and surfaces
    # missing IDs explicitly (a positional `.values` overlay would silently
    # mismerge if any source CSV got resorted).
    src_cache: dict[str, pd.DataFrame] = {}
    for key in required_keys:
        src_df = pd.read_csv(source_csvs[key]).set_index("ID")
        src_t1_ids = {idx for idx in src_df.index if str(idx).startswith("track1")}
        missing = base_t1_ids - src_t1_ids
        if missing:
            raise ValueError(
                f"merge_ml_sources: source '{key}' is missing Track-1 IDs from "
                f"the base table: {sorted(missing)}"
            )
        src_cache[key] = src_df  # full DataFrame, ID-indexed

    for col, src_key in ITEM_SOURCE_MAP.items():
        df.loc[t1_mask, col] = (
            df.loc[t1_mask, "ID"].map(src_cache[src_key][col]).astype(int).values
        )

    item_cols = [f"L{i}" for i in range(1, 18)] + [f"R{i}" for i in range(1, 18)]
    df.loc[t1_mask, "Total"] = df.loc[t1_mask, item_cols].sum(axis=1).astype(int)

    return df


def merge_ml_sources_from_default_layout(
    repo_root: str | pathlib.Path,
    use_runtime: bool = True,
) -> pd.DataFrame:
    """Resolve the standard staging layout and call :func:`merge_ml_sources`.

    Full-retrain mode (``use_runtime=True``) requires every freshly produced
    runtime artifact under ``5_outputs/``; if any is missing this raises
    ``FileNotFoundError`` with the responsible notebook in the message.
    Reviewer reproduction mode (``use_runtime=False``) reads only the
    committed release artifacts under ``submissions/``.
    """
    root = pathlib.Path(repo_root)

    if use_runtime:
        base_path = root / RUNTIME_BASE_REL
        if not base_path.exists():
            raise FileNotFoundError(
                f"ML-merge base CSV not found at {base_path}.\n"
                "Run notebook 02_train_pipeline/CV4CHL_02_train_xgb_baseline.ipynb, "
                "or pass --skip-train to use the committed release artifact."
            )
        sources: dict[str, pathlib.Path] = {}
        missing: list[str] = []
        for key, runtime_rel in RUNTIME_ML_SOURCE_FILES.items():
            runtime_path = root / runtime_rel
            if runtime_path.exists():
                sources[key] = runtime_path
            else:
                nb = RUNTIME_ML_SOURCE_NOTEBOOK.get(key, "<unknown>")
                missing.append(f"  - {runtime_path}  (run {nb})")
        if missing:
            raise FileNotFoundError(
                "Full-retrain ML merge is missing freshly produced runtime "
                "artifacts. Re-run the responsible training notebook(s) before "
                "assembly, or pass --skip-train to use the committed release "
                "artifacts.\nMissing:\n" + "\n".join(missing)
            )
    else:
        base_path = root / DEFAULT_BASE_REL
        if not base_path.exists():
            raise FileNotFoundError(f"ML-merge base CSV not found at {base_path}")
        ml_src = root / DEFAULT_ML_SOURCES_REL
        sources = {
            "evgs_rules_item3":  ml_src / "item03_evgs_rules.csv",
            "evgs_rules_item9":  ml_src / "item09_evgs_rules.csv",
            "evgs_rules_item11": ml_src / "item11_evgs_rules.csv",
            "evgs_rules_item15": ml_src / "item15_evgs_rules.csv",
            "evgs_rules_item16": ml_src / "item16_evgs_rules.csv",
            "evgs_rules_item17": ml_src / "item17_evgs_rules.csv",
            "item05_classifier_ensemble": ml_src / "item05_classifier_ensemble.csv",
            "model_diversity_item12":     ml_src / "item12_model_diversity.csv",
            "tcn_item14":                 ml_src / "item14_tcn.csv",
        }

    return merge_ml_sources(base_path, sources)


def _track1_items(df: pd.DataFrame,
                  source_name: str = "pre-clinical ML merge") -> pd.DataFrame:
    """Filter to Track-1 rows + cast EVGS items to int + reset index.

    Shared helper used by ``write_preclinical_ml_merge`` and
    ``clinical_consistency.py``. The ``source_name`` is only used in the
    error message when required columns are missing.
    """
    item_cols = [f"L{i}" for i in range(1, 18)] + [f"R{i}" for i in range(1, 18)]
    required = {"ID", *item_cols}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{source_name} is missing required columns: {sorted(missing)}")
    out = df.loc[df["ID"].astype(str).str.startswith("track1"), ["ID", *item_cols]].copy()
    for col in item_cols:
        out[col] = out[col].astype(int)
    return out.reset_index(drop=True)


def write_preclinical_ml_merge(
    repo_root: str | pathlib.Path,
    output: str | pathlib.Path | None = None,
    use_runtime: bool = True,
) -> pathlib.Path:
    """Write the Track-1 pre-clinical ML-merge checkpoint.

    ``use_runtime=False`` writes the committed reference checkpoint into the
    requested output path.  ``use_runtime=True`` recomputes the checkpoint
    from the available baseline and per-item source CSVs.
    """
    root = pathlib.Path(repo_root)
    out_path = pathlib.Path(output) if output else root / RUNTIME_PRECLINICAL_ML_MERGE
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if use_runtime:
        df = merge_ml_sources_from_default_layout(root, use_runtime=True)
        preclinical = _track1_items(df)
    else:
        ref_path = root / PRECLINICAL_ML_MERGE_REF
        if not ref_path.exists():
            raise FileNotFoundError(f"pre-clinical ML-merge reference not found: {ref_path}")
        preclinical = _track1_items(pd.read_csv(ref_path))

    preclinical.to_csv(out_path, index=False)
    return out_path


__all__ = [
    "ITEM_SOURCE_MAP",
    "DEFAULT_BASE_REL",
    "DEFAULT_ML_SOURCES_REL",
    "PRECLINICAL_ML_MERGE_REF",
    "RUNTIME_PRECLINICAL_ML_MERGE",
    "merge_ml_sources",
    "merge_ml_sources_from_default_layout",
    "write_preclinical_ml_merge",
]
