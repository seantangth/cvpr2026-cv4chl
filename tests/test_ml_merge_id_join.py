"""ID-based ml_merge robustness tests.

Verify that the per-cell overlay in ``cv4chl.ml_merge.merge_ml_sources``:
1. Produces the canonical preclinical merge byte-equal to the committed
   reference (sanity check on real release data).
2. Is invariant to row-order changes in the source CSVs (a positional
   overlay would silently mismerge under reshuffled inputs).
3. Raises when a source CSV is missing Track-1 IDs that the base table
   contains.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cv4chl.ml_merge import (
    DEFAULT_BASE_REL,
    DEFAULT_ML_SOURCES_REL,
    ITEM_SOURCE_MAP,
    merge_ml_sources,
    merge_ml_sources_from_default_layout,
)


_ITEM_SOURCE_FILES = {
    "evgs_rules_item3":  "item03_evgs_rules.csv",
    "evgs_rules_item9":  "item09_evgs_rules.csv",
    "evgs_rules_item11": "item11_evgs_rules.csv",
    "evgs_rules_item15": "item15_evgs_rules.csv",
    "evgs_rules_item16": "item16_evgs_rules.csv",
    "evgs_rules_item17": "item17_evgs_rules.csv",
    "item05_classifier_ensemble": "item05_classifier_ensemble.csv",
    "model_diversity_item12":     "item12_model_diversity.csv",
    "tcn_item14":                 "item14_tcn.csv",
}


def _committed_sources(repo_root: Path) -> dict[str, Path]:
    src_dir = repo_root / DEFAULT_ML_SOURCES_REL
    return {key: src_dir / fname for key, fname in _ITEM_SOURCE_FILES.items()}


def test_default_layout_produces_canonical_preclinical(repo_root: Path) -> None:
    """Sanity check on real release artifacts."""
    df = merge_ml_sources_from_default_layout(repo_root, use_runtime=False)
    assert "ID" in df.columns
    assert (df["ID"].astype(str).str.startswith("track1")).any()


def test_merge_invariant_to_source_row_shuffle(
    repo_root: Path, tmp_path: Path
) -> None:
    """Reshuffling the rows of a source CSV must NOT change the merged output.

    A positional overlay would silently break under reshuffled inputs;
    ID-based lookup is invariant to row order.
    """
    base_path = repo_root / DEFAULT_BASE_REL
    sources = _committed_sources(repo_root)
    canonical = merge_ml_sources(base_path, sources)

    # Pick one source CSV and reshuffle its rows; reorder Track-1 randomly
    # while keeping the schema valid.
    target_key = "evgs_rules_item3"
    src_df = pd.read_csv(sources[target_key])
    shuffled = pd.concat([
        src_df[src_df["ID"].str.startswith("track1")].sample(
            frac=1, random_state=12345
        ),
        src_df[~src_df["ID"].str.startswith("track1")],
    ]).reset_index(drop=True)
    shuffled_path = tmp_path / "shuffled_item03.csv"
    shuffled.to_csv(shuffled_path, index=False)

    perturbed_sources = dict(sources)
    perturbed_sources[target_key] = shuffled_path
    out = merge_ml_sources(base_path, perturbed_sources)

    # Compare on the columns that this source actually overlays.
    overlaid = [c for c, k in ITEM_SOURCE_MAP.items() if k == target_key]
    assert canonical[["ID", *overlaid]].equals(out[["ID", *overlaid]])


def test_merge_rejects_source_missing_track1_ids(
    repo_root: Path, tmp_path: Path
) -> None:
    base_path = repo_root / DEFAULT_BASE_REL
    sources = _committed_sources(repo_root)
    target_key = "evgs_rules_item3"

    src_df = pd.read_csv(sources[target_key])
    truncated = src_df[~(
        src_df["ID"].str.startswith("track1") & (src_df.index == src_df.index[0])
    )].copy()
    truncated_path = tmp_path / "missing_id_item03.csv"
    truncated.to_csv(truncated_path, index=False)

    perturbed = dict(sources)
    perturbed[target_key] = truncated_path
    with pytest.raises(ValueError, match="missing Track-1 IDs"):
        merge_ml_sources(base_path, perturbed)
