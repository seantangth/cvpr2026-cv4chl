"""ID-based final_assembly robustness tests.

Mirrors test_ml_merge_id_join: the Track-1 baseline overlay inside
``cv4chl.final_assembly.assemble_final`` must use ID-based lookup rather
than positional `.values`. If a future training notebook emits
``xgb_baseline.csv`` with rows in a different order, the assembled final
CSV must still be identical.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from cv4chl.final_assembly import (
    BASELINE_REF,
    assemble_final,
)


def _stage_release_layout_into(tmp_root: Path, repo_root: Path) -> None:
    """Stage the committed release layout under tmp_root so we can perturb it."""
    for rel in [
        "submissions/reference/xgb_baseline.csv",
        "submissions/intermediate/track2_classifier_output.csv",
        "submissions/intermediate/preclinical_ml_merge_track1.csv",
        "submissions/item_sources/clinical_consistency_track1.csv",
    ]:
        (tmp_root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo_root / rel, tmp_root / rel)
    src_dir = repo_root / "submissions" / "item_sources"
    dst_dir = tmp_root / "submissions" / "item_sources"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for csv in src_dir.glob("item*.csv"):
        shutil.copy2(csv, dst_dir / csv.name)


def test_assembly_invariant_to_baseline_row_shuffle(
    repo_root: Path, tmp_path: Path
) -> None:
    """Reshuffling baseline.csv rows must NOT change the assembled final CSV.

    A positional `.values` overlay would silently mismerge under reshuffled
    inputs; ID-based lookup is invariant to row order.
    """
    _stage_release_layout_into(tmp_path, repo_root)
    canonical = assemble_final(tmp_path, use_runtime=False)

    baseline_path = tmp_path / BASELINE_REF
    baseline = pd.read_csv(baseline_path)
    shuffled = pd.concat([
        baseline[baseline["ID"].str.startswith("track1")].sample(
            frac=1, random_state=98765
        ),
        baseline[~baseline["ID"].str.startswith("track1")],
    ]).reset_index(drop=True)
    shuffled.to_csv(baseline_path, index=False)

    out = assemble_final(tmp_path, use_runtime=False)
    assert canonical.equals(out), (
        "assembled final CSV changed when baseline.csv rows were reshuffled — "
        "indicates a positional overlay regression in final_assembly.py"
    )


def test_assembly_rejects_baseline_missing_track1_ids(
    repo_root: Path, tmp_path: Path
) -> None:
    """If baseline.csv is missing Track-1 IDs that the track2 base table has,
    fail fast with a clear message rather than silently NaN-filling."""
    _stage_release_layout_into(tmp_path, repo_root)
    baseline_path = tmp_path / BASELINE_REF
    baseline = pd.read_csv(baseline_path)
    truncated = baseline[~(
        baseline["ID"].str.startswith("track1")
        & (baseline.index == baseline.index[0])
    )].copy()
    truncated.to_csv(baseline_path, index=False)

    with pytest.raises(ValueError, match="missing Track-1 IDs"):
        assemble_final(tmp_path, use_runtime=False)
