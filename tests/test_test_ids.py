"""Verify the canonical Track-1 / Track-2 test ID lists match the
committed Track-1 first-pass artifact and the committed final CSV.

The hardcoded lists in the preprocessing notebook and the Track-2
classifier notebook must match the IDs that actually appear in
``submissions/final/final_submission.csv`` and the preclinical merge.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from cv4chl.ml_merge import PRECLINICAL_ML_MERGE_REF


CANONICAL_TRACK1_TEST_IDS = [4, 5, 18, 26, 28, 40, 42, 43, 47, 48, 53, 54, 72, 78, 83, 85]
CANONICAL_TRACK2_TEST_IDS = [4, 6, 7, 13, 26, 35, 39, 42, 50]


def _ids_with_prefix(df: pd.DataFrame, prefix: str) -> list[int]:
    return sorted(
        int(s.split("-")[1])
        for s in df["ID"].astype(str)
        if s.startswith(prefix)
    )


def test_committed_preclinical_track1_ids(repo_root: Path) -> None:
    df = pd.read_csv(repo_root / PRECLINICAL_ML_MERGE_REF)
    assert _ids_with_prefix(df, "track1-") == sorted(CANONICAL_TRACK1_TEST_IDS)


def test_committed_final_csv_track_ids(repo_root: Path) -> None:
    df = pd.read_csv(repo_root / "submissions" / "final" / "final_submission.csv")
    assert _ids_with_prefix(df, "track1-") == sorted(CANONICAL_TRACK1_TEST_IDS)
    assert _ids_with_prefix(df, "track2-") == sorted(CANONICAL_TRACK2_TEST_IDS)
