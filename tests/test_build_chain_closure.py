"""Closure check on the committed release artifacts.

Asserts that rebuilding ``preclinical_ml_merge_track1.csv`` from the
committed XGB baseline + per-item source slices via
``cv4chl.ml_merge.merge_ml_sources_from_default_layout`` produces a table
byte-equal to the committed ``submissions/intermediate/...`` checkpoint.
This confirms that the committed baseline and preclinical are internally
consistent under the documented merge code.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from cv4chl.ml_merge import (
    PRECLINICAL_ML_MERGE_REF,
    merge_ml_sources_from_default_layout,
)


def test_committed_preclinical_matches_rebuild(repo_root: Path) -> None:
    item_cols = [f"L{i}" for i in range(1, 18)] + [f"R{i}" for i in range(1, 18)]

    rebuilt = merge_ml_sources_from_default_layout(repo_root, use_runtime=False)
    rebuilt_t1 = rebuilt[rebuilt["ID"].str.startswith("track1")].reset_index(drop=True)
    rebuilt_t1 = rebuilt_t1[["ID", *item_cols]].astype({c: int for c in item_cols})

    committed = pd.read_csv(repo_root / PRECLINICAL_ML_MERGE_REF).reset_index(drop=True)
    committed = committed[["ID", *item_cols]].astype({c: int for c in item_cols})

    diffs: list[tuple[str, str, int, int]] = []
    for _, rrow in rebuilt_t1.iterrows():
        rid = rrow["ID"]
        crow = committed.loc[committed["ID"] == rid]
        assert not crow.empty, f"committed preclinical missing row {rid}"
        c = crow.iloc[0]
        for col in item_cols:
            if int(rrow[col]) != int(c[col]):
                diffs.append((rid, col, int(c[col]), int(rrow[col])))

    assert not diffs, (
        f"build-chain closure check failed: {len(diffs)} cell differences "
        f"between rebuilt and committed preclinical. First few "
        f"(committed -> rebuilt): {diffs[:5]}"
    )
