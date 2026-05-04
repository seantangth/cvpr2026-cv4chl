"""Smoke + log-volume tests for the Track-1 second-pass clinical rules.

These cover:
1. Idempotency on already-clean inputs (all-zero EVGS row -> no change).
2. Determinism (same input -> same output / same log).
3. Cell-change count regression on the committed preclinical checkpoint.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from cv4chl.clinical_rules import TRACK1_ITEM_COLS, apply_track1_second_pass
from cv4chl.ml_merge import PRECLINICAL_ML_MERGE_REF


EXPECTED_SECOND_PASS_CELLS = 37


def test_apply_second_pass_is_a_noop_for_all_zero_row() -> None:
    preds = {col: 0 for col in TRACK1_ITEM_COLS}
    out, log = apply_track1_second_pass(preds)
    assert out == preds
    assert log == []


def test_apply_second_pass_is_deterministic() -> None:
    preds = {col: (1 if i % 3 == 0 else 0) for i, col in enumerate(TRACK1_ITEM_COLS)}
    out1, log1 = apply_track1_second_pass(preds)
    out2, log2 = apply_track1_second_pass(preds)
    assert out1 == out2
    assert log1 == log2


def test_committed_preclinical_yields_expected_cell_count(repo_root: Path) -> None:
    """Regression test: the documented second-pass rules produce the
    expected number of cell changes on the committed preclinical
    checkpoint. A failure here surfaces when rule logic or upstream
    predictions change unexpectedly."""
    df = pd.read_csv(repo_root / PRECLINICAL_ML_MERGE_REF)
    df = df[df["ID"].astype(str).str.startswith("track1")].reset_index(drop=True)

    total_changes = 0
    for _, row in df.iterrows():
        preds = {col: int(row[col]) for col in TRACK1_ITEM_COLS}
        _, log = apply_track1_second_pass(preds)
        total_changes += len(log)

    assert total_changes == EXPECTED_SECOND_PASS_CELLS, (
        f"clinical second-pass changed {total_changes} cells, expected "
        f"{EXPECTED_SECOND_PASS_CELLS} (see README 'Clinical post-processing')."
    )
