"""Generate the Track-1 clinical-consistency source CSV.

This stage starts from the pre-clinical ML first-pass prediction table and
applies the patient-agnostic clinical-knowledge second pass implemented in
``cv4chl.clinical_rules``.

The second pass inspects only the EVGS first-pass row (item pattern, side
totals, row total, L/R symmetry).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    src_dir = Path(__file__).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

from cv4chl.clinical_rules import TRACK1_ITEM_COLS, apply_track1_second_pass
from cv4chl.final_assembly import CLINICAL_CONSISTENCY_REF
from cv4chl.ml_merge import (
    PRECLINICAL_ML_MERGE_REF,
    _track1_items,
    merge_ml_sources_from_default_layout,
)


FRESH_CLINICAL_CONSISTENCY = Path("5_outputs/submissions/clinical_consistency_track1.csv")
FRESH_CLINICAL_LOG = Path("5_outputs/reports/clinical_second_pass_log.csv")


def load_preclinical_ml_merge(repo_root: str | Path, use_runtime: bool = True) -> pd.DataFrame:
    """Load or compute the Track-1 pre-clinical ML first-pass table.

    Full-retrain mode (``use_runtime=True``) always recomputes the merge
    from the current runtime per-item sources, ensuring the resulting
    checkpoint reflects the freshly trained predictions. The recompute
    call fails fast (with an actionable per-notebook hint) if any runtime
    source is missing. Reviewer reproduction (``use_runtime=False``) reads
    only the committed ``PRECLINICAL_ML_MERGE_REF`` artifact.
    """
    root = Path(repo_root)
    reference_path = root / PRECLINICAL_ML_MERGE_REF

    if use_runtime:
        merged = merge_ml_sources_from_default_layout(root, use_runtime=True)
        return _track1_items(merged, "merge_ml_sources_from_default_layout")

    if not reference_path.exists():
        raise FileNotFoundError(f"pre-clinical ML-merge reference not found: {reference_path}")
    return _track1_items(pd.read_csv(reference_path), str(reference_path))


def apply_clinical_consistency(preclinical: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the clinical-knowledge second-pass filter to every Track-1 row."""
    first_pass = _track1_items(preclinical, "preclinical")
    rows: list[dict[str, int | str]] = []
    log_rows: list[dict[str, object]] = []

    for _, row in first_pass.iterrows():
        row_id = str(row["ID"])
        preds = {col: int(row[col]) for col in TRACK1_ITEM_COLS}
        updated, decisions = apply_track1_second_pass(preds)
        rows.append({"ID": row_id, **updated})
        for decision in decisions:
            log_rows.append({"ID": row_id, **decision})

    clinical = pd.DataFrame(rows, columns=["ID", *TRACK1_ITEM_COLS])
    log = pd.DataFrame(
        log_rows,
        columns=["ID", "column", "before", "after", "rule_id", "evidence"],
    )
    return clinical, log


def _resolve_output(root: Path, output: str | Path | None, default: Path) -> Path:
    out_path = Path(output) if output else root / default
    return out_path if out_path.is_absolute() else root / out_path


def write_clinical_consistency(
    repo_root: str | Path,
    output: str | Path | None = None,
    log_output: str | Path | None = None,
    use_runtime: bool = True,
    verify_reference: bool = False,
) -> tuple[Path, Path]:
    """Generate and write ``clinical_consistency_track1.csv`` and its rule log."""
    root = Path(repo_root)
    out_path = _resolve_output(root, output, FRESH_CLINICAL_CONSISTENCY)
    log_path = _resolve_output(root, log_output, FRESH_CLINICAL_LOG)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    preclinical = load_preclinical_ml_merge(root, use_runtime=use_runtime)
    clinical, log = apply_clinical_consistency(preclinical)
    clinical.to_csv(out_path, index=False)
    log.to_csv(log_path, index=False)

    if verify_reference:
        ref_path = root / CLINICAL_CONSISTENCY_REF
        ref = _track1_items(pd.read_csv(ref_path), str(ref_path))
        generated = _track1_items(pd.read_csv(out_path), str(out_path))
        if not generated.equals(ref):
            raise AssertionError(
                f"generated clinical-consistency CSV differs from {ref_path}"
            )

    return out_path, log_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--log-output", type=Path, default=None)
    parser.add_argument(
        "--release-artifacts",
        action="store_true",
        help="Use the committed pre-clinical checkpoint instead of runtime 5_outputs/.",
    )
    parser.add_argument(
        "--verify-reference",
        action="store_true",
        help="Assert the generated CSV equals submissions/item_sources/clinical_consistency_track1.csv.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_path, log_path = write_clinical_consistency(
        args.repo_root,
        output=args.output,
        log_output=args.log_output,
        use_runtime=not args.release_artifacts,
        verify_reference=args.verify_reference,
    )
    n_decisions = len(pd.read_csv(log_path)) if log_path.exists() else 0
    print(f"clinical consistency source written: {out_path}")
    print(f"clinical second-pass log written: {log_path}")
    print(f"clinical second-pass cells: {n_decisions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
