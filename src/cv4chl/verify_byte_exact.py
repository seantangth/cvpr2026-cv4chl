#!/usr/bin/env python3
"""Standalone CLI: verify the submitted CSV artifact hash.

By default this hashes ``submissions/final/final_submission.csv`` and
compares it with the recorded submitted-file SHA-256. If ``--reproduced`` is
provided, it also compares an existing retrained/reproduced CSV against that
submitted artifact.

Exit codes
----------
0
    Submitted artifact hash confirmed, and reproduced file matched if provided.
1
    Mismatch (a compact cell-level diff is printed to stderr).
2
    I/O or configuration error.

Usage
-----
::

    # Verify the submitted artifact hash
    python src/cv4chl/verify_byte_exact.py

    # Compare an existing retrained CSV to the submitted artifact
    python src/cv4chl/verify_byte_exact.py --reproduced 5_outputs/submissions/retrained_final.csv
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_TARGET = REPO_ROOT / "submissions/final/final_submission.csv"
EXPECTED_SUBMITTED_SHA = "cfebf119712f900288156a8b409442a792aef677b49c66358d52ec3399905a26"


def sha256_of_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--target", type=pathlib.Path, default=DEFAULT_TARGET,
        help="The shipped final CSV to compare against.",
    )
    p.add_argument("--reproduced", type=pathlib.Path, default=None,
                   help="Path to the reproduced CSV.")
    args = p.parse_args(argv)

    reproduced = args.reproduced

    if not args.target.exists():
        print(f"ERROR: required input not found: {args.target}", file=sys.stderr)
        return 2

    h_tgt = sha256_of_file(args.target)
    print(f"target:     {args.target}  sha256 = {h_tgt}")
    print(f"expected:   {EXPECTED_SUBMITTED_SHA}")
    if h_tgt != EXPECTED_SUBMITTED_SHA:
        print("FAIL: submitted artifact hash mismatch", file=sys.stderr)
        return 1

    if reproduced is None:
        print("OK: submitted artifact hash confirmed")
        return 0

    if not reproduced.exists():
        print(f"ERROR: required input not found: {reproduced}", file=sys.stderr)
        return 2

    h_out = sha256_of_file(reproduced)
    print(f"reproduced: {reproduced}  sha256 = {h_out}")
    if h_out == h_tgt:
        print("OK: byte-exact match")
        return 0

    print("FAIL: mismatch", file=sys.stderr)
    import pandas as pd
    a = pd.read_csv(reproduced)
    b = pd.read_csv(args.target)
    merged = a.merge(b, on="ID", suffixes=("_out", "_tgt"))
    item_cols = [c for c in a.columns if c != "ID"]
    for col in item_cols:
        out_col, tgt_col = f"{col}_out", f"{col}_tgt"
        if out_col not in merged or tgt_col not in merged:
            continue
        diff_mask = merged[out_col].astype(str) != merged[tgt_col].astype(str)
        if diff_mask.any():
            for _, row in merged[diff_mask].iterrows():
                print(
                    f"  {row['ID']} {col}: reproduced={row[out_col]} "
                    f"target={row[tgt_col]}",
                    file=sys.stderr,
                )
    return 1


if __name__ == "__main__":
    sys.exit(main())
