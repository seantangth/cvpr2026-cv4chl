"""End-to-end byte-exact reproduction check.

Runs ``reproduce.py --skip-train`` and asserts the resulting final CSV
matches the published SHA-256. This is the strongest single test in the
suite because it exercises the whole post-training pipeline and confirms
that no committed artifact has drifted.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

EXPECTED_FINAL_SHA256 = (
    "cfebf119712f900288156a8b409442a792aef677b49c66358d52ec3399905a26"
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def test_skip_train_reproduces_final_hash(repo_root: Path) -> None:
    final_csv = repo_root / "submissions" / "final" / "final_submission.csv"
    result = subprocess.run(
        [sys.executable, str(repo_root / "reproduce.py"), "--skip-train"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"reproduce.py --skip-train failed (exit {result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert final_csv.exists(), f"final CSV missing after run: {final_csv}"
    assert _sha256(final_csv) == EXPECTED_FINAL_SHA256, (
        "final CSV SHA-256 differs from the published reference"
    )
