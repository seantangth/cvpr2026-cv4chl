"""Shared test fixtures and path setup for the CV4CHL pytest suite."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT
