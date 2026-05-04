"""cv4chl — CVPR 2026 AI for Children Challenge solution package.

Lightweight utility package for the submitted solution to
[CVPR 2026] The First AI for Children Challenge (Kaggle competition
``cvpr-2026-the-first-ai-children-challenge``).

The training and clinical post-processing logic live in the Jupyter
notebooks under ``notebooks/02_train_pipeline/``; this package
provides only thin Python utilities.

Sub-modules
-----------
``ml_merge``
    Per-item ensemble merge: combines the per-stage ML prediction CSVs
    (XGBoost base, EVGS rules, item-12 model diversity, item-5 probability
    ensemble, and item-14 1D-TCN) into a single per-cell baseline
    that is then refined by deterministic assembly.
``verify_byte_exact``
    CLI helper that verifies the historical submitted CSV hash and can
    compare an existing retrained CSV against it.
``clinical_consistency``
    Generates the Track-1 clinical-consistency source from the pre-clinical
    ML-merge checkpoint and a documented clinical-knowledge second pass.
``final_assembly``
    Deterministic retrained CSV assembly from runtime model outputs and
    selected item-source artifacts.
"""

from __future__ import annotations

__all__ = [
    "ml_merge",
    "clinical_consistency",
    "verify_byte_exact",
    "final_assembly",
]

__version__ = "1.0.0"
