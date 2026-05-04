#!/usr/bin/env python3
"""End-to-end reproduction of `final_submission.csv`.

Pipeline orchestrated by this script::

    raw data (Kaggle CGPS)
      │  1. preprocess
      ▼
    1_data/processed/{keypoints, train_ready}.pkl
      │  2. 6 training notebooks
      ▼
    5_outputs/submissions/
      │  3. pre-clinical ML merge checkpoint
      ▼
    5_outputs/submissions/preclinical_ml_merge_track1.csv
      │  4. clinical-knowledge second-pass consistency filter
      ▼
    5_outputs/submissions/clinical_consistency_track1.csv
      │  5. deterministic assembly module
      │     - apply generated clinical consistency source
      │     - recompute Total
      ▼
    submissions/final/final_submission.csv
      │  6. SHA-256 compare against shipped reference
      ▼
    OK / FAIL

Usage
-----
::

    python reproduce.py                  # full retrain (uses GPU for TCN if present)
    python reproduce.py --skip-tcn       # use shipped item-14 TCN slice; retrain everything else
    python reproduce.py --skip-train     # only run deterministic assembly from release artifacts
    python reproduce.py --dry-run-tcn    # 2-min TCN smoke test (final CSV will NOT match)
    python reproduce.py --from clinical_consistency --to clinical_consistency
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# --------------------------------------------------------------------- #
# Stage definitions                                                     #
# --------------------------------------------------------------------- #

NB_STAGES: list[tuple[str, str, str]] = [
    ("preprocess", "Preprocess raw keypoints + EVGS labels → train_ready.pkl",
     "01_preprocess/CV4CHL_01_preprocess_features.ipynb"),
    ("xgb_baseline", "Train 17 XGBoost classifiers + Track-2 multiclass baseline",
     "02_train_pipeline/CV4CHL_02_train_xgb_baseline.ipynb"),
    ("track2_clinical", "Track-2 Rodda & Graham clinical classifier",
     "02_train_pipeline/CV4CHL_03_train_track2_clinical_classifier.ipynb"),
    ("evgs_rule_sources", "EVGS angle-threshold sources for items {3,9,11,15,16,17}",
     "02_train_pipeline/CV4CHL_04_train_evgs_rule_sources.ipynb"),
    ("model_diversity_item12", "CatBoost / LightGBM model diversity for item 12 R-side",
     "02_train_pipeline/CV4CHL_05_train_model_diversity_item12.ipynb"),
    ("item05_classifier_ensemble", "Classifier ensemble for item 5 (XGB+LGB+CatBoost)",
     "02_train_pipeline/CV4CHL_06_train_item05_classifier_ensemble.ipynb"),
    ("tcn_item14", "1D-TCN for item 14 (L14 + R14)",
     "02_train_pipeline/CV4CHL_07_train_tcn_item14.ipynb"),
]
SCRIPT_STAGES: dict[str, str] = {
    "preclinical_ml_merge": "Write the Track-1 pre-clinical ML-merge checkpoint",
    "clinical_consistency": "Apply the clinical-knowledge second-pass filter",
    "assemble": "Assemble the final CSV and recompute Track-1 Total",
}
ALL_STAGES = [s[0] for s in NB_STAGES] + list(SCRIPT_STAGES) + ["verify"]
NOTEBOOK_STAGE_NAMES = {s[0] for s in NB_STAGES}

# Input scaffolding only: training notebooks 04-07 each load
# `5_outputs/submissions/reference_baseline.csv` as the base table they
# overlay their per-item predictions onto. Notebook 02 (XGB baseline)
# overwrites this file with its freshly trained output before any later
# notebook reads it on a default full-retrain run, so the staging is only
# load-bearing on resume runs that skip notebook 02.
#
# `5_outputs/submissions/xgb_baseline.csv` is intentionally NOT staged
# here: it is a runtime artifact produced by notebook 02 and consumed by
# the assembly stage. Resume runs that skip notebook 02 must use
# --skip-train, which reads from the committed release artifacts directly.
REFERENCE_STAGING: list[tuple[str, str]] = [
    ("reference/xgb_baseline.csv", "5_outputs/submissions/reference_baseline.csv"),
]

# When --skip-tcn is set, the committed item-14 slice is staged into the
# canonical runtime path so downstream stages see it as the TCN notebook's
# fresh output. This mirrors the documented --skip-tcn semantics: skip
# only the training, reuse the shipped per-item-14 prediction.
SKIP_TCN_STAGING: tuple[str, str] = (
    "item_sources/item14_tcn.csv",
    "5_outputs/submissions/tcn_item14/item14_tcn.csv",
)

REQUIRED_RAW: list[str] = [
    "1_data/raw/track1_train.json",
    "1_data/raw/track2_train.json",
    "1_data/raw/dataset",
]

REQUIRED_PROCESSED: list[str] = [
    "1_data/processed/keypoints.pkl",
    "1_data/processed/train_ready.pkl",
]

TARGET_FINAL = REPO_ROOT / "submissions" / "final" / "final_submission.csv"
RUNTIME_PRECLINICAL = REPO_ROOT / "5_outputs" / "submissions" / "preclinical_ml_merge_track1.csv"
RUNTIME_CLINICAL = REPO_ROOT / "5_outputs" / "submissions" / "clinical_consistency_track1.csv"


# --------------------------------------------------------------------- #
# Console output helpers                                                #
# --------------------------------------------------------------------- #

def ui(level: str, msg: str) -> None:
    sym = {"step": "▶", "ok": "✓", "warn": "⚠", "fail": "✗", "info": "·"}.get(level, " ")
    print(f"  {sym} {msg}", flush=True)


def banner(title: str) -> None:
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}", flush=True)


def ts() -> str:
    return time.strftime("%H:%M:%S")


# --------------------------------------------------------------------- #
# Data-presence checks                                                  #
# --------------------------------------------------------------------- #

def check_raw_data() -> bool:
    missing = [p for p in REQUIRED_RAW if not (REPO_ROOT / p).exists()]
    if not missing:
        ui("ok", f"raw data present at {REPO_ROOT / '1_data/raw'}")
        return True
    ui("fail", "raw data missing:")
    for m in missing:
        print(f"      - {REPO_ROOT / m}")
    print(
        "\n  Download the CGPS dataset (1.07 GB) from Kaggle:\n"
        "    https://www.kaggle.com/competitions/cvpr-2026-the-first-ai-children-challenge/data\n"
        "  Required files (place under 1_data/raw/):\n"
        "    - track1_train.json\n"
        "    - track2_train.json\n"
        "    - dataset.tar.bz2  (extract to 1_data/raw/dataset/)\n",
        flush=True,
    )
    return False


def check_processed() -> bool:
    return all((REPO_ROOT / p).exists() for p in REQUIRED_PROCESSED)


# --------------------------------------------------------------------- #
# Stage 1: stage anchor CSVs                                            #
# --------------------------------------------------------------------- #

def stage_reference_inputs() -> None:
    for src_rel, dst_rel in REFERENCE_STAGING:
        src = REPO_ROOT / "submissions" / src_rel
        dst = REPO_ROOT / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            raise FileNotFoundError(f"reference input missing: {src}")
        shutil.copy2(src, dst)
        ui("ok", f"staged  {dst.relative_to(REPO_ROOT)}")


def stage_skip_tcn_artifact() -> None:
    """Stage the committed item-14 slice as the canonical runtime output.

    Used only when --skip-tcn is active. This is the one place in the
    pipeline where a committed slice is intentionally promoted to a
    runtime path, marked with an explicit log line at staging time.
    """
    src_rel, dst_rel = SKIP_TCN_STAGING
    src = REPO_ROOT / "submissions" / src_rel
    dst = REPO_ROOT / dst_rel
    if not src.exists():
        raise FileNotFoundError(
            f"--skip-tcn requires committed item-14 slice at {src}"
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    ui("ok", f"staged  {dst.relative_to(REPO_ROOT)}  (--skip-tcn item-14 reuse)")


def run_preclinical_ml_merge(use_runtime: bool = True) -> Path:
    from cv4chl.ml_merge import write_preclinical_ml_merge

    ui("step", f"[{ts()}] preclinical_ml_merge  {RUNTIME_PRECLINICAL.relative_to(REPO_ROOT)}")
    t0 = time.time()
    out_path = write_preclinical_ml_merge(
        REPO_ROOT,
        RUNTIME_PRECLINICAL,
        use_runtime=use_runtime,
    )
    ui("ok", f"[{ts()}] wrote     {out_path.relative_to(REPO_ROOT)}  ({time.time()-t0:.1f}s)")
    return out_path


def run_clinical_consistency(use_runtime: bool = True) -> Path:
    import pandas as pd
    from cv4chl.clinical_consistency import write_clinical_consistency

    ui("step", f"[{ts()}] clinical_consistency  {RUNTIME_CLINICAL.relative_to(REPO_ROOT)}")
    t0 = time.time()
    # verify_reference=True compares the regenerated CSV against the committed
    # submissions/item_sources/clinical_consistency_track1.csv. That match is
    # only guaranteed in --skip-train mode (deterministic post-training stages
    # over committed inputs). In full-retrain mode the freshly trained sources
    # may produce a legitimately different intermediate, and the byte-exactness
    # guarantee belongs to the SHA-256 check on the *final* CSV, not on this
    # intermediate. Tying verify_reference to use_runtime also stops the clean-
    # room rerun (which hides submissions/item_sources/) from crashing here.
    out_path, log_path = write_clinical_consistency(
        REPO_ROOT,
        RUNTIME_CLINICAL,
        use_runtime=use_runtime,
        verify_reference=not use_runtime,
    )
    n_decisions = len(pd.read_csv(log_path)) if log_path.exists() else 0
    ui("ok", f"[{ts()}] wrote     {out_path.relative_to(REPO_ROOT)}  "
             f"({n_decisions} second-pass cells, {time.time()-t0:.1f}s)")
    return out_path


def run_assembly(use_runtime: bool = True,
                 clinical_source_path: Path | None = None) -> None:
    from cv4chl.final_assembly import write_final

    ui("step", f"[{ts()}] assemble  submissions/final/{TARGET_FINAL.name}")
    t0 = time.time()
    out_path = write_final(
        REPO_ROOT,
        TARGET_FINAL,
        use_runtime=use_runtime,
        clinical_source_path=clinical_source_path,
    )
    ui("ok", f"[{ts()}] wrote     {out_path.relative_to(REPO_ROOT)}  ({time.time()-t0:.1f}s)")


# --------------------------------------------------------------------- #
# Notebook execution                                                    #
# --------------------------------------------------------------------- #

def run_notebook(rel_path: str, extra_env: dict[str, str] | None = None,
                 timeout: int = 7200) -> None:
    nb_path = REPO_ROOT / "notebooks" / rel_path
    if not nb_path.exists():
        raise FileNotFoundError(f"notebook not found: {nb_path}")
    env = dict(os.environ)
    env.setdefault("CV4CHL_REPO_ROOT", str(REPO_ROOT))
    if extra_env:
        env.update(extra_env)
    cmd = [
        sys.executable, "-m", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        f"--ExecutePreprocessor.timeout={timeout}",
        str(nb_path),
    ]
    ui("step", f"[{ts()}] execute  {rel_path}")
    t0 = time.time()
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        ui("fail", f"notebook failed: {rel_path} (exit {e.returncode})")
        raise
    ui("ok", f"[{ts()}] done     {rel_path}  ({time.time()-t0:.1f}s)")


# --------------------------------------------------------------------- #
# Verification                                                          #
# --------------------------------------------------------------------- #

def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


EXPECTED_FINAL_SHA256 = "cfebf119712f900288156a8b409442a792aef677b49c66358d52ec3399905a26"


def run_verify(strict: bool = True) -> bool:
    """Verify the final CSV hash. ``strict=True`` returns False on mismatch
    (used by ``--skip-train``, which must be byte-exact). ``strict=False``
    returns True with a warning on mismatch (used by full retrain, which is
    documented as functionally equivalent but not necessarily byte-exact —
    see README "Byte-exact reproduction scope")."""
    if not TARGET_FINAL.exists():
        ui("fail", f"shipped target missing: {TARGET_FINAL}")
        return False
    h_target = sha256_of(TARGET_FINAL)
    print(f"\n  final csv sha256: {h_target}")
    print(f"  expected sha256:  {EXPECTED_FINAL_SHA256}")
    if h_target == EXPECTED_FINAL_SHA256:
        ui("ok", "byte-exact reproduction confirmed")
        return True
    if strict:
        ui("fail", "SHA-256 differs from the published reference (--skip-train "
                   "must be byte-exact).")
        return False
    ui("warn", "SHA-256 differs from the published reference. Full retrain is "
               "documented as functionally equivalent, not byte-exact (training-time "
               "non-determinism across hardware / library minor versions). Use "
               "--skip-train for the byte-exact check.")
    return True


def run_build_chain_self_check() -> bool:
    """Closure check on the committed release artifacts.

    Asserts: rebuilding the Track-1 pre-clinical merge from
        - submissions/reference/xgb_baseline.csv
        - submissions/item_sources/*.csv
        - src/cv4chl/ml_merge.py
    produces a table byte-equal to the committed
        submissions/intermediate/preclinical_ml_merge_track1.csv.

    This closure check ensures that --skip-train (which uses committed
    intermediates) cannot diverge from full retrain (which uses runtime
    artifacts under 5_outputs/). It catches the failure mode where the
    committed baseline and committed preclinical are not derivable from
    each other under the documented merge code in src/cv4chl/ml_merge.py.
    """
    import pandas as pd

    from cv4chl.ml_merge import (
        DEFAULT_BASE_REL,
        DEFAULT_ML_SOURCES_REL,
        PRECLINICAL_ML_MERGE_REF,
        merge_ml_sources_from_default_layout,
    )

    # Friendly pre-check: if the committed release artifacts that this
    # closure check inspects are absent (e.g. user followed the README
    # clean-room recipe but forgot --skip-release-self-check), point them
    # at the right flag instead of letting pandas raise a bare
    # FileNotFoundError several layers deep.
    needed = [
        REPO_ROOT / DEFAULT_BASE_REL,
        REPO_ROOT / PRECLINICAL_ML_MERGE_REF,
        REPO_ROOT / DEFAULT_ML_SOURCES_REL,
    ]
    missing_release = [p for p in needed if not p.exists()]
    if missing_release:
        ui("fail",
           "build-chain self-check requires committed release artifacts that "
           "are not present:\n      " +
           "\n      ".join(f"- {p}" for p in missing_release) +
           "\n      If you intentionally hid these (clean-room reproducibility "
           "recipe in README), re-run with --skip-release-self-check.")
        return False

    item_cols = [f"L{i}" for i in range(1, 18)] + [f"R{i}" for i in range(1, 18)]
    rebuilt = merge_ml_sources_from_default_layout(REPO_ROOT, use_runtime=False)
    rebuilt_t1 = rebuilt[rebuilt["ID"].str.startswith("track1")].reset_index(drop=True)
    rebuilt_t1 = rebuilt_t1[["ID", *item_cols]].astype({c: "int" for c in item_cols})

    committed_path = REPO_ROOT / PRECLINICAL_ML_MERGE_REF
    if not committed_path.exists():
        ui("fail", f"committed preclinical missing: {committed_path}")
        return False
    committed = pd.read_csv(committed_path).reset_index(drop=True)
    committed = committed[["ID", *item_cols]].astype({c: "int" for c in item_cols})

    diffs: list[tuple[str, str, int, int]] = []
    for _, rrow in rebuilt_t1.iterrows():
        rid = rrow["ID"]
        crow_match = committed.loc[committed["ID"] == rid]
        if crow_match.empty:
            ui("fail", f"committed preclinical missing row {rid}")
            return False
        crow = crow_match.iloc[0]
        for col in item_cols:
            if int(rrow[col]) != int(crow[col]):
                diffs.append((rid, col, int(crow[col]), int(rrow[col])))

    if not diffs:
        ui("ok", "build-chain self-consistency confirmed "
                 "(rebuilt preclinical byte-equal committed checkpoint)")
        return True

    ui("fail",
       f"build-chain self-consistency FAILED: {len(diffs)} cell differences "
       f"between rebuilt and committed preclinical.\n"
       f"      The committed submissions/reference/xgb_baseline.csv and committed\n"
       f"      submissions/intermediate/preclinical_ml_merge_track1.csv are not\n"
       f"      derivable from each other via merge_ml_sources_from_default_layout.")
    for rid, col, c, r in diffs[:10]:
        print(f"      first diffs (committed -> rebuilt): {rid} {col}: {c} -> {r}")
    if len(diffs) > 10:
        print(f"      ... and {len(diffs) - 10} more")
    return False


# --------------------------------------------------------------------- #
# Main                                                                  #
# --------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--skip-train", action="store_true",
                   help="Skip preprocessing + 6 training notebooks; only run assembly + verify.")
    p.add_argument("--skip-tcn", action="store_true",
                   help="Skip item-14 TCN; reuse shipped item-14 source slice.")
    p.add_argument("--dry-run-tcn", action="store_true",
                   help="Run TCN in 2-min DRY_RUN mode (final CSV will NOT match).")
    p.add_argument("--skip-release-self-check", action="store_true",
                   help=(
                       "Skip the build-chain closure check on committed release artifacts. "
                       "Use this for the clean-room reproducibility test where "
                       "submissions/item_sources/ and submissions/intermediate/ are "
                       "intentionally hidden — without this flag the self-check would fail "
                       "at Stage 0 because it inspects those committed files."
                   ))
    p.add_argument("--from", dest="from_stage", choices=ALL_STAGES, default=None,
                   help="Resume from this stage (skip earlier).")
    p.add_argument("--to", dest="to_stage", choices=ALL_STAGES, default=None,
                   help="Stop after this stage.")
    return p.parse_args()


def stages_to_run(from_stage: str | None, to_stage: str | None) -> list[str]:
    start = ALL_STAGES.index(from_stage) if from_stage else 0
    end = ALL_STAGES.index(to_stage) + 1 if to_stage else len(ALL_STAGES)
    return ALL_STAGES[start:end]


def main() -> int:
    args = parse_args()

    banner("CV4CHL 2026 — End-to-end reproduction of final CSV")
    print(f"  Repo root : {REPO_ROOT}")
    print(f"  Python    : {sys.executable}")
    print(f"  Stages    : {ALL_STAGES}")

    selected = stages_to_run(args.from_stage, args.to_stage)
    if args.skip_train:
        selected = [s for s in selected if s not in NOTEBOOK_STAGE_NAMES]
    nb_selected = [s for s in selected if s in NOTEBOOK_STAGE_NAMES]
    script_selected = [s for s in selected if s in SCRIPT_STAGES]
    print(f"  Will run  : {selected}")

    banner("Stage 0: Verify data presence + resolve assembly inputs")
    # Build-chain self-check on committed release artifacts. This runs first
    # because it is cheap (~50ms) and a failure means the package itself is
    # internally inconsistent — no point burning a full GPU retrain only to
    # find the committed checkpoints don't match the documented merge code.
    #
    # The clean-room reproducibility test (README "Clean-room reproducibility
    # check") deliberately hides submissions/item_sources/ and
    # submissions/intermediate/ to verify that nothing reads from them during
    # full retrain. The self-check itself reads those files, so it has to be
    # opt-out via --skip-release-self-check for that scenario.
    if args.skip_release_self_check:
        ui("info", "skipping build-chain self-check (--skip-release-self-check)")
    elif not run_build_chain_self_check():
        return 2
    if "preprocess" in nb_selected:
        if not check_raw_data():
            return 2
    needs_processed = any(
        s in nb_selected and s != "preprocess"
        for s, _, _ in NB_STAGES
    )
    if needs_processed and not check_processed() and not args.skip_train:
        ui("warn", "processed pkl files missing — run with --from preprocess")
        return 2
    if args.skip_train:
        ui("info", "using committed release artifacts; no runtime staging needed")
    else:
        stage_reference_inputs()
        if args.skip_tcn:
            stage_skip_tcn_artifact()
        # Surface the case where a resume run skips notebook 02 (XGB baseline) — the
        # staged committed reference_baseline.csv will then act as the "fresh" base
        # for notebooks 04-07 instead of being overwritten by a freshly trained
        # output. Documented behavior, but make it visible so a reviewer is not
        # surprised when comparing fresh notebook outputs against a frozen base.
        if args.from_stage and "xgb_baseline" not in nb_selected:
            ui("warn",
               f"resume from {args.from_stage}: notebook 02 (xgb_baseline) is not in "
               f"this run, so 5_outputs/submissions/reference_baseline.csv will use "
               f"the committed frozen baseline as the base table for notebooks 04-07.")

    if nb_selected:
        banner(f"Training stages: Run {len(nb_selected)} notebook(s)")
        for stage_name, desc, nb_rel in NB_STAGES:
            if stage_name not in nb_selected:
                continue
            if args.skip_tcn and stage_name == "tcn_item14":
                ui("warn", "skipping item-14 TCN (--skip-tcn); will reuse shipped item14_tcn.csv")
                continue
            print(f"\n--- Stage: {stage_name}  ({desc}) ---")
            env_over: dict[str, str] = {}
            if stage_name == "tcn_item14" and args.dry_run_tcn:
                env_over["CV4CHL_DRY_RUN"] = "1"
                ui("warn", "TCN in DRY_RUN mode (3 items, 5 epochs, ~2 min)")
            run_notebook(nb_rel, env_over)

    generated_clinical: Path | None = None
    if script_selected:
        banner(f"Deterministic stages: Run {len(script_selected)} stage(s)")
        for stage_name in script_selected:
            print(f"\n--- Stage: {stage_name}  ({SCRIPT_STAGES[stage_name]}) ---")
            if stage_name == "preclinical_ml_merge":
                run_preclinical_ml_merge(use_runtime=not args.skip_train)
            elif stage_name == "clinical_consistency":
                generated_clinical = run_clinical_consistency(use_runtime=not args.skip_train)
            elif stage_name == "assemble":
                run_assembly(
                    use_runtime=not args.skip_train,
                    clinical_source_path=generated_clinical,
                )

    if "verify" in selected:
        banner("Final stage: SHA-256 byte-exact verification")
        ok = run_verify(strict=args.skip_train)
        return 0 if ok else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
