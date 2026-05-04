"""Clinical post-processing rules for CV4CHL Track-1 EVGS predictions.

Implements the Track-1 second-pass rules applied to the per-item ML
ensemble first-pass output. Rules trigger from broad EVGS-row statistics
(row total, side totals, isolated-positive patterns, type-3 marker counts).

Each rule cites cross-track training rates and clinical literature
(Rodda & Graham 2004 doi:10.1302/0301-620X.86B2.13878;
Wren et al. 2005 PMID:15614065;
Ong, Hillman & Robb 2008 PMID:18328710;
Kanko et al. 2021 (markerless 2D pose limitations);
Read et al. 2003 PMID:12724590) in its inline comment.

The Track-2 gait-subtype classifier lives alongside its training and
inference flow in
``notebooks/02_train_pipeline/CV4CHL_03_train_track2_clinical_classifier.ipynb``
as ``classify_clinical()``.
"""
from __future__ import annotations


TRACK1_ITEM_COLS = [f"L{i}" for i in range(1, 18)] + [f"R{i}" for i in range(1, 18)]


def _active_items(preds: dict[str, int], side: str) -> set[int]:
    return {i for i in range(1, 18) if int(preds[f"{side}{i}"]) == 1}


def _set_second_pass_value(
    preds: dict[str, int],
    decisions: list[dict[str, object]],
    column: str,
    value: int,
    rule_id: str,
    evidence: str,
) -> None:
    before = int(preds[column])
    if before == value:
        return
    preds[column] = value
    decisions.append({
        "column": column,
        "before": before,
        "after": value,
        "rule_id": rule_id,
        "evidence": evidence,
    })


def apply_track1_second_pass(
    preds: dict[str, int],
) -> tuple[dict[str, int], list[dict[str, object]]]:
    """Apply clinical-knowledge rules to Track-1 EVGS items.

    Each rule fires from broad EVGS-row statistics (row total, side totals,
    L/R asymmetry magnitude, isolated-positive patterns).

    Returns
    -------
    tuple
        ``(updated_predictions, decision_log)`` where ``decision_log`` has
        one entry per changed EVGS cell.
    """
    p = {col: int(preds[col]) for col in TRACK1_ITEM_COLS}
    decisions: list[dict[str, object]] = []

    def side_total(side: str) -> int:
        return sum(p[f"{side}{i}"] for i in range(1, 18))

    def row_total() -> int:
        return side_total("L") + side_total("R")

    def evidence() -> str:
        return (
            f"row_total={row_total()};"
            f"L_total={side_total('L')};R_total={side_total('R')};"
            f"L={sorted(_active_items(p, 'L'))};"
            f"R={sorted(_active_items(p, 'R'))}"
        )

    total = row_total()

    # Type-3 high-rate marker items derived from the cross-track training overlap
    # (n=9 type3 sides among 17 patients × 2 sides). Items with rate >= 0.50,
    # excluding i8 because i8 is the most discriminative type-3 item (rate 0.89)
    # and is treated separately as the "type-3 under-call" indicator.
    TYPE3_SIDE_MARKERS = frozenset({1, 2, 4, 10, 11, 12, 15, 16})

    # Type-3 contralateral consequence pattern: items often present on the
    # mirroring side (i8, i9, i11, i15, i16, i17) when the patient is
    # bilaterally type3. Used by the contralateral confirmation branch.
    TYPE3_CONTRA_MARKERS = frozenset({8, 9, 11, 15, 16, 17})

    # Low-total L4/L8 pair cleanup when companion non-coronal evidence is present.
    # L4 (HindfootVarusValgus) and L8 (KneeProgressionAngle) are coronal/transverse-
    # plane items; from sagittal-view 2D pose keypoints they are correlated
    # first-pass false positives in low-total rows.
    # Item-level reliability (Ong, Hillman, Robb 2008 PMID:18328710): i4 κ≈0.45,
    # i8 κ≈0.22 — among the lowest in EVGS. Markerless 2D pose cannot recover
    # transverse/coronal axial rotation (Kanko et al. 2021).
    if (
        13 <= total <= 14
        and p["L4"] == 1 and p["L8"] == 1
        and any(p[f"L{i}"] == 1 for i in (7, 12, 17))
    ):
        ev = evidence()
        for col in ("L4", "L8"):
            _set_second_pass_value(p, decisions, col, 0, "low_total_l4_l8_pair_filter", ev)

    # Right-dominant low-total rows: left-side swing/proximal positives are
    # treated as likely first-pass over-calls (hemiplegia-side cleanup).
    if side_total("L") <= 5 and side_total("R") - side_total("L") >= 4:
        ev = evidence()
        for col in ("L12", "L13", "L15"):
            _set_second_pass_value(p, decisions, col, 0, "right_dominant_low_total_left_cleanup", ev)

    # Severe bilateral rows: restore missing contralateral item 5 when the
    # opposite side has it and both sides are high-load.
    if total >= 24 and p["L5"] == 1 and p["R5"] == 0 and side_total("R") >= 12:
        _set_second_pass_value(
            p, decisions, "R5", 1, "severe_bilateral_item5_symmetry_add", evidence()
        )

    # Severe bilateral rows: restore item 16 only when the right side is also
    # high-load, avoiding lower-confidence asymmetric rows.
    if total >= 24 and p["L16"] == 1 and p["R16"] == 0 and side_total("R") >= 14:
        _set_second_pass_value(
            p, decisions, "R16", 1, "severe_bilateral_item16_symmetry_add", evidence()
        )

    # Very-mild asymmetric item-1 cleanup: isolated L1 in a near-WNL row.
    if total <= 4 and p["L1"] == 1 and p["R1"] == 0 and p["L2"] == 0 and p["L8"] == 0:
        _set_second_pass_value(p, decisions, "L1", 0, "very_mild_item1_asymmetry_cleanup", evidence())

    # Very-mild isolated item-15 cleanup: drop pelvic-rotation positives in
    # near-WNL rows, except when the bilateral item-8 + item-15 WNL-like
    # pattern is preserved.
    if total <= 4 and not (p["L8"] == 1 and p["R8"] == 1):
        ev = evidence()
        for col in ("L15", "R15"):
            _set_second_pass_value(p, decisions, col, 0, "very_mild_item15_isolated_cleanup", ev)

    # Very-mild isolated R11 cleanup: drop knee-swing positive in near-WNL row.
    if total <= 4 and p["R11"] == 1:
        _set_second_pass_value(p, decisions, "R11", 0, "very_mild_isolated_r11_cleanup", evidence())

    # Type-3 under-call rule (left side):
    # When a row is moderate severity (total in [17,21]) and the L-side has at
    # least 7 of 8 type-3 high-rate marker items active but i8 and i9 are both
    # absent, this is a "type-3-shaped row missing the most discriminative type-3
    # markers". Cross-track type3 rates: i8=0.89, i9=0.44 (n=9). Action: flip
    # L8/L9 to 1 (under-call), and flip L3 to 0 if active (cross-track type3
    # i3=0.11, so i3=1 is atypical of type3).
    # See Rodda & Graham 2004 (doi:10.1302/0301-620X.86B2.13878) for the
    # apparent-equinus kinematic pattern.
    if (
        17 <= total <= 21
        and len(TYPE3_SIDE_MARKERS & _active_items(p, "L")) >= 7
        and p["L8"] == 0 and p["L9"] == 0
    ):
        ev = evidence()
        if p["L3"] == 1:
            _set_second_pass_value(p, decisions, "L3", 0, "type3_undercall_left", ev)
        for col in ("L8", "L9"):
            _set_second_pass_value(p, decisions, col, 1, "type3_undercall_left", ev)

    # Type-3 contralateral consequence (right side):
    # When the contralateral side is also type-3-shaped (>=5 of 6 contra markers
    # active) but i10/i14 are both absent — i10 has cross-track type3 rate
    # 1.00 (universal), i14 rate 0.44 — this is the mirror under-call. Action:
    # add R1, R10, R14 to align with R&G type3 pattern (i1=0.78, i14=0.44).
    if (
        17 <= total <= 21
        and len(TYPE3_CONTRA_MARKERS & _active_items(p, "R")) >= 5
        and p["R10"] == 0 and p["R14"] == 0
    ):
        ev = evidence()
        for col in ("R1", "R10", "R14"):
            _set_second_pass_value(p, decisions, col, 1, "type3_undercall_right", ev)

    # Bilateral type-3 companion completion:
    # Bilateral type-3 row (high marker count on both sides, moderate-severe
    # total) with first-pass missing several type-3 companion items. Action:
    # add the under-called type-3 companions {13, 14, 16} on L and {5, 11, 14, 16}
    # on R. Cross-track type3 rates (n=9): i13=0.22, i14=0.44, i16=0.78,
    # i5=0.67, i11=0.44. Bilateral CP symmetry prior:
    # Wren et al. 2005 PMID:15614065. R&G type-3 description:
    # Rodda & Graham 2004 doi:10.1302/0301-620X.86B2.13878.
    if (
        20 <= total <= 22
        and len(TYPE3_SIDE_MARKERS & _active_items(p, "L")) >= 7
        and len(TYPE3_SIDE_MARKERS & _active_items(p, "R")) >= 6
        and p["L11"] == 1 and p["R11"] == 0
        and p["L13"] == 0 and p["R13"] == 0
        and p["L14"] == 0 and p["R14"] == 0
    ):
        ev = evidence()
        for col in ("L13", "L14", "L16", "R5", "R11", "R14", "R16"):
            _set_second_pass_value(p, decisions, col, 1,
                                   "bilateral_type3_companion_completion", ev)

    # Compact bilateral type-3 corrections:
    # Compact type-3 row (Total around 13) with discordant R7=1, L7=0.
    # R&G 2004 defines type-3 (apparent equinus) as normal ankle DF in swing
    # (i7=0). Cross-track type-3 rate i7=0.00 (n=9 sides). Action: correct R7
    # to 0 and complete the type-3 companions on both sides (i11, i16 plus
    # L1 contralateral pairing).
    if (
        12 <= total <= 14
        and len(TYPE3_SIDE_MARKERS & _active_items(p, "L")) >= 5
        and len(TYPE3_SIDE_MARKERS & _active_items(p, "R")) >= 4
        and p["L7"] == 0 and p["R7"] == 1
        and p["L11"] == 0 and p["R11"] == 0
        and p["L16"] == 0 and p["R16"] == 0
    ):
        ev = evidence()
        _set_second_pass_value(p, decisions, "R7", 0, "compact_type3_corrections", ev)
        for col in ("L11", "L16", "R1", "R11", "R16"):
            _set_second_pass_value(p, decisions, col, 1, "compact_type3_corrections", ev)

    # Low-total balanced row: align R1 toward L1=0 when bilateral profile is
    # mild and balanced. 94-patient classifier-derived WNL rate i1=0.08
    # (n=72 sides). Bilateral CP symmetry prior: Wren et al. 2005 PMID:15614065.
    if (
        total <= 9
        and abs(side_total("L") - side_total("R")) <= 2
        and p["L8"] == 1 and p["L10"] == 1
        and p["L1"] == 0 and p["R1"] == 1
    ):
        _set_second_pass_value(p, decisions, "R1", 0,
                               "low_total_balanced_i1_align_down", evidence())

    # Right-side WNL canonical Hamming-1 completion:
    # Near-WNL row (total<=5) where the R-side bit-pattern matches the
    # cross-track WNL canonical pattern {2, 5, 8, 14, 15} with R5 missing
    # (Hamming distance 1). Cross-track WNL n=2 sides: items 2,5,8,14,15
    # all rate 1.00. EVGS structure: Read et al. 2003 PMID:12724590.
    if total <= 5 and _active_items(p, "R") == {2, 8, 14, 15}:
        _set_second_pass_value(p, decisions, "R5", 1,
                               "wnl_canonical_r5_completion", evidence())

    return p, decisions
