"""End-speech minimum-spend foreclosure (Phase 1.5 intervention).

A POLICY-LAYER training scaffold, exactly like the opening curriculum (curriculum.py)
and NOT an environment legality rule: it AND-composes over the env legal-action mask,
can only turn `end_speech` OFF for sampling (never widen legality), and the judge never
sees it. The env's `is_legal(state, EndSpeech())` is UNCHANGED -- end_speech remains
structurally legal, so no env-reachable terminal graph is removed; the foreclosure only
removes `end_speech` from the POLICY's sampled action set in states where the speech has
not yet spent a minimum fraction of its budget.

Motivation: `end_speech` costs zero budget and is always structurally legal, so a
near-random policy defaults to it (the same shape as entropy-runaway toward zero-cost
max-entropy actions), ending speeches with the budget unused. This mask raises the floor
on per-speech spend so the extra sweep budget is actually exercised.

Threshold semantics: unlock `end_speech` once `moves_used >= ceil(frac * slot_budget)`
-- a fraction of BUDGET SPENT (moves_used), NOT a node count (extends/connects consume
budget without creating nodes; a node-count minimum would wrongly force introduces in
rebuttals). Rounding is CEIL ("must have spent at least this fraction"): at frac=1.0 the
threshold is the full budget, i.e. end_speech is removed until the budget is exhausted.

Deadlock safety: end_speech is foreclosed only when at least one other action type is
legal in the state -- if end_speech is the ONLY legal move it stays available, so the
env is never left with an empty action set.

Configured once at import (mirrors CDAF_SPEECH_BUDGET): `CDAF_ENDSPEECH_MIN_FRAC`, a
float in [0, 1]; 0.0 (default) disables the mask entirely (identity -> base behaviour).
"""
from __future__ import annotations

import math
import os

import numpy as np

from .masking import LegalActionMask, ACTION_TYPES
from .curriculum import CurriculumLegalActionMask

_END = ACTION_TYPES.index("end_speech")


def _load_frac() -> float:
    raw = os.environ.get("CDAF_ENDSPEECH_MIN_FRAC")
    if raw is None:
        return 0.0
    f = float(raw)                                       # raises on non-float
    if not (0.0 <= f <= 1.0):
        raise ValueError(f"CDAF_ENDSPEECH_MIN_FRAC must be in [0, 1], got {f}")
    return f


ENDSPEECH_MIN_FRAC = _load_frac()


def min_spend_binds() -> bool:
    """True iff the foreclosure is active (frac > 0). Read at call time so a test may
    monkeypatch ENDSPEECH_MIN_FRAC."""
    return ENDSPEECH_MIN_FRAC > 0.0


def endspeech_min_moves(state) -> int:
    """ceil(frac * slot_budget): budget that must be SPENT before end_speech unlocks."""
    return math.ceil(ENDSPEECH_MIN_FRAC * state.slot_budget)


def forecloses(state) -> bool:
    """True iff, ignoring the deadlock guard, this state is below its min-spend floor."""
    return min_spend_binds() and state.moves_used < endspeech_min_moves(state)


def apply_endspeech_min_spend(state, m: np.ndarray) -> np.ndarray:
    """Overlay the foreclosure on an already-composed (6,) type mask `m`. Turns end_speech
    OFF when the speech is below its min-spend floor, UNLESS end_speech is the only legal
    type (deadlock safety). Never widens `m`."""
    if not (forecloses(state) and m[_END]):
        return m
    others = m.copy(); others[_END] = False
    if not others.any():                                 # end_speech is the only move -> keep it
        return m
    m = m.copy(); m[_END] = False
    return m


class MinSpendLegalActionMask(LegalActionMask):
    """`LegalActionMask` with the end-speech min-spend foreclosure over its type mask."""

    def type_mask(self) -> np.ndarray:
        return apply_endspeech_min_spend(self.state, super().type_mask())


class MinSpendCurriculumLegalActionMask(CurriculumLegalActionMask):
    """Both scaffolds composed: env legality AND the opening ladder AND min-spend. Applied
    after the curriculum overlay so the deadlock guard reads the already-narrowed mask."""

    def type_mask(self) -> np.ndarray:
        return apply_endspeech_min_spend(self.state, super().type_mask())
