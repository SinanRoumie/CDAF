"""End-speech minimum-spend foreclosure (policy/min_spend.py).

Verifies the Phase-1.5 intervention: a policy-layer mask that removes `end_speech` from
the sampled action set until the speech has SPENT a minimum fraction of its budget
(ceil rounding), with a deadlock guard and env-legality purity.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import CDAFEnvironment, is_legal
from env.actions import EndSpeech
from policy import min_spend
from policy.min_spend import (
    MinSpendLegalActionMask, apply_endspeech_min_spend, endspeech_min_moves,
    min_spend_binds,
)
from policy.masking import LegalActionMask, ACTION_TYPES

END = ACTION_TYPES.index("end_speech")


class _FakeState:
    """Minimal state carrying just the budget accounting the foreclosure reads."""
    def __init__(self, moves_used, slot_budget):
        self.moves_used = moves_used
        self.slot_budget = slot_budget


def test_default_is_off():
    """Unset env var -> frac 0.0 -> mask does not bind, apply is identity."""
    assert min_spend.ENDSPEECH_MIN_FRAC == 0.0
    assert not min_spend_binds()
    m = np.ones(len(ACTION_TYPES), dtype=bool)
    assert np.array_equal(apply_endspeech_min_spend(_FakeState(0, 8), m), m)


@pytest.mark.parametrize("frac,budget,expected", [
    (0.25, 8, 2), (0.25, 10, 3), (0.25, 13, 4),        # ceil: 2.0->2, 2.5->3, 3.25->4
    (0.5, 8, 4), (0.5, 13, 7),                          # 4.0->4, 6.5->7
    (0.75, 10, 8), (1.0, 10, 10), (1.0, 13, 13),        # 7.5->8, full-budget at frac 1.0
])
def test_threshold_is_ceil_of_budget(monkeypatch, frac, budget, expected):
    monkeypatch.setattr(min_spend, "ENDSPEECH_MIN_FRAC", frac)
    assert endspeech_min_moves(_FakeState(0, budget)) == expected


def test_forecloses_below_floor_keeps_others(monkeypatch):
    monkeypatch.setattr(min_spend, "ENDSPEECH_MIN_FRAC", 0.5)
    m = np.ones(len(ACTION_TYPES), dtype=bool)          # every type legal
    out = apply_endspeech_min_spend(_FakeState(0, 8), m)   # need 4, spent 0 -> foreclose
    assert not out[END]
    assert out[:END].all()                               # other types untouched


def test_unlocks_at_floor(monkeypatch):
    monkeypatch.setattr(min_spend, "ENDSPEECH_MIN_FRAC", 0.5)
    m = np.ones(len(ACTION_TYPES), dtype=bool)
    assert apply_endspeech_min_spend(_FakeState(4, 8), m)[END]   # spent==need -> unlocked
    assert apply_endspeech_min_spend(_FakeState(7, 8), m)[END]   # above need -> unlocked


def test_deadlock_guard_keeps_endspeech_when_only_move(monkeypatch):
    monkeypatch.setattr(min_spend, "ENDSPEECH_MIN_FRAC", 1.0)
    only_end = np.zeros(len(ACTION_TYPES), dtype=bool); only_end[END] = True
    out = apply_endspeech_min_spend(_FakeState(0, 8), only_end)  # below floor but sole move
    assert out[END]                                      # kept -> no empty action set


def test_never_widens(monkeypatch):
    """If end_speech is already illegal, the overlay cannot turn it on."""
    monkeypatch.setattr(min_spend, "ENDSPEECH_MIN_FRAC", 0.5)
    m = np.array([True, True, False, False, False, False])   # end_speech already False
    assert not apply_endspeech_min_spend(_FakeState(0, 8), m)[END]


def test_integration_move0_forecloses(monkeypatch):
    """On a fresh 1AC (moves_used=0, budget=8), frac=0.5 forecloses end_speech while
    introduce stays available; env is_legal(EndSpeech) is UNCHANGED (purity)."""
    monkeypatch.setattr(min_spend, "ENDSPEECH_MIN_FRAC", 0.5)
    env = CDAFEnvironment(); env.reset()
    st = env.state
    assert st.moves_used == 0 and endspeech_min_moves(st) == 4
    base = LegalActionMask(st).type_mask()
    masked = MinSpendLegalActionMask(st).type_mask()
    assert base[END] and not masked[END]                 # foreclosed only in the min-spend mask
    assert masked[:END].any()                            # a real move remains
    assert is_legal(st, EndSpeech())                     # env legality untouched


def test_integration_unlocks_after_spend(monkeypatch):
    """Stepping real (non-end) moves until moves_used>=need re-legalises end_speech."""
    import random
    from fuzz_env_shell import sample_legal
    monkeypatch.setattr(min_spend, "ENDSPEECH_MIN_FRAC", 0.5)
    env = CDAFEnvironment(); env.reset()
    need = endspeech_min_moves(env.state)
    rng = random.Random(0)
    guard = 0
    while env.state.moves_used < need and not env.state.terminated and guard < 200:
        guard += 1
        a, _ = sample_legal(env.state, rng)
        if isinstance(a, EndSpeech):
            continue                                     # don't end the speech; resample
        env.step(a)
    assert env.state.moves_used >= need
    assert MinSpendLegalActionMask(env.state).type_mask()[END]   # unlocked
