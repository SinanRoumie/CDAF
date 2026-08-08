"""Tests for the warm-start fixture->action-sequence converter (`warmstart.convert`).

Verifies the spec's guarantees (docs/warm_start_data_spec.md) mechanically:
  * every converted action sequence replays LEGALLY end-to-end through the real env;
  * replay reproduces a JUDGE-IDENTICAL verdict to the oracle fixture (ruling A) --
    now all 46/46;
  * edge orientation is natural-forward-with-flips (E), spot-checked on a known flipped
    cross-speech support edge;
  * `connect` placement is earliest-legal-speech, that speech's side (C), including the
    budget-deferral case;
  * verb labeling follows the cross-side ownership rule (D) for every carriage.

F was previously the one known gap (its authored `link: 2NR: contested`, with no attack
edge, was not structurally derivable, so the env-built round scored differently). The
'no new offense in rebuttals' bug fix (judge_spec §6, per-path) resolved it: F's impact
is introduced in the 2AR (a rebuttal) and is now correctly disqualified regardless of
the contested status, so BOTH the oracle and the env-built round score NEG presumption
-- F converts cleanly. The whole corpus (46/46) now round-trips verdict-identically.
"""

from __future__ import annotations

import glob
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

from model import serialize, SPEECH_SIDE
from env import CDAFEnvironment, is_legal, observe
from env.actions import Extend, Concede
from warmstart import convert

ORACLE_DIR = os.path.join(os.path.dirname(__file__), "oracle")

# No known gaps remain: the whole corpus round-trips verdict-identically (see module
# docstring; the former F gap was resolved by the §6 rebuttal-offense bug fix).
KNOWN_GAP = frozenset()


def _names():
    return sorted(os.path.basename(f)[:-5] for f in glob.glob(os.path.join(ORACLE_DIR, "*.json")))


# Convert the whole corpus once (module-level cache).
_RESULTS = {n: convert(serialize.load(os.path.join(ORACLE_DIR, n + ".json")), n) for n in _names()}
_ALL = sorted(_RESULTS)


def _replay(actions):
    """Replay an action list through a fresh env, asserting each step is legal at the
    point it is taken. Returns the terminated env."""
    env = CDAFEnvironment(); env.reset()
    for i, a in enumerate(actions):
        assert is_legal(env.state, a), f"action {i} ({type(a).__name__}) illegal on replay"
        env.step(a)
    return env


# --- corpus sanity -----------------------------------------------------------

def test_corpus_is_46_v2_fixtures():
    assert len(_ALL) == 46                       # 45 §11 probes + full_round_aff_outweighs


# --- legality of the emitted sequence ----------------------------------------

@pytest.mark.parametrize("name", _ALL)
def test_sequence_replays_legally_and_terminates(name):
    r = _RESULTS[name]
    assert r.error is None, f"{name}: conversion errored: {r.error}"
    env = _replay(r.actions)
    assert env.state.terminated, f"{name}: replay did not terminate"


@pytest.mark.parametrize("name", _ALL)
def test_pairs_align_with_actions(name):
    r = _RESULTS[name]
    assert len(r.pairs) == len(r.actions)
    for (obs, act), a in zip(r.pairs, r.actions):
        assert act is a
        assert isinstance(obs, dict) and "graph" in obs and "sequence" in obs


def test_fresh_replay_reproduces_the_recorded_pairs():
    """The output is an action list; the training loop replays it in a fresh env to get
    (observation, action) pairs. A fresh replay must reproduce the recorded observations
    exactly (determinism -- same ids, same graph, same accrual)."""
    r = _RESULTS["r9"]
    env = CDAFEnvironment(); env.reset()
    for (recorded_obs, act) in r.pairs:
        assert observe(env.state) == recorded_obs
        env.step(act)


# --- judge-equivalence (ruling A) + the documented reachability gap -----------

def test_verdict_equivalence_whole_corpus():
    passed = {n for n, r in _RESULTS.items() if r.ok}
    failed = {n for n, r in _RESULTS.items() if not r.ok}
    assert failed == set(KNOWN_GAP), f"unexpected verdict outcomes: failed={sorted(failed)}"
    assert len(passed) == len(_ALL) == 46          # no known gaps remain


@pytest.mark.parametrize("name", sorted(set(_names()) - KNOWN_GAP))
def test_passing_fixtures_reproduce_the_oracle_verdict(name):
    r = _RESULTS[name]
    assert r.ok, f"{name}: replay {r.replay_verdict} != oracle {r.oracle_verdict}"
    assert r.replay_verdict == r.oracle_verdict


# --- ruling D: verb labeling by target ownership -----------------------------

@pytest.mark.parametrize("name", _ALL)
def test_verb_labeling_follows_ownership(name):
    """Every carriage is `concede` iff the acting side != the target's owner, else
    `extend` (D). Checked by replay: read current side + target owner at each step."""
    r = _RESULTS[name]
    env = CDAFEnvironment(); env.reset()
    for a in r.actions:
        if isinstance(a, (Extend, Concede)):
            side = env.state.current_side
            owner = env.state.nodes[a.node_id].owner
            if isinstance(a, Concede):
                assert owner != side, f"{name}: Concede on own-side node {a.node_id}"
            else:
                assert owner == side, f"{name}: Extend on opponent node {a.node_id}"
        env.step(a)


def test_cross_side_carriage_produces_concede():
    """r4 is a turn fixture where NEG carries the AFF link at NEG speeches -> concedes."""
    assert _RESULTS["r4"].diag["concedes"] > 0


# --- ruling E: natural-forward-with-flips orientation -------------------------

def test_support_edge_flipped_from_authored_orientation():
    """r9 authors `im -> bd` (impact 1AC -> BD 2AR, source-earlier). Reconstruction
    introduces bd at 2AR attaching onto the existing im, so the built edge is bd -> im
    -- flipped from the authored direction (E). No connect is used for it."""
    r = _RESULTS["r9"]
    bd_edge = next(e for e in r.diag["intro_edges"] if e["child"] == "bd")
    assert bd_edge["parent"] == "im" and bd_edge["kind"] == "support"
    assert bd_edge["authored"] == ("im", "bd") and bd_edge["flipped"] is True
    assert r.diag["flipped_support"] >= 1
    assert ("bd", "im") not in [(c["pair"]) for c in r.diag["connects"]]   # it's an intro edge


# --- ruling C: connect placement = earliest-legal-speech, that speech's side --

def test_connect_placed_at_endpoints_speech_when_affordable():
    """r32's convergence diamond is authored entirely in 1AC and 1AC has room, so the
    closing `connect` lands at 1AC, performed by AFF."""
    conns = _RESULTS["r32"].diag["connects"]
    assert len(conns) == 1
    assert conns[0]["slot"] == "1AC" and conns[0]["side"] == "AFF"


def test_connect_defers_to_next_legal_speech_when_budget_full():
    """r33's 1AC is full (8 introduces), so its 1AC-authored diamond connect defers to
    the next speech where both endpoints exist and the move is affordable -- 1NC,
    performed by NEG (earliest-legal-speech, that speech's side; C)."""
    conns = _RESULTS["r33"].diag["connects"]
    assert len(conns) == 1
    assert conns[0]["slot"] == "1NC" and conns[0]["side"] == "NEG"


@pytest.mark.parametrize("name", ["E", "r37", "r38", "r39"])
def test_forest_fixtures_use_no_connect(name):
    """A fixture whose structural edges form a spanning forest needs zero `connect`s --
    every edge is a node's introduction edge."""
    assert _RESULTS[name].diag["connects"] == []
