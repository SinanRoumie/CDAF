"""Episode-validity fence tests (gap-audit STEP 4).

The judge is a pure scoring function; the RL ENVIRONMENT owns which graphs a
training agent may generate episodes over. `env.validator` fences three corners
the judge is not audited on. These tests pin each fence: a graph that reaches the
corner is REJECTED, a clean graph is ADMITTED, and every oracle fixture the judge
suite relies on validates clean (so the fences never refuse a round we score).
"""

import glob
import os

from model import (
    Round, Advocacy, Uniqueness, Link, Impact, BallotDirective,
    Support, serialize, SPEECH_ORDER,
)
from judge import trace as T
from env import validate_round, is_valid
from env.validator import trace_has_convergence_marker, CONVERGENCE_MARKER

ORACLE_DIR = os.path.join(os.path.dirname(__file__), "oracle")

# Fixtures the oracle SUITE actually loads -- these must all validate clean (the
# fence must never refuse a round the judge is scored on). A genuine multi-terminal
# oracle round (r37) is admitted since Fence A retired (v11) -- asserted separately below.
SUITE_FIXTURES = [
    "r29.json", "r31.json", "r32.json", "r33.json", "r34.json", "r35.json",
    "AFFLinkturnsNeg.json", "AFFLinkturnsNeg_impactanchor.json",
    "AFFturnOutweighed.json", "fw_wash_no_extend.json", "fw_weigh_lockout.json",
]


def _n(cls, nid, side, speech, label="x"):
    from model import SPEECH_SIDE
    live = {s: "conceded" for s in SPEECH_ORDER
            if SPEECH_SIDE.get(s) == side and SPEECH_ORDER.index(s) >= SPEECH_ORDER.index(speech)} \
        if speech in SPEECH_ORDER else {}
    return cls(id=nid, label=label, side=side, speech=speech, liveness=live)


def _sup(eid, a, b):
    return Support(id=eid, source=a.id, target=b.id)


# --- clean rounds admit; suite fixtures admit -------------------------------

def test_clean_round_admitted():
    adv = _n(Advocacy, "n1", "AFF", "1AC"); uni = _n(Uniqueness, "n2", "AFF", "1AC")
    lk = _n(Link, "n3", "AFF", "1AC"); im = _n(Impact, "n4", "AFF", "1AC")
    bd = _n(BallotDirective, "n5", "AFF", "2AR")
    rnd = Round(elements=[adv, uni, lk, im, bd,
                          _sup("e1", adv, uni), _sup("e2", uni, lk),
                          _sup("e3", lk, im), _sup("e4", im, bd)], version=2)
    v = validate_round(rnd)
    assert v.ok and v.reasons == []
    assert is_valid(rnd)


def test_all_suite_fixtures_validate_clean():
    for name in SUITE_FIXTURES:
        rnd = serialize.load(os.path.join(ORACLE_DIR, name))
        v = validate_round(rnd)
        assert v.ok, f"{name} should validate clean, got {v.reasons}"


# --- Fence A RETIRED (judge v11): multi-terminal components are now LEGAL ----

def test_multiterminal_component_admitted():
    """As of judge v11 a same-side Support component with two terminal impacts (a link
    diverging to two independent impacts) is FIRST-CLASS -- scored per branch, summed
    at the ballot. Structural admission (Fence G only) no longer refuses it."""
    adv = _n(Advocacy, "n1", "AFF", "1AC"); lk = _n(Link, "n2", "AFF", "1AC")
    im1 = _n(Impact, "n3", "AFF", "1AC"); im2 = _n(Impact, "n4", "AFF", "1AC")
    bd = _n(BallotDirective, "n5", "AFF", "2AR")
    rnd = Round(elements=[adv, lk, im1, im2, bd,
                          _sup("e1", adv, lk), _sup("e2", lk, im1), _sup("e3", lk, im2),
                          _sup("e4", im1, bd), _sup("e5", im2, bd)], version=2)
    v = validate_round(rnd)
    assert v.ok and v.reasons == []


def test_real_multiterminal_round_admitted():
    """r37 is a genuine multi-terminal oracle round (a same-side Support component with
    two terminal impacts); with Fence A retired it is admitted (structural admission is
    Fence G only). Its judge verdict is not pinned here."""
    rnd = serialize.load(os.path.join(ORACLE_DIR, "r37.json"))
    assert validate_round(rnd).ok


# --- FENCE G: off-vocab speech ----------------------------------------------

def test_fence_G_offvocab_speech_rejected():
    """A node whose speech is not in SPEECH_ORDER is not inert in the judge (it
    silently short-circuits extension and dodges drop detection); the fence rejects
    it as malformed at ingest."""
    adv = _n(Advocacy, "n1", "AFF", "1AC"); uni = _n(Uniqueness, "n2", "AFF", "1AC")
    lk = _n(Link, "n3", "AFF", "1AC")
    im = Impact(id="n4", label="x", side="AFF", speech="9NC", liveness={})   # off-vocab
    bd = _n(BallotDirective, "n5", "AFF", "2AR")
    rnd = Round(elements=[adv, uni, lk, im, bd,
                          _sup("e1", adv, uni), _sup("e2", uni, lk),
                          _sup("e3", lk, im), _sup("e4", im, bd)], version=2)
    v = validate_round(rnd)
    assert not v.ok
    assert any(r.startswith("G/off-vocab-speech") and "n4" in r for r in v.reasons)


# --- FENCE B: unequal-magnitude convergence marker ---------------------------

def test_fence_B_marker_scan_detects_and_rejects():
    """The B fence reads the judge's write-only CONVERGENCE_OUT_OF_SCOPE marker off
    the trace. The unequal-magnitude branch is unreachable in binary V1 (every live
    path magnitude is 1.0 -> the equal-mag wash fires instead), so the mechanism is
    pinned directly: a trace carrying the marker is detected."""
    trace_without = [T.Chain(chain_id="c", sign=1, mag=1.0, delta=1.0)]
    trace_with = trace_without + [T.ConvergenceOutOfScope(impact_id="n3", pos_mag=1.0, neg_mag=0.5)]
    assert not trace_has_convergence_marker(trace_without)
    assert trace_has_convergence_marker(trace_with)
    assert T.ConvergenceOutOfScope.kind == CONVERGENCE_MARKER


def test_fence_B_clean_round_has_no_marker():
    """A normal round never emits the marker (the branch is not reached), so the B
    fence adds no reason for clean graphs."""
    rnd = serialize.load(os.path.join(ORACLE_DIR, "r34.json"))   # equal-mag wash, not the unequal branch
    v = validate_round(rnd)
    assert not any(r.startswith("B/") for r in v.reasons)
