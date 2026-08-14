"""M0 renderer tests (docs/render_spec.md §8). No LLM, deterministic.

Read-only over the oracle fixtures; a few checks use run exports and skip if the
export tree is absent (it is gitignored). Nothing here touches judge/env/model.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.round import Round
from model.speeches import speech_index
from render.adapter import analyze, load_round, INCOMPLETE
from render.linearize import render, summary_stats

ORACLE = os.path.join(os.path.dirname(__file__), "oracle")
EXPORT_DIRS = [
    "runs/round_export_matrix_A5_20260812",
    "runs/round_export_matrix_A_20260812",
    "runs_verify/B4post/exports",
]


def _oracle(name):
    return load_round(os.path.join(ORACLE, name))


def _export(stem):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for d in EXPORT_DIRS:
        p = os.path.join(root, d, stem + ".json")
        if os.path.exists(p):
            return load_round(p)
    pytest.skip(f"export {stem} not present")


# --- determinism & topology-derived order (RS14) -----------------------------

def test_render_is_deterministic():
    rnd = _oracle("full_round_aff_outweighs.json")
    a = render(analyze(rnd), rnd, "x")
    b = render(analyze(rnd), rnd, "x")
    assert a == b


def test_order_independent_RS14():
    """Identical graphs constructed in different orders render identically."""
    rnd = _oracle("full_round_aff_outweighs.json")
    base = render(analyze(rnd), rnd, "x")
    shuffled = Round(elements=list(reversed(rnd.elements)), version=rnd.version)
    assert render(analyze(shuffled), shuffled, "x") == base


# --- coverage: every node & extension renders (RS23 new node, RS26) ----------

def test_every_node_has_one_claim_line():
    rnd = _oracle("full_round_aff_outweighs.json")
    out = render(analyze(rnd), rnd, "x")
    assert out.count("[claim: ") == len(rnd.nodes)


def test_extension_count_matches_liveness_RS26():
    rnd = _oracle("full_round_aff_outweighs.json")
    out = render(analyze(rnd), rnd, "x")
    expected = sum(
        1 for n in rnd.nodes for sp in n.liveness
        if speech_index(sp) > speech_index(n.speech))
    assert out.count("[extend ") == expected


# --- RS13 assert vs RS13c incomplete -----------------------------------------

def test_E_renders_incomplete_not_asserted():
    """v1.4: E.json (no Advocacy anywhere) is INCOMPLETE, not a crash. The renderer
    does not halt on any input; structurally it is identical to a legal no-advocacy
    export, so no assert can separate them."""
    rnd = _oracle("E.json")
    an = analyze(rnd)
    assert any(c.register == INCOMPLETE for c in an.components)
    out = render(an, rnd, "E")               # must not raise
    assert "never connected to advocacy or framework" in out


def test_RS13c_incomplete_renders_with_marker():
    """A no-register component in a round that DOES contain an Advocacy renders
    with the incomplete marker and does not raise."""
    rnd = _export("A5_nearmiss_seed2_u25_ep004")
    an = analyze(rnd)
    assert any(c.register == INCOMPLETE for c in an.components)
    out = render(an, rnd, "x")               # must not raise
    assert "never connected to advocacy or framework" in out


# --- RS10c: divergence recorded, never raised --------------------------------

def test_RS10c_divergence_is_recorded_not_raised():
    rnd = _export("B4post_nearmiss_seed4_u25_ep003")
    an = analyze(rnd)
    assert len(an.rs10c_divergences) >= 1     # known BD-conductor divergence
    render(an, rnd, "x")                       # recorded, so render must not raise


# --- RS27: judge-invisible edges marked --------------------------------------

def test_RS27_marks_invisible_edges():
    rnd = _export("B4post_negwin_seed0_u25_ep015")
    an = analyze(rnd)
    out = render(an, rnd, "x")
    assert len(an.invisible_edges) >= 1
    assert out.count("judge-invisible") == len(an.invisible_edges)


# --- substantive register survives on the r36 cross-side-rooted shape --------

def test_r36_is_substantive():
    rnd = _oracle("r36.json")
    an = analyze(rnd)
    negs = [c for c in an.components if c.side == "NEG"]
    assert any(c.register == "substantive" for c in negs)


def test_summary_stats_shape():
    rnd = _oracle("r1.json")
    st = summary_stats(analyze(rnd), rnd)
    assert set(st) == {"components", "register", "rs13c_markers",
                       "rs27_markers", "rs10c_divergences", "words_per_speech"}
