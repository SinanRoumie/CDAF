"""Response-window enforcement (judge_spec §4): an attack arriving AFTER its target's
response window closes is INERT (not scored), reported via a WINDOW_CLOSED trace record.

Default window = the single immediately-following opposing speech; the 1AC exception
extends a 1AC node's window through 2NC/1NR. This is response VALIDITY/timing on an
existing argument -- distinct from the no-new-offense-in-rebuttals rule.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

from model import serialize as mser
from model.speeches import SPEECH_ORDER
from judge import judge
from judge.passes import window_close, response_window, in_response_window, node_accrual

AFF_SP = ["1AC", "2AC", "1AR", "2AR"]
NEG_SP = ["1NC", "2NC/1NR", "2NR"]


def _live(sp_list, frm):
    i = SPEECH_ORDER.index(frm)
    return {s: "conceded" for s in sp_list if SPEECH_ORDER.index(s) >= i}


def _node(nid, ntype, side, speech, liv):
    return {"data": {"id": nid, "label": nid, "ntype": ntype, "side": side,
                     "speech": speech, "liveness": liv}}


def _edge(nid, s, t, et):
    return {"data": {"id": nid, "source": s, "target": t, "etype": et}}


def _round_with_attack(target_speech, target_side, attacker_speech):
    """A target Link + one opposing defensive attack introduced at `attacker_speech`.
    Returns (trace, sigma_of_target)."""
    tside_sp = AFF_SP if target_side == "AFF" else NEG_SP
    aside_sp = NEG_SP if target_side == "AFF" else AFF_SP
    aside = "NEG" if target_side == "AFF" else "AFF"
    els = [
        _node("tgt", "Link", target_side, target_speech, _live(tside_sp, target_speech)),
        _node("att", "Link", aside, attacker_speech, _live(aside_sp, attacker_speech)),
        _edge("e1", "att", "tgt", "DefensiveAttackEdge"),
    ]
    rnd = mser.from_dict({"version": 2, "elements": els})
    _ballot, trace = judge(rnd)

    class _NV:
        _K = {"Link": "link"}
        def __init__(s, d):
            s.id = d["id"]; s.kind = s._K[d["ntype"]]; s.side = d["side"]
            s.speech = d["speech"]; s.liveness = d["liveness"]; s.ntype = d["ntype"]

    class _EV:
        def __init__(s, d):
            s.id = d["id"]; s.source = d["source"]; s.target = d["target"]
            s.kind = "defensive_attack"

    nvs = [_NV(e["data"]) for e in els if "source" not in e["data"]]
    evs = [_EV(e["data"]) for e in els if "source" in e["data"]]
    acc = node_accrual(nvs, evs)
    return trace, acc.sigma.get("tgt"), acc.trace


def _window_closed_edges(trace):
    return {r.edge_id for r in trace if getattr(r, "kind", None) == "WINDOW_CLOSED"}


# --- helper correctness ------------------------------------------------------

def test_window_close_1ac_exception():
    assert window_close("1AC", "AFF") == "2NC/1NR"          # extended through both NEG constructives
    assert response_window("1AC", "AFF") == "1NC"           # open is still 1NC
    assert window_close("1NC", "NEG") == "2AC"              # default = single next opposing speech
    assert window_close("2AC", "AFF") == "2NC/1NR"


def test_in_response_window_ranges():
    # 1AC node: 1NC and 2NC/1NR are in-window; 1AR is not
    assert in_response_window("1NC", "1AC", "AFF")
    assert in_response_window("2NC/1NR", "1AC", "AFF")
    assert not in_response_window("1AR", "1AC", "AFF")
    # 1NC node (default single-speech window = 2AC): only 2AC is in-window
    assert in_response_window("2AC", "1NC", "NEG")
    assert not in_response_window("1AR", "1NC", "NEG")


# --- in-window attacks are scored -------------------------------------------

def test_in_window_attack_is_scored():
    _tr, sigma, acc_tr = _round_with_attack("1AC", "AFF", "1NC")   # 1NC is in 1AC's window
    assert sigma is not None and sigma < 1.0                        # reduced -> evaluated
    assert not _window_closed_edges(acc_tr)


def test_1ac_exception_2nc_answer_is_in_window():
    """A 1AC node answered at 2NC/1NR is WITHIN window (exception) -> scored, not inert."""
    _tr, sigma, acc_tr = _round_with_attack("1AC", "AFF", "2NC/1NR")
    assert sigma is not None and sigma < 1.0
    assert not _window_closed_edges(acc_tr)


# --- late attacks are inert + reported --------------------------------------

def test_late_attack_on_1ac_is_inert_and_reported():
    """2NR is past the 1AC window close (2NC/1NR) -> inert, WINDOW_CLOSED emitted."""
    _tr, sigma, acc_tr = _round_with_attack("1AC", "AFF", "2NR")
    assert sigma == 1.0                                            # untouched -> not evaluated
    wc = [r for r in acc_tr if getattr(r, "kind", None) == "WINDOW_CLOSED"]
    assert len(wc) == 1
    r = wc[0]
    assert (r.attacker_speech, r.target_speech, r.window_close) == ("2NR", "1AC", "2NC/1NR")
    assert r.attacker_id == "att" and r.target_id == "tgt" and r.edge_id == "e1"


def test_late_attack_on_1nc_is_inert():
    """A 1NC node's window is 2AC; answering first at 1AR (past it) is inert."""
    _tr, sigma, acc_tr = _round_with_attack("1NC", "NEG", "1AR")
    assert sigma == 1.0
    wc = [r for r in acc_tr if getattr(r, "kind", None) == "WINDOW_CLOSED"]
    assert len(wc) == 1 and wc[0].window_close == "2AC"
