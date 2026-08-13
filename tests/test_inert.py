"""Tests for the inert-action handling after the 2026-08-08 masking ruling.

Two mechanisms, split by whether the inertness is logically incoherent or
context-dependent:

  * STRUCTURAL INCOHERENCE -> ILLEGAL (masked). Three moves that can never be meaningful
    in any round state are rejected by `check_legality`: a same-side attack, an offense
    at a non-polarity node, and a redundant connect. They are never sampled.
  * CONTEXT-DEPENDENT INERTNESS -> LEGAL, priced via COST. A no-op re-extend (extend/
    concede on a node already carried this speech) stays legal but costs a FULL SLOT,
    bypassing the K-batch discount; `is_inert` still flags it for reward-side/diagnostic
    bookkeeping (reward coefficient dormant at 0.0).

The judge is unaffected: it must stay robust to incoherent graphs regardless of what the
env now permits (see tests/oracle for G2/G3/r22/r25 judge-direct tests).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

from env import CDAFEnvironment, RoundState, is_inert, is_legal, check_legality
from env.actions import Introduce, Extend, Concede, Connect, EndSpeech
from env.state import action_cost
from env.legal_actions import INERT_NOOP_REEXTEND
from judge.config import AFF, NEG


# --- structural incoherence is now ILLEGAL (masked) --------------------------

def test_same_side_attack_introduce_illegal():
    st = RoundState()
    tgt = st.add_node("a", AFF, "link")              # AFF, 1AC
    act = Introduce("b", "link", target=tgt, edge_type="defensive_attack")
    ok, reason = check_legality(st, act)             # new node is AFF -> same-side attack
    assert not ok and "same-side" in reason
    assert not is_legal(st, act)


def test_same_side_attack_connect_illegal():
    st = RoundState()
    a = st.add_node("a", AFF, "link")
    b = st.add_node("b", AFF, "impact")
    ok, reason = check_legality(st, Connect(a, b, "offensive_attack"))
    assert not ok and "same-side" in reason


def test_cross_side_attack_is_legal():
    st = RoundState()
    tgt = st.add_node("a", AFF, "link")              # AFF, 1AC
    st.advance_speech()                              # 1NC (NEG)
    assert is_legal(st, Introduce("b", "link", target=tgt, edge_type="defensive_attack"))


def test_offense_at_nonpolarity_target_illegal():
    st = RoundState()
    uni = st.add_node("u", AFF, "uniqueness")        # non-polarity target
    st.advance_speech()                              # 1NC (NEG)
    ok, reason = check_legality(st, Introduce("x", "link", target=uni, edge_type="offensive_attack"))
    assert not ok and "non-polarity" in reason


def test_offense_from_nonpolarity_attacker_role_illegal():
    st = RoundState()
    link = st.add_node("l", AFF, "link")             # offense-bearing target
    st.advance_speech()                              # 1NC (NEG)
    # attacker role is a uniqueness -> not offense-bearing -> illegal even at a link
    ok, reason = check_legality(st, Introduce("x", "uniqueness", target=link, edge_type="offensive_attack"))
    assert not ok and "non-polarity" in reason


def test_valid_cross_side_offense_between_polarity_nodes_is_legal():
    st = RoundState()
    link = st.add_node("l", AFF, "link")
    st.advance_speech()                              # 1NC (NEG)
    assert is_legal(st, Introduce("x", "impact", target=link, edge_type="offensive_attack"))


def test_defensive_attack_at_nonpolarity_node_is_legal():
    """§3.4 guards OFFENSIVE attacks only -- a defensive_attack at a framework is coherent
    (it lowers magnitude), so it stays legal."""
    st = RoundState()
    fw = st.add_node("f", AFF, "framework")
    st.advance_speech()                              # 1NC (NEG)
    assert is_legal(st, Introduce("x", "link", target=fw, edge_type="defensive_attack"))


def test_redundant_connect_illegal():
    st = RoundState()
    a = st.add_node("a", AFF, "link")
    b = st.add_node("b", AFF, "impact")
    st.add_edge(a, b, "support")
    ok, reason = check_legality(st, Connect(a, b, "support"))
    assert not ok and "redundant" in reason


def test_non_duplicate_connect_is_legal():
    st = RoundState()
    a = st.add_node("a", AFF, "link")
    b = st.add_node("b", AFF, "impact")
    c = st.add_node("c", AFF, "impact")
    st.add_edge(a, b, "support")
    # different target -> not a duplicate; support -> not an attack; no cycle
    assert is_legal(st, Connect(a, c, "support"))


# --- no-op re-extend is now ILLEGAL; is_inert stays a live predicate but is unreachable
#     via legal play (Yaz ruling). --------

def test_reextend_in_introduction_speech_is_inert_predicate_but_illegal():
    st = RoundState()
    nid = st.add_node("x", AFF, "impact")            # carried = {1AC}
    inert, kind = is_inert(st, Extend(nid))          # predicate still detects the no-op
    assert inert and kind == INERT_NOOP_REEXTEND
    assert not is_legal(st, Extend(nid))             # but MASKED: same-speech extend illegal


def test_first_extend_in_a_new_speech_is_not_inert_then_reextend_is():
    st = RoundState()
    nid = st.add_node("x", AFF, "impact")            # intro 1AC, carried={1AC}
    st.advance_speech()                              # 1NC
    st.advance_speech()                              # 2AC (AFF again)
    assert not is_inert(st, Extend(nid))[0]         # 2AC not yet carried -> real carry
    st.carry(nid)                                    # now carried at 2AC
    inert, kind = is_inert(st, Extend(nid))          # re-extend same speech -> inert
    assert inert and kind == INERT_NOOP_REEXTEND


def test_concede_is_classified_like_extend():
    st = RoundState()
    nid = st.add_node("x", AFF, "impact")
    inert, kind = is_inert(st, Concede(nid))
    assert inert and kind == INERT_NOOP_REEXTEND


def test_new_node_introduce_and_endspeech_never_inert():
    st = RoundState()
    assert is_inert(st, Introduce("x", "impact"))[0] is False    # fresh NEW node
    assert is_inert(st, EndSpeech())[0] is False


# --- cost model: no-op re-extend costs a full slot, distinct carriage keeps batch rate ---

def test_noop_reextend_costs_full_slot_vs_distinct_batch_rate():
    st = RoundState()
    a = st.add_node("a", AFF, "link")
    b = st.add_node("b", AFF, "impact")
    st.advance_speech()                              # 1NC
    st.advance_speech()                              # 2AC (AFF); a,b not yet carried here
    # first distinct carriage: batch-first of the speech -> 1
    assert action_cost(st, Extend("n1")) == 1
    st.apply(Extend("n1"))                           # extends_this_speech -> 1
    # second distinct carriage: within the same K-group -> batch rate 0
    assert action_cost(st, Extend("n2")) == 0
    st.apply(Extend("n2"))                           # extends_this_speech -> 2
    # re-extend a (already carried this speech) -> NO-OP -> now ILLEGAL (masked). The cost
    # branch still prices it a full slot, but it is a dead defensive path.
    assert st.already_carried_this_speech("n1")
    assert action_cost(st, Extend("n1")) == 1
    assert not is_legal(st, Extend("n1"))            # masked, not priced-and-legal


def test_distinct_carriages_keep_k_batch_discount():
    """K distinct carriages cost ceil(K/K)=1 slot total (the unchanged batch rate)."""
    st = RoundState()
    ids = [st.add_node(f"n{i}", AFF, "impact") for i in range(4)]
    st.advance_speech(); st.advance_speech()         # 2AC (AFF)
    costs = []
    for nid in ids:                                  # 4 distinct carriages, K=4
        costs.append(action_cost(st, Extend(nid)))
        st.apply(Extend(nid))
    assert costs == [1, 0, 0, 0]                     # one slot buys K distinct carriages


# --- terminal penalty wiring (hook retained; coef dormant at 0.0) ------------

def _drive_to_termination(env):
    info = {}
    while not env.state.terminated:
        _obs, _r, _done, info = env.step(EndSpeech())
    return info


def test_env_rejects_noop_reextend_so_penalty_hook_is_unreachable():
    """Post-ruling, a no-op re-extend (the sole inert class) is ILLEGAL, so the env rejects
    it -- the inert-penalty hook can no longer be triggered by any legal play. A clean round
    therefore carries zero inert penalty / counts even with the coef set."""
    env = CDAFEnvironment(inert_penalty_coef=0.01)
    env.reset()
    env.step(Introduce("x", "advocacy"))             # -> node n1, carried {1AC} (legal root)
    with pytest.raises(ValueError):
        env.step(Extend("n1"))                        # no-op re-extend now illegal
    info = _drive_to_termination(env)                 # remainder of the round, no inert moves
    bd = info["reward_breakdown"]
    assert bd[AFF]["inert_penalty"] == pytest.approx(0.0)
    assert bd[NEG]["inert_penalty"] == pytest.approx(0.0)
    assert not info["inert_counts"]                   # nothing inert could occur


def test_default_coef_clean_round_has_no_inert_penalty_or_counts():
    """Default coefficient 0.0 (dormant); a legal round records no inert penalty or counts
    (the only inert class, no-op re-extend, is unreachable via legal play)."""
    env = CDAFEnvironment()                           # inert_penalty_coef defaults 0.0
    env.reset()
    env.step(Introduce("x", "advocacy"))              # legal floating root
    info = _drive_to_termination(env)
    assert info["reward_breakdown"][AFF]["inert_penalty"] == pytest.approx(0.0)
    assert not info["inert_counts"]
