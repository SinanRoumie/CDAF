"""Tests for chain-level extend/concede and its variable cost (ceil(path_len / K)).

Covers the four behaviours the ruling introduces:
  * `action_cost` across path lengths (K=4 worked examples: 2->1, 6->2, 8->2, 9->3),
  * an extend that is affordable vs. one that is not, given remaining budget (masked,
    not merely discouraged),
  * divergent-branch INDEPENDENT pricing (two branches off one trunk each pay their own
    path cost; no pooling/discount),
  * idempotent re-stamping still costs full price (the trunk is never banked).

States are built directly as `RoundState` graphs where budget control matters (so a
9-node path can be priced without fighting the 1AC budget), and via the real
environment where the masking path is what's under test.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

from env.state import RoundState, NodeRecord, EdgeRecord, action_cost, _extend_slot_cost
from env.actions import Introduce, Extend, Concede, EndSpeech, EXTEND_COST_K, NEW
from env import CDAFEnvironment, is_legal
from policy import LegalActionMask
from policy.masking import ACTION_TYPES


def _linear_chain(length: int, *, owner="AFF", speech="1AC") -> RoundState:
    """A single linear Support spine of `length` nodes: advocacy - link... - impact,
    all same side/speech. `extend_path` from any node returns all `length` nodes (no
    satellites), so it isolates the cost formula from walk-shape effects."""
    st = RoundState()
    ids = []
    for i in range(length):
        role = "advocacy" if i == 0 else ("impact" if i == length - 1 else "link")
        nid = f"n{i}"
        st.nodes[nid] = NodeRecord(id=nid, content="", owner=owner,
                                   introduction_speech=speech, role=role, carried={speech})
        ids.append(nid)
    for i in range(length - 1):
        # source = the node farther from advocacy (supports the one before it)
        st.edges.append(EdgeRecord(id=f"e{i}", source=ids[i + 1], target=ids[i],
                                   edge_type="support"))
    return st, ids


# --- action_cost across path lengths -----------------------------------------

def test_extend_slot_cost_worked_examples():
    assert EXTEND_COST_K == 4
    assert _extend_slot_cost(2) == 1
    assert _extend_slot_cost(6) == 2
    assert _extend_slot_cost(8) == 2
    assert _extend_slot_cost(9) == 3
    assert _extend_slot_cost(1) == 1            # never below 1


@pytest.mark.parametrize("length,expected", [(2, 1), (6, 2), (8, 2), (9, 3)])
def test_action_cost_scales_with_path_length(length, expected):
    st, ids = _linear_chain(length)
    impact = ids[-1]
    assert len(st.extend_path(impact)) == length        # whole spine walked
    assert action_cost(st, Extend(impact)) == expected
    assert action_cost(st, Concede(impact)) == expected
    # naming any node on the chain prices the same whole path
    assert action_cost(st, Extend(ids[0])) == expected


def test_non_extend_actions_cost_one_endspeech_zero():
    st, ids = _linear_chain(3)
    assert action_cost(st, Introduce("", "link", NEW)) == 1
    assert action_cost(st, EndSpeech()) == 0


def test_walk_includes_satellite_uniqueness():
    """A satellite Uniqueness hanging off a spine node is stamped (and thus priced),
    per the resolved Open Question 1."""
    st = RoundState()
    st.nodes["adv"] = NodeRecord("adv", "", "AFF", "1AC", "advocacy", {"1AC"})
    st.nodes["lk"] = NodeRecord("lk", "", "AFF", "1AC", "link", {"1AC"})
    st.nodes["im"] = NodeRecord("im", "", "AFF", "1AC", "impact", {"1AC"})
    st.nodes["u"] = NodeRecord("u", "", "AFF", "1AC", "uniqueness", {"1AC"})
    st.edges += [EdgeRecord("e1", "lk", "adv", "support"),
                 EdgeRecord("e2", "im", "lk", "support"),
                 EdgeRecord("e3", "u", "lk", "support")]    # u is a satellite off lk
    path = st.extend_path("im")
    assert path == {"adv", "lk", "im", "u"}                 # satellite included
    assert action_cost(st, Extend("im")) == 1               # ceil(4/4)


# --- affordability / masking -------------------------------------------------

def test_extend_masked_when_unaffordable():
    """A cost-2 extend is illegal (masked) when only 1 slot remains, and legal when 2
    remain -- structural unaffordability, same category as budget exhaustion."""
    st, ids = _linear_chain(6)                  # cost ceil(6/4) = 2
    impact = ids[-1]
    assert action_cost(st, Extend(impact)) == 2

    st.moves_used = st.slot_budget - 1          # 1 slot left
    assert not is_legal(st, Extend(impact))
    mask = LegalActionMask(st)
    assert not mask.extend_target_mask().any()  # every extend target too expensive
    assert not mask.type_mask()[ACTION_TYPES.index("extend")]   # extend type masked off

    st.moves_used = st.slot_budget - 2          # 2 slots left
    assert is_legal(st, Extend(impact))
    assert LegalActionMask(st).extend_target_mask().all()


def test_short_path_affordable_when_long_path_is_not():
    """With 1 slot left, a cost-1 (short) chain's extend stays legal while a cost-2
    chain's extend is masked -- affordability is per-target, not a flat gate."""
    st = RoundState()
    # short chain (2 nodes -> cost 1): a1 - i1
    st.nodes["a1"] = NodeRecord("a1", "", "AFF", "1AC", "advocacy", {"1AC"})
    st.nodes["i1"] = NodeRecord("i1", "", "AFF", "1AC", "impact", {"1AC"})
    st.edges.append(EdgeRecord("e1", "i1", "a1", "support"))
    # long chain (6 nodes -> cost 2), disjoint component
    for k, (nid, role) in enumerate([("a2", "advocacy"), ("l1", "link"), ("l2", "link"),
                                     ("l3", "link"), ("l4", "link"), ("i2", "impact")]):
        st.nodes[nid] = NodeRecord(nid, "", "AFF", "1AC", role, {"1AC"})
    chain = ["a2", "l1", "l2", "l3", "l4", "i2"]
    for x, y in zip(chain, chain[1:]):
        st.edges.append(EdgeRecord(f"e_{x}", y, x, "support"))

    st.moves_used = st.slot_budget - 1          # exactly 1 slot left
    assert action_cost(st, Extend("i1")) == 1 and is_legal(st, Extend("i1"))
    assert action_cost(st, Extend("i2")) == 2 and not is_legal(st, Extend("i2"))
    m = LegalActionMask(st)
    ids = m.node_ids
    em = m.extend_target_mask()
    assert em[ids.index("i1")] and not em[ids.index("i2")]


# --- divergence: independent pricing, no discount ----------------------------

def _divergent() -> RoundState:
    """Shared advocacy trunk fanning to two impacts:
       adv <- la <- ia   and   adv <- lb <- ib   (support arrows point toward adv)."""
    st = RoundState()
    for nid, role in [("adv", "advocacy"), ("la", "link"), ("lb", "link"),
                      ("ia", "impact"), ("ib", "impact")]:
        st.nodes[nid] = NodeRecord(nid, "", "AFF", "1AC", role, {"1AC"})
    st.edges += [EdgeRecord("e1", "la", "adv", "support"),
                 EdgeRecord("e2", "lb", "adv", "support"),
                 EdgeRecord("e3", "ia", "la", "support"),
                 EdgeRecord("e4", "ib", "lb", "support")]
    return st


def test_divergent_branches_priced_independently():
    """Naming one branch stamps only that branch's root->impact path (trunk included,
    the other branch excluded); each branch costs its own path length."""
    st = _divergent()
    assert st.extend_path("ia") == {"ia", "la", "adv"}     # trunk in, branch b out
    assert st.extend_path("ib") == {"ib", "lb", "adv"}
    assert action_cost(st, Extend("ia")) == 1              # ceil(3/4)
    assert action_cost(st, Extend("ib")) == 1


def test_shared_trunk_selection_covers_only_one_branch_no_bulk_discount():
    """Naming the shared root/trunk node stamps a SINGLE root->impact path (one branch,
    deterministically), never every branch through it -- so there is no bulk-extend
    discount. It covers exactly one terminal impact; the other still needs its own
    extend."""
    st = _divergent()
    walk = st.extend_path("adv")
    impacts_covered = {n for n in walk if st.nodes[n].role == "impact"}
    assert len(impacts_covered) == 1                        # NOT both ia and ib
    assert walk in ({"adv", "la", "ia"}, {"adv", "lb", "ib"})
    # deterministic tiebreak -> the smaller impact id ("ia")
    assert walk == {"adv", "la", "ia"}
    assert action_cost(st, Extend("adv")) == 1


def test_worked_example_trunk_cannot_undercut_two_branch_extends():
    """Worked example: 2 trunk nodes + two 3-node branches (each full path = 5 nodes).
    Extending both branches costs ceil(5/4)+ceil(5/4) = 2+2 = 4. Naming the shared
    trunk/root cannot cover both impacts for less -- it covers ONE branch (cost 2), so
    covering both still costs 4. No pooling loophole."""
    st = RoundState()
    roles = {"R": "advocacy", "T": "link", "a1": "link", "a2": "link", "iA": "impact",
             "b1": "link", "b2": "link", "iB": "impact"}
    for nid, role in roles.items():
        st.nodes[nid] = NodeRecord(nid, "", "AFF", "1AC", role, {"1AC"})
    for s, t in [("T", "R"), ("a1", "T"), ("a2", "a1"), ("iA", "a2"),
                 ("b1", "T"), ("b2", "b1"), ("iB", "b2")]:
        st.edges.append(EdgeRecord(f"e_{s}{t}", s, t, "support"))

    # (a) each branch tip: full 5-node path, cost 2 each; both branches = 4 total
    assert len(st.extend_path("iA")) == 5 and action_cost(st, Extend("iA")) == 2
    assert len(st.extend_path("iB")) == 5 and action_cost(st, Extend("iB")) == 2
    assert action_cost(st, Extend("iA")) + action_cost(st, Extend("iB")) == 4

    # (b) naming the trunk/root covers ONE branch (one terminal impact), cost 2 -- it
    # does NOT stamp all 8 nodes / both impacts for 2 (the old bulk bug).
    for anchor in ("T", "R"):
        walk = st.extend_path(anchor)
        impacts = {n for n in walk if st.nodes[n].role == "impact"}
        assert len(impacts) == 1, f"{anchor} bulk-covered {impacts}"
        assert action_cost(st, Extend(anchor)) == 2
    # covering the branch the trunk-extend missed still costs a full separate extend
    assert action_cost(st, Extend("iB")) == 2


def test_two_branches_cost_the_sum_no_trunk_discount():
    """Extending both branches costs the sum of their independent path costs; the shared
    trunk is re-walked and re-paid, never banked after the first branch."""
    st = _divergent()
    st.moves_used = 0
    st.apply(Extend("ia"))
    after_a = st.moves_used
    assert after_a == 1
    st.apply(Extend("ib"))
    assert st.moves_used == after_a + 1                    # +full cost, no discount
    # both branches (and the shared trunk) are now carried this speech
    for nid in ("adv", "la", "ia", "lb", "ib"):
        assert "1AC" in st.nodes[nid].carried


# --- idempotent re-stamping still costs full price ---------------------------

def test_idempotent_restamp_still_full_cost():
    """Re-extending an already-stamped chain in the same speech changes no liveness but
    still costs full price (no banking)."""
    st, ids = _linear_chain(6)                  # cost 2 per extend
    impact = ids[-1]
    st.moves_used = 0
    carried_before = {nid: set(st.nodes[nid].carried) for nid in ids}
    st.apply(Extend(impact))
    assert st.moves_used == 2
    carried_after_first = {nid: set(st.nodes[nid].carried) for nid in ids}
    # first extend already stamped the whole chain for 1AC (all were introduced at 1AC,
    # so the liveness set is unchanged -- idempotent) ...
    assert carried_after_first == carried_before
    # ... and a second, redundant extend re-walks the trunk and pays full price again.
    st.apply(Extend(impact))
    assert st.moves_used == 4
    assert {nid: set(st.nodes[nid].carried) for nid in ids} == carried_before


def test_env_extend_charges_variable_cost_end_to_end():
    """Through the real env: build a 6-node AFF spine, carry it at 2AC with a single
    extend, and confirm the whole spine is stamped for one cost-2 move."""
    env = CDAFEnvironment(); env.reset()
    # 1AC: build advocacy-link-link-link-link-impact (6 nodes, 6 introduces <= budget 8)
    env.step(Introduce("", "advocacy", NEW))
    adv = list(env.state.nodes)[-1]
    prev = adv
    for _ in range(4):
        env.step(Introduce("", "link", prev, "support")); prev = list(env.state.nodes)[-1]
    env.step(Introduce("", "impact", prev, "support")); impact = list(env.state.nodes)[-1]
    spine = list(env.state.nodes)                # all 6
    env.step(EndSpeech())                         # -> 1NC
    env.step(EndSpeech())                         # -> 2AC (AFF)
    used_before = env.state.moves_used
    env.step(Extend(impact))                      # one chain-level extend
    assert env.state.moves_used - used_before == 2        # ceil(6/4)
    for nid in spine:
        assert "2AC" in env.state.nodes[nid].carried      # whole spine carried at 2AC
