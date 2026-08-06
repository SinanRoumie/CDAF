"""Tests for ATOMIC extend/concede with speech-wide batched cost.

The mechanic (action_schema_spec §extend / environment_shell_spec §Liveness stamping):
  * extend/concede stamp ONLY the named node -- no path-walk, no propagation, so
    partial-chain carriage is expressible (extend a link, let its impact die);
  * cost is the MARGINAL of a speech-wide `ceil(count / K)` batch:
        marginal = ceil((count+1)/K) - ceil(count/K)
    (count = extends_this_speech) -- 1 on the 1st, (K+1)-th, (2K+1)-th ... carriage of
    the speech, 0 otherwise, so N carriages cost ceil(N/K) slots total. The discount is
    speech-wide (same rate whether the K nodes are on one chain or scattered).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

from env.state import RoundState, NodeRecord, action_cost
from env.actions import Introduce, Extend, Concede, Weigh, Connect, EndSpeech, EXTEND_COST_K, NEW
from env import CDAFEnvironment, is_legal
from policy import LegalActionMask
from policy.masking import ACTION_TYPES

K = EXTEND_COST_K


def _state_with_node(nid="a", *, moves_used=0, extends_this_speech=0):
    """A minimal 1AC state carrying one node, with the counters set directly (so cost /
    affordability can be probed without driving a whole speech)."""
    st = RoundState()
    st.nodes[nid] = NodeRecord(nid, "", "AFF", "1AC", "link", {"1AC"})
    st.moves_used = moves_used
    st.extends_this_speech = extends_this_speech
    return st


# --- marginal cost by count (worked examples) --------------------------------

def test_marginal_cost_by_count():
    assert K == 4
    st = _state_with_node()
    marginals = []
    for c in range(9):
        st.extends_this_speech = c
        marginals.append(action_cost(st, Extend("a")))
    # 1 on the 1st, 5th, 9th carriage (count 0, 4, 8); 0 otherwise
    assert marginals == [1, 0, 0, 0, 1, 0, 0, 0, 1]
    # cumulative cost of N carriages == ceil(N / K)
    cumulative = [sum(marginals[:n]) for n in range(1, 10)]
    assert cumulative == [1, 1, 1, 1, 2, 2, 2, 2, 3]


def test_concede_costs_the_same_as_extend():
    st = _state_with_node()
    for c in (0, 1, 3, 4):
        st.extends_this_speech = c
        assert action_cost(st, Concede("a")) == action_cost(st, Extend("a"))


def test_non_carriage_costs_unchanged():
    st = _state_with_node()
    assert action_cost(st, Introduce("", "link", NEW)) == 1
    assert action_cost(st, Weigh("a", "a", "a")) == 1
    assert action_cost(st, Connect("a", "a", "support")) == 1
    assert action_cost(st, EndSpeech()) == 0


# --- atomic stamping ---------------------------------------------------------

def test_extend_stamps_only_the_named_node():
    """Extending a link at a later speech stamps only the link -- not its advocacy,
    uniqueness, or impact (partial-chain carriage)."""
    env = CDAFEnvironment(); env.reset()
    env.step(Introduce("", "advocacy", NEW)); adv = list(env.state.nodes)[-1]
    env.step(Introduce("", "uniqueness", adv, "support")); uni = list(env.state.nodes)[-1]
    env.step(Introduce("", "link", uni, "support")); lk = list(env.state.nodes)[-1]
    env.step(Introduce("", "impact", lk, "support")); im = list(env.state.nodes)[-1]
    env.step(EndSpeech()); env.step(EndSpeech())              # -> 2AC (AFF)
    before = {n: set(env.state.nodes[n].carried) for n in (adv, uni, lk, im)}
    env.step(Extend(lk))
    after = {n: set(env.state.nodes[n].carried) for n in (adv, uni, lk, im)}
    assert after[lk] - before[lk] == {"2AC"}                  # only the link gained 2AC
    assert after[adv] == before[adv] and after[uni] == before[uni] and after[im] == before[im]
    assert not hasattr(env.state, "extend_path")             # walk machinery removed


# --- speech-wide batching (discount is not chain-scoped) ---------------------

def test_batch_discount_is_speech_wide_across_unrelated_nodes():
    """Carrying K unrelated (unconnected) nodes costs one slot total -- the same
    `1-slot-per-K` rate a single chain would get; the discount is speech-wide."""
    env = CDAFEnvironment(); env.reset()
    ids = []
    for _ in range(K + 1):                                    # K+1 isolated AFF nodes
        env.step(Introduce("", "link", NEW)); ids.append(list(env.state.nodes)[-1])
    env.step(EndSpeech()); env.step(EndSpeech())              # -> 2AC
    used0 = env.state.moves_used
    for nid in ids:                                           # extend all K+1 (scattered)
        env.step(Extend(nid))
    # K+1 carriages -> ceil((K+1)/K) = 2 slots
    assert env.state.moves_used - used0 == 2
    assert env.state.extends_this_speech == K + 1


def test_counter_resets_each_speech():
    env = CDAFEnvironment(); env.reset()
    env.step(Introduce("", "link", NEW)); lk = list(env.state.nodes)[-1]
    env.step(EndSpeech()); env.step(EndSpeech())              # -> 2AC
    env.step(Extend(lk)); env.step(Extend(lk))
    assert env.state.extends_this_speech == 2
    env.step(EndSpeech())                                     # -> 2NC/1NR (NEG)
    assert env.state.extends_this_speech == 0                 # reset at the boundary


# --- masking / affordability -------------------------------------------------

def test_group_starting_carriage_blocked_at_zero_budget_free_one_allowed():
    """With no budget left, the carriage that would START a new K-group (cost 1) is
    illegal, but a mid-group carriage (cost 0) stays legal regardless of headroom --
    affordability keys on the marginal cost, not a flat 1."""
    budget = RoundState().slot_budget                        # 1AC budget

    # next carriage starts a new K-group (count 0) -> cost 1 -> blocked at 0 budget
    st = _state_with_node(moves_used=budget, extends_this_speech=0)
    assert st.remaining_budget == 0
    assert action_cost(st, Extend("a")) == 1
    assert not is_legal(st, Extend("a"))
    assert not LegalActionMask(st).extend_target_mask().any()
    assert not LegalActionMask(st).type_mask()[ACTION_TYPES.index("extend")]

    # next carriage is mid-group (count 1) -> cost 0 -> legal even at 0 budget
    st = _state_with_node(moves_used=budget, extends_this_speech=1)
    assert st.remaining_budget == 0
    assert action_cost(st, Extend("a")) == 0
    assert is_legal(st, Extend("a"))
    assert LegalActionMask(st).extend_target_mask().all()


def test_marginal_affordability_is_a_simple_lookup():
    """The next carriage's cost depends only on the current count -- no lookahead."""
    st = _state_with_node(moves_used=RoundState().slot_budget - 1, extends_this_speech=3)
    # count 3 -> next (the 4th) is mid-group, cost 0; count 4 -> next (5th) starts a group
    assert action_cost(st, Extend("a")) == 0 and is_legal(st, Extend("a"))
    st.extends_this_speech = 4
    assert action_cost(st, Extend("a")) == 1 and is_legal(st, Extend("a"))   # still 1 slot left
    st.moves_used = RoundState().slot_budget                                 # 0 left
    assert not is_legal(st, Extend("a"))                                     # now blocked
