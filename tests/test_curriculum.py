"""Opening unlock-curriculum mask (rl_training_spec §Opening curriculum;
encoder_spec §Curriculum mask composition).

The curriculum is a POLICY-LAYER training scaffold that AND-composes with the env
legal-action mask during the AFF 1AC. These tests pin its composition contract:
  * move 0 of the 1AC admits exactly one action: introduce(advocacy, NEW); end_speech masked;
  * each ladder rung unlocks exactly the intended roles and no others;
  * BallotDirective unlocks on an Impact ALONE, with no Uniqueness read (the r36 case);
  * the mask is identity for every non-1AC slot and every NEG slot;
  * the curriculum never admits an action the env legal-action mask rejects (never widens).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

from env import CDAFEnvironment, NEW
from env.actions import Introduce, EndSpeech
from env.observation import observe
from policy import GraphEncoder, ActorCritic, LegalActionMask
from policy.masking import ACTION_TYPES, ACTION_ROLE_ORDER
from policy.curriculum import (
    CurriculumLegalActionMask, allowed_roles, allowed_types, curriculum_binds,
)


# --- builders ----------------------------------------------------------------

def _build(chain):
    """chain: list of (role, parent_index_or_None). None -> a NEW root; an index ->
    support-attach to the already-built node at that index. Returns the RoundState."""
    env = CDAFEnvironment(); env.reset()
    ids = []
    for role, parent in chain:
        if parent is None:
            env.step(Introduce("", role, NEW))
        else:
            env.step(Introduce("", role, ids[parent], "support"))
        ids.append(list(env.state.nodes)[-1])
    return env.state


def _roleset(mask) -> set:
    return {r for r, b in zip(ACTION_ROLE_ORDER, mask) if b}


_OPENING = None                                   # env.reset() -- empty 1AC graph
_ADV = [("advocacy", None)]
_ADV_LINK = [("advocacy", None), ("link", 0)]
_ADV_LINK_IMPACT = [("advocacy", None), ("link", 0), ("impact", 1)]


def _reset_state():
    env = CDAFEnvironment(); env.reset()
    return env.state


# --- move 0: exactly introduce(advocacy, NEW) --------------------------------

def test_move0_admits_only_advocacy_new():
    state = _reset_state()
    cm = CurriculumLegalActionMask(state)
    tmask = cm.type_mask()
    # only `introduce` is legal; end_speech (and everything else) masked
    assert _set_types(tmask) == {"introduce"}
    assert tmask[ACTION_TYPES.index("end_speech")] == False
    # the only introduce target is the NEW sentinel (empty graph -> mask width 1)
    itmask = cm.introduce_target_mask()
    assert itmask.tolist() == [True]              # just the trailing NEW slot
    # and the only role for NEW is advocacy
    assert _roleset(cm.introduce_role_mask(NEW)) == {"advocacy"}


def _set_types(mask):
    return {t for t, b in zip(ACTION_TYPES, mask) if b}


# --- ladder rungs: exactly the intended roles --------------------------------

def test_ladder_rungs_unlock_exactly_intended_roles():
    # move 0 (empty): advocacy only
    assert _roleset(allowed_roles(_reset_state())) == {"advocacy"}
    # after advocacy: advocacy + link (link unlocked from move 1)
    assert _roleset(allowed_roles(_build(_ADV))) == {"advocacy", "link"}
    # reading a Link unlocks uniqueness + impact (but NOT framework/ballot_directive yet)
    assert _roleset(allowed_roles(_build(_ADV_LINK))) == {
        "advocacy", "link", "uniqueness", "impact"}
    # reading an Impact unlocks framework + ballot_directive (all six now available)
    assert _roleset(allowed_roles(_build(_ADV_LINK_IMPACT))) == set(ACTION_ROLE_ORDER)


def test_bd_gates_on_impact_alone_not_uniqueness():
    # BEFORE an Impact is read: ballot_directive (and framework) are locked...
    before = _roleset(allowed_roles(_build(_ADV_LINK)))
    assert "ballot_directive" not in before and "framework" not in before
    # ...and AFTER an Impact -- with NO Uniqueness ever read (the r36 shape) -- BD unlocks.
    after = _roleset(allowed_roles(_build(_ADV_LINK_IMPACT)))
    assert "ballot_directive" in after and "framework" in after
    # sanity: the r36 chain really has no uniqueness node
    state = _build(_ADV_LINK_IMPACT)
    assert "uniqueness" not in {rec.role for rec in state.nodes.values()}


# --- identity off the AFF 1AC ------------------------------------------------

def _advance_to(slot):
    """Return a state at `slot` (an advocacy is read so the 1AC is non-empty first)."""
    env = CDAFEnvironment(); env.reset()
    env.step(Introduce("", "advocacy", NEW))
    while env.state.current_slot != slot:
        env.step(EndSpeech())
    return env.state


def test_identity_on_neg_slot_and_non_1ac_aff_slot():
    for slot in ("1NC", "2AC", "2NR"):           # NEG (1NC/2NR) and a later AFF slot (2AC)
        state = _advance_to(slot)
        assert curriculum_binds(state) is False
        assert np.array_equal(allowed_types(state), np.ones(len(ACTION_TYPES), bool))
        assert np.array_equal(allowed_roles(state), np.ones(len(ACTION_ROLE_ORDER), bool))
        cm, lm = CurriculumLegalActionMask(state), LegalActionMask(state)
        assert np.array_equal(cm.type_mask(), lm.type_mask())
        assert np.array_equal(cm.introduce_role_mask(NEW), lm.introduce_role_mask(NEW))
        for nid in state.nodes:                  # attaching-target role masks unchanged too
            assert np.array_equal(cm.introduce_role_mask(nid), lm.introduce_role_mask(nid))


# --- never widens legality ---------------------------------------------------

def test_curriculum_never_admits_an_env_illegal_action():
    states = [
        _reset_state(), _build(_ADV), _build(_ADV_LINK), _build(_ADV_LINK_IMPACT),
        _advance_to("1NC"), _advance_to("2AC"),
    ]
    for state in states:
        cm, lm = CurriculumLegalActionMask(state), LegalActionMask(state)
        # composed => env for the type head...
        assert not (cm.type_mask() & ~lm.type_mask()).any()
        # ...and for the introduce-role head at NEW and at every existing target.
        for tgt in [NEW, *state.nodes]:
            assert not (cm.introduce_role_mask(tgt) & ~lm.introduce_role_mask(tgt)).any()


# --- wiring: ActorCritic honours the flag ------------------------------------

def test_actorcritic_curriculum_flag_forces_opening_advocacy():
    torch.manual_seed(0)
    enc = GraphEncoder()
    ac = ActorCritic(enc, curriculum=True); ac.eval()
    env = CDAFEnvironment(); env.reset()
    out = ac.evaluate(observe(env.state))
    gen = torch.Generator(); gen.manual_seed(0)
    sa = ac.sample_action(out, env.state, generator=gen)
    # move 0 under the curriculum has exactly one legal action, so sampling is forced.
    assert isinstance(sa.action, Introduce)
    assert sa.action.role == "advocacy" and sa.action.target == NEW


def test_actorcritic_curriculum_off_by_default():
    enc = GraphEncoder()
    ac = ActorCritic(enc)
    assert ac.curriculum is False
