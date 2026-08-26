"""Opening unlock-curriculum mask (rl_training_spec §Opening curriculum).

A POLICY-LAYER training scaffold, NOT a legality rule. It AND-composes with the
environment's legal-action mask (never widening) to gate the AFF 1AC opening into a
progressive unlock ladder, so warm-start / early PPO explore a real case-construction
opening rather than the full factored action space from an empty graph. It binds on the
**AFF 1AC only** -- identity from the 1NC onward and on every NEG slot -- and is never
consulted by the judge (encoder_spec §Curriculum mask composition).

The ladder (AFF 1AC only):

    move 0 (empty graph) -> only introduce(role=advocacy, target=NEW); end_speech masked
                            (AFF may not pass the 1AC).
    move >= 1            -> advocacy and link always available; reading a Link unlocks
                            {uniqueness, impact}; reading an Impact unlocks
                            {framework, ballot_directive}. BallotDirective gates on Impact
                            ALONE, never on Uniqueness (the r36 case; the retracted
                            mandatory-uniqueness rule must not return).

`weigh` / `connect` / attaching `introduce` / attack edges need no curriculum gate: they
are already structurally unavailable on an empty graph and become available naturally as
their endpoints appear, so the ladder gates only the *role* of a fresh `introduce` and
the availability of `end_speech` at move 0.
"""
from __future__ import annotations

import numpy as np

from .masking import LegalActionMask, ACTION_TYPES, ACTION_ROLE_ORDER

_N_TYPES = len(ACTION_TYPES)
_N_ROLES = len(ACTION_ROLE_ORDER)
_INTRODUCE = ACTION_TYPES.index("introduce")


def curriculum_binds(state) -> bool:
    """The opening ladder binds only during the AFF 1AC; everywhere else it is identity."""
    return state.current_slot == "1AC" and state.current_side == "AFF"


def _at_opening(state) -> bool:
    """Move 0 of the 1AC -- the first decision, taken on the empty graph."""
    return len(state.nodes) == 0


def _roles_read(state):
    return {rec.role for rec in state.nodes.values()}


def allowed_types(state) -> np.ndarray:
    """(len(ACTION_TYPES),) bool overlay over ACTION_TYPES. At move 0 of the AFF 1AC only
    `introduce` is permitted (`end_speech` masked -- AFF may not pass the 1AC); everywhere
    else this is all-True (identity)."""
    m = np.ones(_N_TYPES, dtype=bool)
    if curriculum_binds(state) and _at_opening(state):
        m[:] = False
        m[_INTRODUCE] = True
    return m


def allowed_roles(state) -> np.ndarray:
    """(len(ACTION_ROLE_ORDER),) bool overlay over ACTION_ROLE_ORDER: which introduce roles
    the ladder permits at this decision. All-True (identity) outside the AFF 1AC."""
    if not curriculum_binds(state):
        return np.ones(_N_ROLES, dtype=bool)
    if _at_opening(state):
        allow = {"advocacy"}                              # move 0: advocacy only
    else:
        read = _roles_read(state)
        allow = {"advocacy", "link"}                      # advocacy always; link from move 1
        if "link" in read:
            allow |= {"uniqueness", "impact"}             # a Link unlocks uniqueness + impact
        if "impact" in read:
            allow |= {"framework", "ballot_directive"}    # an Impact unlocks framework + BD
    return np.array([r in allow for r in ACTION_ROLE_ORDER], dtype=bool)


class CurriculumLegalActionMask(LegalActionMask):
    """`LegalActionMask` AND-composed with the opening unlock ladder. It can only turn a
    legal action OFF for sampling (never widen legality), and is identity outside the AFF
    1AC. Every other stage mask is inherited unchanged, so the composition touches only the
    type head (masking `end_speech` at move 0) and the introduce-role head (the ladder)."""

    def type_mask(self) -> np.ndarray:
        return super().type_mask() & allowed_types(self.state)

    def introduce_role_mask(self, target) -> np.ndarray:
        return super().introduce_role_mask(target) & allowed_roles(self.state)
