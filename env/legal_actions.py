"""Legal-action generation for the CDAF environment -- STRUCTURAL legality only.

Governing principle (environment_shell_spec §Governing principle): this layer
enforces exactly four things and NOTHING else:

  1. it is the acting side's turn (the round has not terminated),
  2. the speech's move budget is not exhausted,
  3. action parameters are well-formed (role/edge_type in vocab, favors points at a
     compared node, target distinct where required),
  4. the target node exists (or target = NEW),

plus the ONE ruled generator-enforced structural invariant (Ruling 2):

  5. Fence A (local, monotonic): no action may leave a REACHABLE same-side Support
     component with more than one terminal impact.

It does NOT enforce strategic legality -- response-window compliance, whether an
extension will count, whether a rebuttal-introduced chain can establish offense,
whether a spike into a conceded-but-uncontested node is inert. Those remain judge
OUTCOMES, scored as inert rather than blocked, so the agent gets the learning
signal (spec §Governing principle, reasons 1-2). Do not add such checks here.

Fence A is the sole exception, and only because deferred repair is FORBIDDEN, which
makes it a local check computable from current state + the candidate action (Ruling
2, confirmed free against the full fixture corpus: every scored fixture constructs
without ever needing deferred repair). It reuses the validator's OWN predicate
(`validator._multiterminal_components`) on the prospective round, so it cannot drift
from the termination-time structural admission it mirrors.
"""

from __future__ import annotations

import copy
from typing import List, Tuple

from .actions import (
    Introduce, Extend, Concede, Weigh, EndSpeech,
    ROLES, RELATIONSHIP_EDGE_TYPES, ATTACH_EDGE_TYPES, NEW,
)
from .state import RoundState
from .validator import _multiterminal_components


def check_legality(state: RoundState, action) -> Tuple[bool, str]:
    """Return (ok, reason). `ok` iff the action is STRUCTURALLY legal in `state`.
    `reason` is "" when legal, else a short tag explaining the rejection."""
    if state.terminated:
        return False, "terminated: no side's turn (round is over)"

    # EndSpeech is always structurally legal while the round is live.
    if isinstance(action, EndSpeech):
        return True, ""

    if state.remaining_budget <= 0:
        return False, "budget exhausted for this speech"

    if isinstance(action, Introduce):
        ok, reason = _check_introduce(state, action)
        if not ok:
            return False, reason
        return _check_fence_a(state, action)

    if isinstance(action, (Extend, Concede)):
        if action.node_id not in state.nodes:
            return False, f"target node {action.node_id!r} does not exist"
        return True, ""

    if isinstance(action, Weigh):
        for nid in (action.node_a, action.node_b):
            if nid not in state.nodes:
                return False, f"weigh target {nid!r} does not exist"
        if action.node_a == action.node_b:
            return False, "weigh compares a node with itself"
        if action.favors not in (action.node_a, action.node_b):
            return False, "favors must point at node_a or node_b"
        return True, ""

    return False, f"unknown action type: {type(action).__name__}"


def _check_introduce(state: RoundState, action: Introduce) -> Tuple[bool, str]:
    """Well-formedness + target-existence for an introduce (rules 3-4)."""
    if action.role not in ROLES:
        return False, f"role {action.role!r} not in vocabulary {sorted(ROLES)}"
    if action.target == NEW:
        if action.edge_type is not None:
            return False, "NEW node must not carry an edge_type"
        return True, ""
    # attaching to an existing node via a relationship edge
    if action.target not in state.nodes:
        return False, f"target node {action.target!r} does not exist"
    if action.edge_type not in ATTACH_EDGE_TYPES:
        return False, (f"edge_type {action.edge_type!r} not in "
                       f"{sorted(ATTACH_EDGE_TYPES)} for an attaching introduce")
    return True, ""


def _check_fence_a(state: RoundState, action: Introduce) -> Tuple[bool, str]:
    """Ruling 2: reject any introduce that would leave a reachable same-side Support
    component with >1 terminal impact. Only a Support attach or a NEW impact node can
    change terminal structure (an attack edge adds no Support edge), but we apply the
    candidate to a COPY and run the validator's own predicate unconditionally, so the
    introduce-impact and Support-union routes are both covered by one faithful check.
    The invariant is inductive: the current state already has zero reachable
    multi-terminal components, so any candidate that creates one is the offending
    action and is blocked (no transient, nothing to repair)."""
    if action.edge_type in RELATIONSHIP_EDGE_TYPES or action.target == NEW:
        probe = copy.deepcopy(state)
        probe.apply(action)
        bad = _multiterminal_components(probe.to_round())
        if bad:
            return False, (f"A/multi-terminal: would leave a same-side Support "
                           f"component with >1 terminal impact: {bad}")
    return True, ""


def legal_targets(state: RoundState) -> List[str]:
    """Every existing node id -- all are legal targets regardless of side (§Any node
    is a legal target). Own-side targeting is legal; strategic value is
    left to the judge/self-play, not restricted here."""
    return list(state.nodes)


def is_legal(state: RoundState, action) -> bool:
    """Convenience boolean over `check_legality`."""
    return check_legality(state, action)[0]
