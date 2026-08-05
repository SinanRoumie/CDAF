"""Legal-action generation for the CDAF environment -- STRUCTURAL legality only.

Governing principle (environment_shell_spec §Governing principle): this layer
enforces exactly these and NOTHING else:

  1. it is the acting side's turn (the round has not terminated),
  2. the speech's move budget is not exhausted,
  3. action parameters are well-formed (role/edge_type in vocab, favors points at a
     compared node, distinct endpoints where required),
  4. the target/endpoint nodes exist (or target = NEW),
  5. a `connect` may not create a self-loop or close a Support cycle.

It does NOT enforce strategic legality -- response-window compliance, whether an
extension will count, whether a rebuttal-introduced chain can establish offense,
whether a spike into a conceded-but-uncontested node is inert. Those remain judge
OUTCOMES, scored as inert rather than blocked, so the agent gets the learning
signal (spec §Governing principle, reasons 1-2). Do not add such checks here.

(There is no Fence A: as of judge v11 divergent chains are first-class, so a
same-side Support component with more than one terminal impact is legal. The old
per-action multi-terminal probe -- a deepcopy + full re-materialization on every
introduce/weigh candidate, ~76ms/round -- is gone with it, so legality checks are
now O(1) structural predicates. The one non-local check that remains is the Support
cycle test for `connect`, which is a cheap reachability query, not a re-judge.)
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Tuple

from .actions import (
    Introduce, Extend, Concede, Weigh, Connect, EndSpeech,
    ROLES, RELATIONSHIP_EDGE_TYPES, ATTACH_EDGE_TYPES, NEW,
)
from .state import RoundState, action_cost


def check_legality(state: RoundState, action) -> Tuple[bool, str]:
    """Return (ok, reason). `ok` iff the action is STRUCTURALLY legal in `state`.
    `reason` is "" when legal, else a short tag explaining the rejection.

    Two gates: (1) structural validity (existence, well-formedness, the connect cycle
    rule), then (2) AFFORDABILITY -- the action's cost must not exceed the speech's
    remaining budget. Cost is variable: `extend`/`concede` cost ceil(path_length / K)
    over the walk they stamp, so a long-path extend late in a speech is unaffordable
    (masked) while a short-path one or a cost-1 `introduce` stays legal. This replaces
    the old flat `remaining_budget <= 0` guard, which assumed every move cost 1."""
    if state.terminated:
        return False, "terminated: no side's turn (round is over)"

    # EndSpeech is always legal while the round is live (it costs 0 and ends the turn).
    if isinstance(action, EndSpeech):
        return True, ""

    ok, reason = _structural_legal(state, action)
    if not ok:
        return False, reason

    cost = action_cost(state, action)
    if cost > state.remaining_budget:
        return False, (f"unaffordable: {type(action).__name__} costs {cost} slot(s), "
                       f"{state.remaining_budget} left this speech")
    return True, ""


def _structural_legal(state: RoundState, action) -> Tuple[bool, str]:
    """Structural validity only (existence, parameter well-formedness, the connect
    Support-cycle rule) -- NO budget. Affordability is a separate gate in
    `check_legality`, since an action's cost is now variable."""
    if isinstance(action, Introduce):
        return _check_introduce(state, action)

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

    if isinstance(action, Connect):
        for nid in (action.source_id, action.target_id):
            if nid not in state.nodes:
                return False, f"connect endpoint {nid!r} does not exist"
        if action.source_id == action.target_id:
            return False, "connect would create a self-loop"
        if action.edge_type not in RELATIONSHIP_EDGE_TYPES:
            return False, (f"edge_type {action.edge_type!r} not in "
                           f"{sorted(RELATIONSHIP_EDGE_TYPES)} for connect")
        if action.edge_type == "support" and _closes_support_cycle(
                state, action.source_id, action.target_id):
            return False, "connect would close a Support cycle"
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


def _closes_support_cycle(state: RoundState, source: str, target: str) -> bool:
    """Would the authored Support edge source->target close a DIRECTED cycle? Follows
    authored source->target Support edges and returns True iff `target` already reaches
    `source` (so source->target would complete a loop). A directed-cycle test -- NOT an
    undirected one -- deliberately: convergence/divergence DAGs (two paths oriented
    toward a shared impact, e.g. r32's shared uniqueness, or a cross-side shared impact)
    have no directed cycle and stay buildable, which is the whole point of `connect`;
    only genuine circular support (a->b->...->a) is refused. Cheap O(edges) reachability,
    not a re-judge. Note: because the judge is direction-agnostic, this bans authored
    directed loops, not every undirected cycle -- undirected cycles that orient as DAGs
    are exactly the audited convergence structures."""
    out = defaultdict(list)
    for e in state.edges:
        if e.edge_type == "support":
            out[e.source].append(e.target)
    seen: set = set()
    stack = list(out[target])
    while stack:
        cur = stack.pop()
        if cur == source:
            return True
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(out[cur])
    return False


def legal_targets(state: RoundState) -> List[str]:
    """Every existing node id -- all are legal targets regardless of side (§Any node
    is a legal target). Own-side targeting is legal; strategic value is
    left to the judge/self-play, not restricted here."""
    return list(state.nodes)


def is_legal(state: RoundState, action) -> bool:
    """Convenience boolean over `check_legality`."""
    return check_legality(state, action)[0]
