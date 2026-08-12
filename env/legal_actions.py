"""Legal-action generation for the CDAF environment -- STRUCTURAL legality only.

Governing principle (environment_shell_spec §Governing principle): this layer
enforces exactly these and NOTHING else:

  1. it is the acting side's turn (the round has not terminated),
  2. the speech's move budget is not exhausted,
  3. action parameters are well-formed (role/edge_type in vocab, favors points at a
     compared node, distinct endpoints where required),
  4. the target/endpoint nodes exist (or target = NEW),
  5. a `connect` may not create a self-loop or close a Support cycle,
  6. STRUCTURAL INCOHERENCE is illegal -- three moves that can never be meaningful in
     ANY round state (so masking them removes no strategic distinction):
       - a same-side attack (an attack edge between two nodes of the same side),
       - an offense at a non-polarity node (an offensive_attack where an endpoint is
         not offense-bearing -- not a Link or Impact),
       - a redundant connect (a `connect` duplicating an existing edge).

It does NOT enforce STRATEGIC legality -- response-window compliance, whether an
extension will count, whether a rebuttal-introduced chain can establish offense,
whether a spike into a conceded-but-uncontested node is inert. Those remain judge
OUTCOMES, scored as inert rather than blocked, so the agent gets the learning
signal (spec §Governing principle, reasons 1-2). Do not add such checks here. The
one CONTEXT-DEPENDENT inert move -- a no-op re-extend -- also stays legal (it is a
real move in most states); it is priced via COST (a full slot; see state.action_cost)
and diagnosed reward-side by `is_inert`, never blocked.

(There is no Fence A: as of judge v11 divergent chains are first-class, so a
same-side Support component with more than one terminal impact is legal. The old
per-action multi-terminal probe -- a deepcopy + full re-materialization on every
introduce/weigh candidate, ~76ms/round -- is gone with it, so legality checks are
now O(1) structural predicates. The one non-local check that remains is the Support
cycle test for `connect`, which is a cheap reachability query, not a re-judge.)
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Optional, Tuple

from .actions import (
    Introduce, Extend, Concede, Weigh, Connect, EndSpeech,
    ROLES, RELATIONSHIP_EDGE_TYPES, ATTACH_EDGE_TYPES, ATTACK_EDGE_TYPES,
    OFFENSE_BEARING_ROLES, ROOT_ELIGIBLE_ROLES, NEW,
)
from .state import RoundState, action_cost


def check_legality(state: RoundState, action) -> Tuple[bool, str]:
    """Return (ok, reason). `ok` iff the action is STRUCTURALLY legal in `state`.
    `reason` is "" when legal, else a short tag explaining the rejection.

    Two gates: (1) structural validity (existence, well-formedness, the connect cycle
    rule), then (2) AFFORDABILITY -- the action's cost must not exceed the speech's
    remaining budget. `extend`/`concede` cost is the MARGINAL of a speech-wide
    `ceil(count / K)` batch (`state.action_cost`): 0 on most carriages, 1 on the one
    that starts a new K-group, so a carriage is only ever unaffordable when it would
    start a new K-group with no budget left. The marginal cost of the next carriage is
    a simple lookup on `extends_this_speech` -- no lookahead over the rest of the speech
    is needed."""
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
        # Rule 6: redundant connect (duplicates an existing edge -- adds no structure).
        if _duplicate_edge(state, action.source_id, action.target_id, action.edge_type):
            return False, ("redundant connect: an edge (same source, target, edge_type) "
                           "already exists")
        # Rule 6: structural incoherence of an attack edge between two existing nodes.
        src, tgt = state.nodes[action.source_id], state.nodes[action.target_id]
        reason = _incoherent_attack(src.owner, src.role, tgt.owner, tgt.role,
                                    action.edge_type)
        if reason:
            return False, reason
        reason = _incoherent_support(src.role, tgt.role, action.edge_type)
        if reason:
            return False, reason
        return True, ""

    return False, f"unknown action type: {type(action).__name__}"


def _check_introduce(state: RoundState, action: Introduce) -> Tuple[bool, str]:
    """Well-formedness + target-existence (rules 3-4) plus the attack-incoherence rules
    (rule 6) for an ATTACHING introduce."""
    if action.role not in ROLES:
        return False, f"role {action.role!r} not in vocabulary {sorted(ROLES)}"
    if action.target == NEW:
        if action.edge_type is not None:
            return False, "NEW node must not carry an edge_type"
        # Floating-root restriction (action_schema_spec §introduce): only an Advocacy
        # or a Framework may be introduced as a floating root, so every connected
        # component contains an Advocacy or a Framework -- the two kinds the judge
        # already treats as chain roots (judge_spec §2, rule 4). Every other role must
        # attach to an existing node at creation.
        if action.role not in ROOT_ELIGIBLE_ROLES:
            return False, (
                f"floating-root restriction: role {action.role!r} may not be introduced "
                f"as a NEW root (only {sorted(ROOT_ELIGIBLE_ROLES)} may root a component)"
            )
        return True, ""
    # attaching to an existing node via a relationship edge
    if action.target not in state.nodes:
        return False, f"target node {action.target!r} does not exist"
    if action.edge_type not in ATTACH_EDGE_TYPES:
        return False, (f"edge_type {action.edge_type!r} not in "
                       f"{sorted(ATTACH_EDGE_TYPES)} for an attaching introduce")
    # Rule 6: the new node is owned by the acting side (state.current_side) with the
    # declared role; the target is an existing node. Reject a same-side attack or an
    # offense at a non-polarity node.
    tgt = state.nodes[action.target]
    reason = _incoherent_attack(state.current_side, action.role, tgt.owner, tgt.role,
                                action.edge_type)
    if reason:
        return False, reason
    reason = _incoherent_support(action.role, tgt.role, action.edge_type)
    if reason:
        return False, reason
    return True, ""


def _incoherent_attack(a_side: str, a_role: str, b_side: str, b_role: str,
                       edge_type: Optional[str]) -> Optional[str]:
    """Rule 6 for an ATTACK edge between endpoint A (side `a_side`, role `a_role`) and
    endpoint B (side `b_side`, role `b_role`). Returns a rejection reason, or None if the
    edge is coherent (or not an attack). Two structurally-incoherent cases:

      - SAME-SIDE ATTACK: an attack edge between two same-side nodes can never enter any
        target's attacker set (mirrors the judge's `same-side attack (incoherent)`).
      - OFFENSE AT A NON-POLARITY NODE: an `offensive_attack` where EITHER endpoint is
        not offense-bearing (Link/Impact) has no polarity to flip (judge §3.4).

    Direction-agnostic in `edge_type`: a `defensive_attack` is checked only for the
    same-side case (it lowers magnitude and never flips, so it stays coherent against a
    non-polarity node); only `offensive_attack` is guarded for polarity -- matching the
    judge's own asymmetry in `_classify_attacks`."""
    if edge_type not in ATTACK_EDGE_TYPES:
        return None
    # An Advocacy may be neither the SOURCE nor the TARGET of any attack edge
    # (defensive or offensive) -- you support or outweigh a proposal, you never attack
    # it or attack FROM it (judge_spec §2). Independent of side and polarity; checked
    # first because it is a hard structural rule on either endpoint.
    if a_role == "advocacy" or b_role == "advocacy":
        return ("advocacy in attack edge: an Advocacy may be neither the source nor the "
                "target of an attack edge (defensive or offensive) (judge_spec §2)")
    if a_side == b_side:
        return "same-side attack (incoherent): attack edge between two same-side nodes"
    if edge_type == "offensive_attack" and not (
            a_role in OFFENSE_BEARING_ROLES and b_role in OFFENSE_BEARING_ROLES):
        return ("offense at a non-polarity node: offensive_attack requires "
                "offense-bearing (Link/Impact) endpoints (§3.4)")
    return None


def _incoherent_support(a_role: str, b_role: str, edge_type: Optional[str]) -> Optional[str]:
    """Rule (Advocacy attachment): an Advocacy may carry `support` edges ONLY to `Link`
    nodes. A Support edge with an Advocacy at either end whose OTHER endpoint is not a
    Link is structurally incoherent (an Advocacy's premises route through a Link, never a
    bare Uniqueness/Impact/BallotDirective/Advocacy). Direction-agnostic (support edges
    are undirected to the judge). Only `support` edges are checked; attack edges are the
    `_incoherent_attack` concern. Returns a rejection reason, or None if coherent."""
    if edge_type != "support":
        return None
    if a_role == "advocacy" and b_role != "link":
        return ("advocacy support-attaches only to Link: an Advocacy's Support edge must "
                f"reach a Link (got role {b_role!r})")
    if b_role == "advocacy" and a_role != "link":
        return ("advocacy support-attaches only to Link: an Advocacy's Support edge must "
                f"reach a Link (got role {a_role!r})")
    return None


def _duplicate_edge(state: RoundState, source: str, target: str,
                    edge_type: str) -> bool:
    """True iff an edge with the same source, target, AND edge_type already exists -- a
    redundant `connect` that would add no structure. O(edges) scan."""
    return any(e.source == source and e.target == target and e.edge_type == edge_type
               for e in state.edges)


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


# --- inert-action classification (REWARD ONLY -- NOT legality) -----------------
#
# `is_inert` is a SIBLING of `check_legality`, deliberately NOT part of it. It now
# classifies the ONE CONTEXT-DEPENDENT inert class that stays legal: a no-op re-extend
# (extend/concede on a node already carried this speech). Extending an UNCARRIED node is
# a real, often-correct move -- only this specific state makes it inert -- so it is not
# masked; it is priced via COST (a full slot; see state.action_cost) and this predicate
# is used only for reward-side/diagnostic bookkeeping. The reward penalty coefficient is
# currently 0.0 (dormant backstop, rl_training_spec §Reward), so at present this predicate
# only feeds diagnostic counters.
#
# The three STRUCTURALLY-INCOHERENT classes it used to classify -- same-side attack,
# offense-at-non-polarity, redundant connect -- are now ILLEGAL (`check_legality` rule 6),
# never sampled, so they are no longer inert classes here. DO NOT call this from
# `check_legality`; DO NOT let it gate a step.

# Inert-class tag (the sole remaining reward-side class; "" when not inert).
INERT_NOOP_REEXTEND = "noop_reextend"


def is_inert(state: RoundState, action) -> Tuple[bool, str]:
    """Return (inert, kind) for the one context-dependent inert class this predicate
    still governs: a NO-OP RE-EXTEND -- an `extend`/`concede` on a node already carried
    THIS speech, so the liveness stamp is an idempotent set-add that changes nothing. The
    introduction speech counts as a carry, so re-extending a node in the speech it was
    introduced is a no-op too. Returns (True, INERT_NOOP_REEXTEND) in that case, else
    (False, "").

    Reward-only / diagnostic; never consulted for legality (the structurally-incoherent
    classes are handled in `check_legality`). Must run at STEP TIME against the PRE-apply
    state: a no-op re-extend leaves no graph trace, so it is unrecoverable afterward.
    Shares the `already_carried_this_speech` predicate with `state.action_cost`, which
    prices the same case as a full slot."""
    if isinstance(action, (Extend, Concede)):
        if state.already_carried_this_speech(action.node_id):
            return True, INERT_NOOP_REEXTEND
    return False, ""
