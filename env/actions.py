"""Action space for the CDAF RL environment (Phase 0 schema, typed).

Five action types, agent-agnostic (LLM-in-loop or trained policy). Each is a thin
dataclass carrying only the STRUCTURAL claim it makes; free text (`content`,
`justification`) is recorded but never read by the judge. Whose turn it is and how
much budget remains are environment state, not action fields -- an action does not
name its side; the current speech slot fixes that (`environment.py`).

Vocabulary is pinned here so the state schema, the legal-action generator, and the
Round materializer share one source of truth (they must not redefine it):

  ROLE_TO_NODE_CLASS   role string -> model node class (agent-declared, §Role).
  EDGE_TYPE_TO_CLASS   relationship edge_type -> model edge class.
  NEW                  target sentinel for a fresh node.
  SPEECH_BUDGET        per-slot move budget (first-iteration defaults, tunable).

The shared-node case is served by ordinary structural targeting, not a distinct
merge mode: a `support` edge into an existing node gives genuine cross-side
in-degree, and the judge's per-node σ propagates an attack on that node to every
dependent chain (Investigation D; the r32 shared-node mechanism). Identity merge
was therefore vestigial in V1 and has been removed.

Weighing is intentionally absent from ROLE_TO_NODE_CLASS: a Weighing node is
produced by the `weigh` action, never by `introduce`. BallotDirective IS
introducible (the discovery root the ballot needs), declared as role
"ballot_directive" -- the concrete "non-spine role" the schema lists by example.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

from model import (
    Uniqueness, Link, Impact, Advocacy, Framework, BallotDirective,
    Support, DefensiveAttack, OffensiveAttack, SPEECH_ORDER,
)

# Target sentinel: introduce a fresh node with no existing identity.
NEW = "NEW"

# Agent-declared role -> model node class. Role is a strategic assertion, never
# derived from graph position (§Role declaration).
ROLE_TO_NODE_CLASS = {
    "uniqueness": Uniqueness,
    "link": Link,
    "impact": Impact,
    "advocacy": Advocacy,
    "framework": Framework,
    "ballot_directive": BallotDirective,
}
ROLES = frozenset(ROLE_TO_NODE_CLASS)

# Relationship edge_type -> model edge class. Turn status is DECLARED, not derived:
# offensive_attack is a turn, defensive_attack is a takeout (action_schema_spec §Edge
# type vocabulary). These map directly onto the model's edge classes at
# materialization, nothing left to infer.
EDGE_TYPE_TO_CLASS = {
    "support": Support,
    "defensive_attack": DefensiveAttack,
    "offensive_attack": OffensiveAttack,
}
RELATIONSHIP_EDGE_TYPES = frozenset(EDGE_TYPE_TO_CLASS)
# Every legal `edge_type` value on an attaching introduce.
ATTACH_EDGE_TYPES = RELATIONSHIP_EDGE_TYPES
# Attack edge_types (defensive/offensive), for the inert-attack classifier.
ATTACK_EDGE_TYPES = frozenset({"defensive_attack", "offensive_attack"})

# Roles that carry polarity -- the ONLY endpoints an offensive_attack can flip
# (§3.4). Mirrors the judge's OFFENSE_BEARING_KINDS ({"link", "impact"},
# judge/passes.py) so the env's inert-offense classifier agrees with the judge's
# InertAttack classification exactly. An offensive_attack where EITHER endpoint is
# not offense-bearing has no polarity to flip and is inert at creation.
OFFENSE_BEARING_ROLES = frozenset({"link", "impact"})

# Roles that may ROOT a connected component -- i.e. may be introduced as a floating
# `NEW` node with no attaching edge (action_schema_spec §introduce → Floating-root
# restriction). Every OTHER role must attach to an existing node at creation, so that
# every connected component contains an Advocacy or a Framework. Advocacy and Framework
# are exactly the two kinds the judge already treats as chain roots (a NEG offense
# chain roots at an AFF Advocacy OR its own Framework -- judge_spec §2, rule 4).
ROOT_ELIGIBLE_ROLES = frozenset({"advocacy", "framework"})

# Diagnostic isolation toggle (NOT a ruling): CDAF_ROOT_ELIGIBLE_ALL=1 makes EVERY role
# root-eligible, turning the floating-root restriction into a no-op for the whole process --
# used to measure whether that rule is implicated in a screen result, holding everything else
# fixed. Read once at import; each screen worker sets it (or not) in its own env. Default
# (unset) preserves the ruled Advocacy/Framework restriction.
if os.environ.get("CDAF_ROOT_ELIGIBLE_ALL") == "1":
    ROOT_ELIGIBLE_ROLES = ROLES

# Per-speech move budget (first-iteration defaults, tunable). Default sum = 52.
_SPEECH_BUDGET_DEFAULT = {
    "1AC": 8, "1NC": 8, "2AC": 8, "2NC/1NR": 13, "1AR": 5, "2NR": 5, "2AR": 5,
}

# Config hook (screen driver sets per run; default preserves current values so nothing
# changes unless overridden). Mirrors the CDAF_ROOT_ELIGIBLE_ALL precedent above:
# read ONCE at import, so a worker sets the env var in its own process before importing
# env. Two overrides:
#   CDAF_SPEECH_BUDGET  -- JSON object mapping SPEECH_ORDER slot -> positive int budget.
#                          May be PARTIAL (merges over the defaults, so B4's rebuttal-only
#                          bump is expressible) or full. Keys must be a subset of
#                          SPEECH_ORDER; values must be positive ints.
#   CDAF_EXTEND_COST_K  -- positive int; the extend/concede batch size K.
# Invalid overrides raise at import (loud, not silent) -- a screen must fail fast rather
# than train against a mis-parsed budget.
def _load_speech_budget() -> dict:
    budget = dict(_SPEECH_BUDGET_DEFAULT)
    raw = os.environ.get("CDAF_SPEECH_BUDGET")
    if raw:
        override = json.loads(raw)                       # raises on malformed JSON
        if not isinstance(override, dict):
            raise ValueError(f"CDAF_SPEECH_BUDGET must be a JSON object, got {type(override).__name__}")
        for slot, val in override.items():
            if slot not in _SPEECH_BUDGET_DEFAULT:
                raise ValueError(f"CDAF_SPEECH_BUDGET: unknown slot {slot!r} "
                                 f"(expected a subset of {sorted(_SPEECH_BUDGET_DEFAULT)})")
            if not isinstance(val, int) or isinstance(val, bool) or val < 1:
                raise ValueError(f"CDAF_SPEECH_BUDGET[{slot!r}] must be a positive int, got {val!r}")
        budget.update(override)
    return budget


def _load_extend_cost_k() -> int:
    raw = os.environ.get("CDAF_EXTEND_COST_K")
    if raw is None:
        return 4
    k = int(raw)                                         # raises on non-int
    if k < 1:
        raise ValueError(f"CDAF_EXTEND_COST_K must be a positive int, got {k}")
    return k


SPEECH_BUDGET = _load_speech_budget()
assert set(SPEECH_BUDGET) == set(SPEECH_ORDER)
TOTAL_BUDGET = sum(SPEECH_BUDGET.values())

# Extend/concede batch size (§Turn structure, action_schema_spec): an `extend`/
# `concede` stamps one node atomically; cost is the marginal of a speech-wide
# `ceil(count / EXTEND_COST_K)` batch (count = extends_this_speech), so K carriages
# cost one slot and keeping many nodes alive in the back half carries real budget
# pressure while ordinary spine carriage stays cheap. Named beside SPEECH_BUDGET and
# tunable on the same footing; first-iteration default 4, overridable via
# CDAF_EXTEND_COST_K (above).
EXTEND_COST_K = _load_extend_cost_k()


# --- action types -------------------------------------------------------------

@dataclass(frozen=True)
class Introduce:
    """Introduce a claim (§introduce). Covers two modes via (target, edge_type):

      - NEW node:        target = NEW,             edge_type = None
      - attach relation: target = <id>,            edge_type ∈ RELATIONSHIP_EDGE_TYPES

    Attaching to an existing node (any node, either side) is how a shared node is
    formed -- there is no separate merge mode. `role` is the agent-declared node role
    (∈ ROLES). `content` is free text, never read by the judge."""
    content: str
    role: str
    target: str = NEW                       # NEW or an existing node id
    edge_type: Optional[str] = None         # None for NEW; else an ATTACH_EDGE_TYPES value


@dataclass(frozen=True)
class Extend:
    """Mark an existing node carried through the current speech (liveness). Status
    (contested/conceded) is derived structurally at materialization, never from the
    verb."""
    node_id: str


@dataclass(frozen=True)
class Concede:
    """Grant an existing node this speech. Like Extend, it records carriage of the
    node through the current speech; status is derived structurally, not from the
    verb (so Extend and Concede have the same structural effect -- a liveness stamp
    -- and differ only as agent-facing intent / logging)."""
    node_id: str


@dataclass(frozen=True)
class Weigh:
    """Introduce a comparison between two existing nodes (§weigh). Materializes a
    Weighing node + two Comparison edges. `favors` points at node_a or node_b;
    `justification` is free text. NOTE: the current judge derives weigh preference
    from the Weighing node's SIDE, not from `favors` (resolve.preferred_node), so
    `favors` is recorded in state but has no structural channel the judge reads --
    a pre-existing Phase-0/judge divergence, flagged not resolved here. The env only
    materializes; it must not encode strategic preference or re-implement judge
    semantics."""
    node_a: str
    node_b: str
    favors: str                             # must equal node_a or node_b
    justification: str = ""


@dataclass(frozen=True)
class Connect:
    """Add a relationship edge between two nodes that ALREADY exist. Creates no node;
    costs one move like every other action (a cross-application costs speech time --
    free edges would break the budget economy). This is the only action that can add an
    edge between two pre-existing nodes, so it is what makes convergence (two paths onto
    a shared impact -- r32's shared uniqueness, the cross-side shared impact of Phase-0
    Option B) and other non-forest structure buildable at all; `introduce`, creating a
    node and its single edge, can only ever grow a forest. `edge_type` ∈
    RELATIONSHIP_EDGE_TYPES. Legality forbids self-loops and Support cycles."""
    source_id: str
    target_id: str
    edge_type: str


@dataclass(frozen=True)
class EndSpeech:
    """Terminate the current speech before exhausting its budget (§end_speech)."""
    pass


Action = (Introduce, Extend, Concede, Weigh, Connect, EndSpeech)
