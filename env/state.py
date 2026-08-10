"""Round state for the CDAF RL environment: the argument graph under construction
plus speech-sequence position.

State is agent-agnostic and holds ONLY structural facts an action produces -- no
accrual, no verdict, no judge state. `to_round()` materializes the accumulated
state into a `model.Round` (the exact object the judge scores) at the termination
step; it is the single place liveness status and edge subtypes are realized.

Liveness stamping (Phase-1 ruling): a node's liveness record is the set of speeches
it was carried through (introduction + every extend/concede on it). The per-speech
contested/conceded STATUS is DERIVED STRUCTURALLY at materialization, mirroring
`model/convert.py`'s predicate (the earlier-speech endpoint of a cross-side attack
is the contested target) -- never taken from which verb the agent used. This
preserves fixture/env verdict equivalence: the same graph scores the same whoever
built it. Status is computed at `to_round()` (end of construction), not when an
extend fires, because later actions in the same speech can change what is true.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from model import (
    Round, SCHEMA_VERSION, Comparison, Weighing,
    SPEECH_ORDER, SPEECH_SIDE, speech_index,
)
from model.nodes import CONTESTED, CONCEDED

from .actions import (
    ROLE_TO_NODE_CLASS, EDGE_TYPE_TO_CLASS, SPEECH_BUDGET, EXTEND_COST_K, NEW,
    Introduce, Extend, Concede, Weigh, Connect, EndSpeech,
)

_ATTACK_EDGE_TYPES = frozenset({"defensive_attack", "offensive_attack"})


@dataclass
class NodeRecord:
    """One node in the graph under construction. `carried` is the set of speeches
    the node was live through (liveness KEYS); status is derived at materialization.
    Weigh fields are populated only for Weighing nodes."""
    id: str
    content: str
    owner: str                              # AFF or NEG (the introducing side)
    introduction_speech: str
    role: str                               # agent-declared; ∈ ROLES, or "weighing"
    carried: set = field(default_factory=set)
    # Weighing-only:
    weigh_a: Optional[str] = None
    weigh_b: Optional[str] = None
    favors: Optional[str] = None


@dataclass
class EdgeRecord:
    id: str
    source: str
    target: str
    edge_type: str                          # support / defensive_attack / offensive_attack / comparison


@dataclass
class RoundState:
    """The mutable graph + sequence position. Construction is append-only within a
    speech; `advance_speech` moves to the next slot. Nothing here is ever scored --
    scoring happens once, on `to_round()`, at termination."""
    nodes: Dict[str, NodeRecord] = field(default_factory=dict)
    edges: List[EdgeRecord] = field(default_factory=list)
    slot_index: int = 0                     # index into SPEECH_ORDER
    moves_used: int = 0
    extends_this_speech: int = 0            # count of extend/concede actions this speech
                                            # (drives the batched marginal cost; reset per slot)
    _seq: int = 0                           # monotonic id counter
    # Inert-action counters (REWARD ONLY -- see legal_actions.is_inert). Accumulate over
    # the WHOLE episode (never reset per speech), incremented at step time BEFORE apply.
    # `inert_by_side` drives the terminal per-side penalty; `inert_by_kind` (flat, both
    # sides) and `inert_by_side_kind` (side -> kind -> count) are diagnostics only -- the
    # side x kind cross is what surfaces e.g. NEG's no-op-re-extend rate (Run 1: 86%).
    inert_by_side: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    inert_by_kind: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    inert_by_side_kind: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int)))

    # --- ids ------------------------------------------------------------------
    def _new_node_id(self) -> str:
        self._seq += 1
        return f"n{self._seq}"

    def _new_edge_id(self) -> str:
        self._seq += 1
        return f"e{self._seq}"

    # --- sequence position ----------------------------------------------------
    @property
    def current_slot(self) -> Optional[str]:
        """The current speech, or None once the round has terminated (past 2AR)."""
        if self.slot_index >= len(SPEECH_ORDER):
            return None
        return SPEECH_ORDER[self.slot_index]

    @property
    def current_side(self) -> Optional[str]:
        slot = self.current_slot
        return SPEECH_SIDE.get(slot) if slot else None

    @property
    def slot_budget(self) -> int:
        slot = self.current_slot
        return SPEECH_BUDGET[slot] if slot else 0

    @property
    def remaining_budget(self) -> int:
        return max(0, self.slot_budget - self.moves_used)

    @property
    def terminated(self) -> bool:
        return self.current_slot is None

    def advance_speech(self) -> None:
        """Move to the next speech slot, resetting the per-speech move counter and the
        extend/concede batching counter."""
        self.slot_index += 1
        self.moves_used = 0
        self.extends_this_speech = 0

    def record_inert(self, side: str, kind: str) -> None:
        """Tally one inert action for `side` (drives the terminal per-side penalty) and
        for `kind` (diagnostics only). Called at step time, BEFORE `apply`, so `side` is
        the acting side of the move being taken. Reward-only bookkeeping -- never affects
        legality, cost, or the materialized graph."""
        self.inert_by_side[side] += 1
        self.inert_by_kind[kind] += 1
        self.inert_by_side_kind[side][kind] += 1

    # --- mutations (structural only; callers enforce legality) ----------------
    def add_node(self, content: str, owner: str, role: str) -> str:
        nid = self._new_node_id()
        slot = self.current_slot
        self.nodes[nid] = NodeRecord(
            id=nid, content=content, owner=owner, introduction_speech=slot,
            role=role, carried={slot},
        )
        return nid

    def add_edge(self, source: str, target: str, edge_type: str) -> str:
        eid = self._new_edge_id()
        self.edges.append(EdgeRecord(id=eid, source=source, target=target,
                                     edge_type=edge_type))
        return eid

    def add_weigh(self, node_a: str, node_b: str, favors: str, owner: str) -> str:
        """Materialize a weigh as a Weighing node + two Comparison edges (source =
        the weighing, per resolve.weigh_pair's load-bearing direction)."""
        wid = self._new_node_id()
        slot = self.current_slot
        self.nodes[wid] = NodeRecord(
            id=wid, content="", owner=owner, introduction_speech=slot,
            role="weighing", carried={slot},
            weigh_a=node_a, weigh_b=node_b, favors=favors,
        )
        self.add_edge(wid, node_a, "comparison")
        self.add_edge(wid, node_b, "comparison")
        return wid

    def already_carried_this_speech(self, node_id: str) -> bool:
        """True iff `node_id` exists and is ALREADY stamped carried for the CURRENT speech
        -- so an extend/concede on it would be a NO-OP RE-EXTEND (idempotent set-add). The
        single source of truth for the no-op predicate, shared by `action_cost` (which
        prices it as a full slot) and `legal_actions.is_inert` (reward/diagnostic side)."""
        rec = self.nodes.get(node_id)
        return rec is not None and self.current_slot in rec.carried

    def carry(self, node_id: str) -> None:
        """Stamp the node as carried through the current speech (extend/concede). ATOMIC:
        stamps ONLY this node -- no path-walking, no propagation to the rest of a chain
        (environment_shell_spec §Liveness stamping). An agent may extend a link while
        deliberately not extending its impact, letting it die."""
        self.nodes[node_id].carried.add(self.current_slot)

    # --- action application ---------------------------------------------------
    def apply(self, action) -> None:
        """Apply one action's STRUCTURAL effect. Legality is the caller's
        responsibility (see legal_actions.check_legality); this only mutates. Every
        non-EndSpeech action consumes one move; EndSpeech advances the speech.

        Shared by `environment.step` and the generator's Fence-A prospective check
        (which applies a candidate to a copy), so the action->primitive mapping lives
        in exactly one place."""
        side = self.current_side
        if isinstance(action, EndSpeech):
            self.advance_speech()
            return

        cost = 1                                    # introduce / weigh / connect
        if isinstance(action, Introduce):
            if action.target == NEW:
                self.add_node(action.content, side, action.role)
            else:
                nid = self.add_node(action.content, side, action.role)
                self.add_edge(nid, action.target, action.edge_type)
        elif isinstance(action, (Extend, Concede)):
            # Atomic: stamp only the named node. Cost is the MARGINAL cost of this
            # carriage against the speech-wide batch counter (computed BEFORE the
            # counter is bumped): 1 on the 1st, (K+1)-th, (2K+1)-th ... extend of the
            # speech, 0 otherwise -- so N extends cost ceil(N / EXTEND_COST_K) total,
            # the discount scoped to the whole speech rather than to any chain.
            cost = action_cost(self, action)
            self.carry(action.node_id)
            self.extends_this_speech += 1
        elif isinstance(action, Weigh):
            self.add_weigh(action.node_a, action.node_b, action.favors, side)
        elif isinstance(action, Connect):
            self.add_edge(action.source_id, action.target_id, action.edge_type)
        else:
            raise TypeError(f"unknown action type: {type(action).__name__}")

        self.moves_used += cost
        # Budget exhaustion ends the speech automatically (turn advance, §step).
        if self.remaining_budget == 0:
            self.advance_speech()

    # --- materialization ------------------------------------------------------
    def to_round(self) -> Round:
        """Materialize the accumulated state into a `model.Round` -- the object the
        judge scores. Realizes node classes from declared roles, edge classes from
        declared edge types, and per-speech liveness STATUS structurally (below)."""
        contested = self._contested_speeches()
        elements = []
        for rec in self.nodes.values():
            liveness = {sp: (CONTESTED if sp in contested.get(rec.id, ()) else CONCEDED)
                        for sp in sorted(rec.carried, key=speech_index)}
            common = dict(id=rec.id, label=rec.content, side=rec.owner,
                          speech=rec.introduction_speech, liveness=liveness)
            if rec.role == "weighing":
                # Agent-constructed rounds always carry explicit favors (§6.5); pass
                # it through so the judge reads the agent's preference, not a default.
                elements.append(Weighing(favors=rec.favors, **common))
            else:
                elements.append(ROLE_TO_NODE_CLASS[rec.role](**common))
        for e in self.edges:
            if e.edge_type == "comparison":
                elements.append(Comparison(id=e.id, source=e.source, target=e.target))
            else:
                elements.append(EDGE_TYPE_TO_CLASS[e.edge_type](
                    id=e.id, source=e.source, target=e.target))
        return Round(elements=elements, version=SCHEMA_VERSION)

    def _contested_speeches(self) -> Dict[str, set]:
        """Per-node set of speeches whose liveness status is CONTESTED, mirroring
        `convert._contested_targets` lifted to native v2 (no per-speech duplicates):
        a cross-side attack from opponent node `m` (speech t) contests the TARGET's
        instance that was live when `m` spoke = the latest carried speech with index
        < index(t). Orientation is by speech recency (§2.2): the later-speech node is
        the attacker, the earlier the contested target. Every other carried speech is
        conceded. Attack SUBTYPE is irrelevant to contestation (both defensive and
        offensive attacks answer the target)."""
        by_id = self.nodes
        out: Dict[str, set] = {}
        for e in self.edges:
            if e.edge_type not in _ATTACK_EDGE_TYPES:
                continue
            a, b = by_id.get(e.source), by_id.get(e.target)
            if a is None or b is None or a.owner == b.owner:
                continue                    # same-side "attack" is incoherent (§2.2)
            ia, ib = speech_index(a.introduction_speech), speech_index(b.introduction_speech)
            if ia == ib:
                continue                    # same speech -> same side; cannot clash
            target, attacker = (a, b) if ia < ib else (b, a)
            t = speech_index(attacker.introduction_speech)
            cand = [sp for sp in target.carried if speech_index(sp) < t]
            if cand:
                sp = max(cand, key=speech_index)
                out.setdefault(target.id, set()).add(sp)
        return out


# --- action cost (speech-budget slots) ----------------------------------------

def action_cost(state: RoundState, action) -> int:
    """The number of speech-budget slots `action` consumes in `state`.

    `end_speech` costs 0 (it ends the turn); `introduce`/`weigh`/`connect` cost 1;
    a DISTINCT `extend`/`concede` carriage costs the MARGINAL of a speech-wide
    `ceil(count / K)` batch,

        marginal = ceil((count + 1) / K) - ceil(count / K)

    where `count` = `state.extends_this_speech` (extends already taken this speech) and
    K = `EXTEND_COST_K`. This is 1 on the 1st, (K+1)-th, (2K+1)-th ... extend of the
    speech and 0 otherwise, so N distinct carriages over a speech cost `ceil(N / K)`
    total -- the "1 slot per K carriages" discount, scoped to the whole speech.

    NO-OP RE-EXTEND EXCEPTION: an extend/concede on a node ALREADY carried this speech
    changes nothing, so it does NOT get the batch discount -- it costs a FULL SLOT (1)
    regardless of `count`, like any wasted move (action_schema_spec §extend). It still
    increments `extends_this_speech` in `apply`. Only distinct carriages earn the batch
    rate. Detection uses `state.already_carried_this_speech` -- the same predicate
    `legal_actions.is_inert` uses -- computed on the PRE-apply state (before `carry`),
    so a genuine no-op (target already carried) is priced at 1 and a distinct carriage
    at the batch marginal.

    Depends only on current state, so the cost of the NEXT carriage is a simple lookup
    -- no knowledge of future actions is needed. Single source of truth: both
    `check_legality`'s affordability gate and `apply`'s `moves_used` increment read
    this, so they cannot drift."""
    if isinstance(action, EndSpeech):
        return 0
    if isinstance(action, (Extend, Concede)):
        if state.already_carried_this_speech(action.node_id):
            return 1                         # no-op re-extend: full slot, no batch discount
        c = state.extends_this_speech
        return math.ceil((c + 1) / EXTEND_COST_K) - math.ceil(c / EXTEND_COST_K)
    return 1                                 # introduce / weigh / connect
