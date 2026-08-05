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
from typing import Dict, List, Optional, Set

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
    _seq: int = 0                           # monotonic id counter

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
        """Move to the next speech slot, resetting the per-speech move counter."""
        self.slot_index += 1
        self.moves_used = 0

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

    def carry(self, node_id: str) -> None:
        """Stamp the node as carried through the current speech (extend/concede)."""
        self.nodes[node_id].carried.add(self.current_slot)

    # --- chain-level extension walk -------------------------------------------
    def extend_path(self, node_id: str) -> Set[str]:
        """The root-to-impact walk an `extend`/`concede` on `node_id` stamps
        (environment_shell_spec §Liveness stamping): every spine node (advocacy/link/
        impact) on a root->impact Support path THROUGH `node_id`, plus satellite
        Uniqueness attached to those nodes. Direction-agnostic (undirected Support,
        matching the judge's per-path liveness, judge_spec §3.3.1a).

        Divergence (v11): a shared trunk fans out to several terminal impacts. Naming a
        node on ONE branch stamps only that branch's root->impact path(s) (the trunk is
        included, never other branches) -- so each branch is walked and priced
        independently (N branches -> N `extend`s, the trunk re-stamped and re-paid each
        time, never banked). Naming a shared trunk/root node stamps every path through
        it (a bulk extend, priced by its total size -- cost scales with what is stamped).

        Off-spine / orphan / impact-less target (Open Question 2): the node is still a
        legal target; the walk stamps whatever same-side spine is reachable from it
        (at least the node itself)."""
        nodes = self.nodes
        if node_id not in nodes:
            return set()
        side = nodes[node_id].owner

        adj: Dict[str, List[str]] = defaultdict(list)
        for e in self.edges:
            if e.edge_type != "support":
                continue
            a, b = nodes.get(e.source), nodes.get(e.target)
            if a is None or b is None or a.owner != side or b.owner != side:
                continue
            adj[e.source].append(e.target)
            adj[e.target].append(e.source)

        # same-side Support component of the target
        comp: Set[str] = set()
        stack = [node_id]
        while stack:
            u = stack.pop()
            if u in comp:
                continue
            comp.add(u)
            for v in adj[u]:
                if v not in comp:
                    stack.append(v)

        role = {n: nodes[n].role for n in comp}
        impacts = [n for n in comp if role[n] == "impact"]
        # roots: advocacies, else non-impact spine, else the impacts themselves
        # (mirrors the judge's `roots` selection in passes._build_chains).
        roots = ([n for n in comp if role[n] == "advocacy"]
                 or [n for n in comp if role[n] != "impact"]
                 or impacts)
        root_set = set(roots)

        # Enumerate the simple Support paths from an impact to a root that pass through
        # `node_id` (small graphs; simple-path DFS). `path[0]` is the terminal impact.
        candidates: List[List[str]] = []

        def dfs(cur, path, seen):
            if cur in root_set and len(path) > 1:
                if node_id in path:
                    candidates.append(list(path))
                return
            for v in adj[cur]:
                if v not in seen:
                    seen.add(v); path.append(v)
                    dfs(v, path, seen)
                    path.pop(); seen.discard(v)

        for imp in impacts:
            dfs(imp, [imp], {imp})

        if candidates:
            # Stamp EXACTLY ONE root->impact path -- the single path this node's walk
            # covers -- so one extend can never cover more than one terminal impact and
            # there is no pooling / no trunk discount across divergent branches (naming
            # a shared trunk/root picks one branch, not all; the other branch still costs
            # its own separate extend, re-paying the trunk). Deterministic tiebreak:
            # smallest (terminal-impact id, path) so the choice is stable across runs.
            stamped: Set[str] = set(min(candidates, key=lambda p: (p[0], tuple(p))))
        else:
            # off-spine / satellite / orphan / impact-less target: not on any
            # root->impact path, so its walk stamps only itself (OQ2 -- still a legal
            # target, and deliberately minimal so naming an off-spine node cannot
            # bulk-stamp a whole component).
            stamped = {node_id}

        # satellite Uniqueness hanging off any stamped spine node (r32's u2/u7/u3). A
        # Uniqueness is never a terminal impact, so this cannot make one action cover a
        # second branch's impact.
        for n in list(stamped):
            for v in adj[n]:
                if role.get(v) == "uniqueness":
                    stamped.add(v)
        return stamped

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
            # Chain-level: stamp the whole root-to-impact walk, priced by its length
            # (ceil(len / EXTEND_COST_K)). Idempotent -- re-stamping a node already
            # carried this speech changes nothing, but the action still costs full price
            # (the trunk is never banked across divergent branches).
            path = self.extend_path(action.node_id)
            for nid in path:
                self.carry(nid)
            cost = _extend_slot_cost(len(path))
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

def _extend_slot_cost(path_len: int) -> int:
    """Slots an extend/concede over a `path_len`-node walk costs: ceil(len / K), never
    below 1. K is `EXTEND_COST_K` (default 4). Worked examples (K=4): 2->1, 6->2, 8->2,
    9->3."""
    return max(1, math.ceil(path_len / EXTEND_COST_K))


def action_cost(state: RoundState, action) -> int:
    """The number of speech-budget slots `action` consumes in `state`.

    `end_speech` costs 0 (it ends the turn); `introduce`/`weigh`/`connect` cost 1;
    `extend`/`concede` cost ceil(path_length / EXTEND_COST_K) over the root-to-impact
    walk they stamp (`RoundState.extend_path`). This is the single source of truth for
    cost -- both `check_legality`'s affordability gate and `apply`'s `moves_used`
    increment read the same formula, so they cannot drift."""
    if isinstance(action, EndSpeech):
        return 0
    if isinstance(action, (Extend, Concede)):
        if action.node_id not in state.nodes:
            return 1                         # nonexistent target: rejected on existence first
        return _extend_slot_cost(len(state.extend_path(action.node_id)))
    return 1                                 # introduce / weigh / connect
