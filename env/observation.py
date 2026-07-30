"""Observation contract for the CDAF environment.

The observation is raw graph structure plus MONOTONIC SETTLED FACTS -- facts that,
once determined, can never be reversed by a later speech (environment_shell_spec
§Observation contract). NO judge pass is executed to produce these; they are derived
from graph structure and speech order alone. Pure, side-effect-free structural
primitives (`response_window`, `side_speeches`) are IMPORTED from the judge rather
than reimplemented, so window/side logic cannot drift (spec §Reuse of judge
structural primitives). Nothing that produces accrual, magnitude, liveness-scoring,
or verdict state is called.

INCLUDED (all monotonic):
  - full graph structure (nodes, edges, ownership, introduction speech, in-degree)
  - closed-window drops   (window already passed with no opposing clash -> settled)
  - permanent extension failure (a spine node already missed an own-side past
    speech -> that chain is dead; no later speech revives it)
  - reachability          (which nodes route to an impact vs are orphaned)
  - sequence state        (current slot, side, moves used, remaining budget)

EXCLUDED (provisional -- must NOT appear): extension ELIGIBILITY for still-live
candidates, DF-QuAD magnitudes / any accrual output, any running or projected
verdict, any lookahead. The environment does not predict; agent-side planning lives
on the agent.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List

from model import speech_index, SPEECH_ORDER
from judge.passes import response_window, side_speeches

from .state import RoundState

# Nodes the judge excludes from drop detection (structural / sub-debate nodes).
_NO_DROP_ROLES = frozenset({"ballot_directive", "weighing", "framework"})
# Spine roles whose own-side coverage lapse permanently kills their chain (§6).
_SPINE_ROLES = frozenset({"uniqueness", "link", "impact", "advocacy"})
_ATTACK_EDGE_TYPES = frozenset({"defensive_attack", "offensive_attack"})


def observe(state: RoundState) -> Dict:
    """Build the observation dict from current state. Every field is a monotonic
    settled fact or raw structure; none is provisional."""
    return {
        "graph": _graph(state),
        "closed_window_drops": _closed_window_drops(state),
        "permanent_extension_failures": _permanent_extension_failures(state),
        "reachability": _reachability(state),
        "sequence": _sequence(state),
    }


def _graph(state: RoundState) -> Dict:
    in_degree = defaultdict(int)
    for e in state.edges:
        in_degree[e.target] += 1
    nodes = [{
        "id": n.id,
        "role": n.role,
        "owner": n.owner,
        "introduction_speech": n.introduction_speech,
        "in_degree": in_degree.get(n.id, 0),
        "carried_speeches": sorted(n.carried, key=speech_index),
    } for n in state.nodes.values()]
    edges = [{"source": e.source, "target": e.target, "edge_type": e.edge_type}
             for e in state.edges]
    return {"nodes": nodes, "edges": edges}


def _closed_window_drops(state: RoundState) -> List[str]:
    """Node ids whose response window has ALREADY passed with no opposing clash --
    a permanently settled drop (§4). The window (next opposing speech, §4) is passed
    iff its index is strictly less than the current slot index; during the window
    speech itself the opponent may still clash, so it is NOT yet settled. Monotonic:
    a past window can never gain a clash."""
    now = state.slot_index
    incident_attack_speeches = _incident_opposing_attack_speeches(state)
    dropped = []
    for n in state.nodes.values():
        if n.role in _NO_DROP_ROLES:
            continue
        win = response_window(n.introduction_speech, n.owner)
        if win is None or speech_index(win) >= now:
            continue                        # no window, or window not yet settled
        if win not in incident_attack_speeches.get(n.id, ()):
            dropped.append(n.id)
    return dropped


def _permanent_extension_failures(state: RoundState) -> List[str]:
    """Spine node ids whose OWN-SIDE extension has definitively lapsed: some own-side
    speech at or after introduction has already passed (index < current slot) with no
    carriage. Monotonic -- a past speech can never be retroactively carried, so the
    ordinary chain through this node is dead (§6, 'no later speech revives it').

    (Side-agnostic union liveness can still keep a node alive for a TURNED chain; that
    is a judge outcome, not a settled structural fact, so it is deliberately not
    inferred here. This flag is the own-side coverage lapse the spec names.)"""
    now = state.slot_index
    failed = []
    for n in state.nodes.values():
        if n.role not in _SPINE_ROLES:
            continue
        intro_idx = speech_index(n.introduction_speech)
        for s in side_speeches(n.owner):
            si = speech_index(s)
            if intro_idx <= si < now and s not in n.carried:
                failed.append(n.id)
                break
    return failed


def _reachability(state: RoundState) -> Dict[str, bool]:
    """Per-node: does it route to an impact over UNDIRECTED connectivity? Nodes not
    connected to any impact-role node are orphaned. Purely structural (edge arrows
    are not load-bearing, §2.2)."""
    adj = defaultdict(list)
    for e in state.edges:
        adj[e.source].append(e.target)
        adj[e.target].append(e.source)
    impacts = [nid for nid, n in state.nodes.items() if n.role == "impact"]
    seen = set(impacts)
    q = deque(impacts)
    while q:
        cur = q.popleft()
        for nbr in adj[cur]:
            if nbr not in seen:
                seen.add(nbr)
                q.append(nbr)
    return {nid: (nid in seen) for nid in state.nodes}


def _sequence(state: RoundState) -> Dict:
    return {
        "slot": state.current_slot,
        "slot_index": state.slot_index,
        "side": state.current_side,
        "moves_used": state.moves_used,
        "remaining_budget": state.remaining_budget,
        "terminated": state.terminated,
    }


def _incident_opposing_attack_speeches(state: RoundState) -> Dict[str, set]:
    """node id -> set of speeches at which an OPPOSING-side attack is incident to it
    (either drawn direction). Used to tell 'answered' from 'dropped' at the window."""
    by_id = state.nodes
    out: Dict[str, set] = defaultdict(set)
    for e in state.edges:
        if e.edge_type not in _ATTACK_EDGE_TYPES:
            continue
        a, b = by_id.get(e.source), by_id.get(e.target)
        if a is None or b is None or a.owner == b.owner:
            continue
        out[a.id].add(b.introduction_speech)
        out[b.id].add(a.introduction_speech)
    return out
