"""Materiality / highlight pass for post-hoc round visualization.

Read-only. Reruns the UNMODIFIED six-pass judge (judge_spec §9) on deep-copied,
single-element-deleted copies of a finished round, and marks each node/edge
"material" when its removal would flip the ballot OR move the net-offense margin.
Also surfaces the single best chain per side that Φ_L reads (PBRS: "Φ reads only
the best chain"). Touches nothing upstream -- no legality, no tabula-rasa, no
judge internals; it only reruns the existing judge on modified COPIES.

Element-deletion semantics mirror the builder's delete-node / delete-edge exactly
(cdaf_app.py:1264-1288): deleting a node drops the node AND every edge incident to
it (source or target == id); deleting an edge drops that edge only. The builder
applies this to cytoscape element dicts; here the same rule runs over a deep-copied
`model.Round` so the judge can re-evaluate it.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from model.round import Round
from model.serialize import elements_from_round
from judge.judge import judge
from judge import trace as T
from judge.passes import (
    best_extended_chain_by_side,   # the shared Φ per-side selection (factored out of phi_maxdiff)
    node_accrual,
    resolve_chains,
    weighing_excluded,
)


# --- baseline judge readout ----------------------------------------------------

def _ballot_margin(rnd: Round) -> Tuple[str, float]:
    """Run the judge and return (winner, margin). The margin is the net offense N
    (Ballot trace record, judge.py:_ballot / §7): N = Σ AFF deltas − Σ NEG deltas
    over validated, non-excluded chains. N is the only continuous quantity the
    ballot turns on (AFF wins iff N > EPSILON and the structural gates hold), so it
    is the natural margin for a materiality delta. The judge returns (winner, trace)
    and never a scalar margin, so we read N off the single Ballot record here."""
    winner, trace = judge(rnd)
    ballot = next((r for r in trace if isinstance(r, T.Ballot)), None)
    margin = ballot.N if ballot is not None else 0.0
    return winner, margin


# --- single-element deletion (mirrors cdaf_app.py:1264-1288) --------------------

def _delete_node(rnd: Round, node_id: str) -> Round:
    """Deep-copy `rnd` and drop node `node_id` plus every incident edge (source or
    target == node_id). Same rule as the builder's node-delete (cdaf_app.py:1264-1267),
    applied to a model.Round rather than cytoscape element dicts."""
    out = copy.deepcopy(rnd)
    out.elements = [el for el in out.elements
                    if getattr(el, "id", None) != node_id
                    and getattr(el, "source", None) != node_id
                    and getattr(el, "target", None) != node_id]
    return out


def _delete_edge(rnd: Round, edge_id: str) -> Round:
    """Deep-copy `rnd` and drop edge `edge_id` only (cdaf_app.py:1286-1287)."""
    out = copy.deepcopy(rnd)
    out.elements = [el for el in out.elements if getattr(el, "id", None) != edge_id]
    return out


# --- materiality ---------------------------------------------------------------

@dataclass
class MaterialityResult:
    """Per-element materiality plus the baseline the deltas are measured against.

    `material` maps every node_id and edge_id in the round to whether deleting it
    changed the verdict (winner flip) or moved the margin beyond `epsilon`.
    """
    baseline_winner: str
    baseline_margin: float
    epsilon: float
    material: Dict[str, bool]   # node_id / edge_id -> is material


def compute_materiality(round_state: Round, epsilon: float = 0.0) -> MaterialityResult:
    """Mark every node/edge material iff deleting it flips the winner or moves the
    net-offense margin by more than `epsilon`.

    O(N) reruns of the six-pass judge, N = node count + edge count (one delete-and-
    rejudge per element, over deep copies -- the input round is never mutated).

    epsilon defaults to 0.0 (zero-tolerance) as a PLACEHOLDER: any margin change at
    all marks the element material. This is intended to be replaced by the
    weighing-inertness margin-change epsilon once that hardening work lands -- see the
    weighing-inertness / margin-change note tracked in docs/render_spec.md (RS27 edge
    invisibility + open questions O-series). TODO(epsilon): swap 0.0 for that ruled
    tolerance so an element whose only effect is a below-threshold, weighing-inert
    margin nudge is NOT flagged material.
    """
    base_winner, base_margin = _ballot_margin(round_state)
    material: Dict[str, bool] = {}
    for node in round_state.nodes:
        w, m = _ballot_margin(_delete_node(round_state, node.id))
        material[node.id] = (w != base_winner) or (abs(m - base_margin) > epsilon)
    for edge in round_state.edges:
        w, m = _ballot_margin(_delete_edge(round_state, edge.id))
        material[edge.id] = (w != base_winner) or (abs(m - base_margin) > epsilon)
    return MaterialityResult(baseline_winner=base_winner, baseline_margin=base_margin,
                             epsilon=epsilon, material=material)


# --- best path per side (Φ_L's per-side selection) -----------------------------

def best_path_per_side(round_state: Round) -> Dict[str, Optional[dict]]:
    """The single best chain per side that Φ_L reads for offense (PBRS note: Φ reads
    only the best chain per side). Reuses `judge.passes.best_extended_chain_by_side`
    over the SAME node_accrual -> resolve_chains -> weighing_excluded pipeline Φ uses
    (env/observation.py:potential), at the terminal horizon (as_of=None) since the
    round is finished. No new path-selection logic -- the selection lives in the judge
    package and is shared byte-for-byte with Φ_maxdiff.

    Returns {"AFF": path|None, "NEG": path|None} where a path is
    {"chain_id", "node_ids" (sorted), "edge_ids", "mag"}; None when no qualifying
    chain favors that side.
    """
    ctx = node_accrual(round_state.nodes, round_state.edges, as_of=None)
    chains = resolve_chains(ctx, emit_trace=False)
    excluded = weighing_excluded(ctx, chains)
    best = best_extended_chain_by_side(ctx, chains, excluded)

    out: Dict[str, Optional[dict]] = {}
    for side, ch in best.items():
        if ch is None:
            out[side] = None
            continue
        members = ch["members"]
        edge_ids = [e.id for e in round_state.edges
                    if e.source in members and e.target in members]
        out[side] = {"chain_id": ch["id"], "node_ids": sorted(members),
                     "edge_ids": edge_ids, "mag": ch["mag"]}
    return out


# --- render-layer wiring -------------------------------------------------------

def annotate_elements(round_state: Round, materiality: MaterialityResult,
                      best_paths: Optional[Dict[str, Optional[dict]]] = None) -> List[dict]:
    """Return the round's on-disk element dicts (the render/frontend M0/M1 ID scheme,
    n*/e*) with a boolean `material` flag stamped into every node/edge `data`, plus an
    optional `best_path` flag when `best_paths` is supplied. The frontend maps
    material=False -> translucent.

    Non-invasive by design: it does NOT mutate the model dataclasses, the judge, or the
    on-disk save schema -- it annotates a fresh export list keyed by node_id/edge_id, so
    the analysis stays fully read-only. Elements absent from the materiality map default
    to material=True (visible), so a partial map can never silently hide structure.
    """
    els = elements_from_round(round_state)
    on_path: set = set()
    if best_paths:
        for path in best_paths.values():
            if path:
                on_path |= set(path["node_ids"]) | set(path["edge_ids"])
    for el in els:
        eid = el["data"]["id"]
        el["data"]["material"] = materiality.material.get(eid, True)
        if best_paths is not None:
            el["data"]["best_path"] = eid in on_path
    return els
