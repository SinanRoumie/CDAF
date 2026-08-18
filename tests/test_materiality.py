"""Tests for the post-hoc materiality / highlight pass (analysis/materiality.py).

Read-only: every assertion reruns the UNMODIFIED judge; none touch legality,
tabula-rasa, or judge internals.
"""

import copy
import os

import pytest

from model import serialize as mser
from analysis.materiality import (
    MaterialityResult,
    _ballot_margin,
    _delete_edge,
    _delete_node,
    annotate_elements,
    best_path_per_side,
    compute_materiality,
)

ORACLE = os.path.join(os.path.dirname(__file__), "oracle")


def _load(name):
    return mser.load(os.path.join(ORACLE, name))


# AFFLinkturnsNeg: AFF wins N=+1.0 with real offense -> some elements are material.
AFF_WIN = "AFFLinkturnsNeg.json"


def test_input_round_is_never_mutated():
    rnd = _load(AFF_WIN)
    before = mser.to_dict(rnd)
    compute_materiality(rnd)
    best_path_per_side(rnd)
    assert mser.to_dict(rnd) == before, "analysis must not mutate the input round"


def test_delete_node_drops_incident_edges():
    """Mirrors the builder: deleting a node removes it AND every incident edge."""
    rnd = _load(AFF_WIN)
    node = rnd.nodes[0]
    incident = {e.id for e in rnd.edges
                if e.source == node.id or e.target == node.id}
    out = _delete_node(rnd, node.id)
    ids = {el.id for el in out.elements}
    assert node.id not in ids
    assert incident.isdisjoint(ids), "incident edges must be removed with the node"
    # every OTHER original node survives
    assert {n.id for n in rnd.nodes} - {node.id} <= ids


def test_delete_edge_drops_only_that_edge():
    rnd = _load(AFF_WIN)
    edge = rnd.edges[0]
    out = _delete_edge(rnd, edge.id)
    ids = {el.id for el in out.elements}
    assert edge.id not in ids
    assert len(out.elements) == len(rnd.elements) - 1
    assert {n.id for n in rnd.nodes} <= ids, "no node may be removed by an edge delete"


def test_materiality_result_shape_and_baseline():
    rnd = _load(AFF_WIN)
    res = compute_materiality(rnd)
    assert isinstance(res, MaterialityResult)
    base_w, base_m = _ballot_margin(rnd)
    assert res.baseline_winner == base_w
    assert res.baseline_margin == pytest.approx(base_m)
    assert res.epsilon == 0.0  # zero-tolerance placeholder
    # every node and edge id is scored exactly once
    ids = {n.id for n in rnd.nodes} | {e.id for e in rnd.edges}
    assert set(res.material) == ids


def test_material_flag_matches_delete_and_rejudge():
    """A material element, when actually deleted, must flip the winner or move N;
    an immaterial one must do neither (epsilon=0.0)."""
    rnd = _load(AFF_WIN)
    res = compute_materiality(rnd, epsilon=0.0)
    base_w, base_m = res.baseline_winner, res.baseline_margin
    assert any(res.material.values()), "AFFLinkturnsNeg has offense -> expect material elements"
    node_ids = {n.id for n in rnd.nodes}
    for eid, is_mat in res.material.items():
        deleted = (_delete_node if eid in node_ids else _delete_edge)(rnd, eid)
        w, m = _ballot_margin(deleted)
        changed = (w != base_w) or (m != base_m)
        assert is_mat == changed


def test_epsilon_tolerance_suppresses_margin_only_changes():
    """A large epsilon can only REDUCE the material set (winner flips still count)."""
    rnd = _load(AFF_WIN)
    strict = compute_materiality(rnd, epsilon=0.0).material
    loose = compute_materiality(rnd, epsilon=10.0).material
    for eid in strict:
        assert not loose[eid] or strict[eid], "loosening epsilon cannot ADD material elements"


def test_best_path_per_side_reuses_phi_selection():
    rnd = _load(AFF_WIN)
    bp = best_path_per_side(rnd)
    assert set(bp) == {"AFF", "NEG"}
    node_ids = {n.id for n in rnd.nodes}
    edge_ids = {e.id for e in rnd.edges}
    for side, path in bp.items():
        if path is None:
            continue
        assert set(path["node_ids"]) <= node_ids
        assert set(path["edge_ids"]) <= edge_ids
        assert path["mag"] >= 0.0


def test_annotate_elements_stamps_material_by_id():
    rnd = _load(AFF_WIN)
    res = compute_materiality(rnd)
    bp = best_path_per_side(rnd)
    els = annotate_elements(rnd, res, bp)
    assert len(els) == len(rnd.elements)
    for el in els:
        data = el["data"]
        assert data["material"] == res.material[data["id"]]
        assert isinstance(data["best_path"], bool)
    # best_path flag is the union of both sides' chain members/edges
    on_path = set()
    for path in bp.values():
        if path:
            on_path |= set(path["node_ids"]) | set(path["edge_ids"])
    assert {el["data"]["id"] for el in els if el["data"]["best_path"]} == on_path


def test_annotate_without_best_paths_omits_flag():
    rnd = _load(AFF_WIN)
    els = annotate_elements(rnd, compute_materiality(rnd))
    assert all("material" in el["data"] for el in els)
    assert all("best_path" not in el["data"] for el in els)
