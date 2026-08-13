"""Tests for the Phase-4 graph encoder (`policy.encoder`).

Verifies the three properties the milestone requires:
  * SHAPE correctness across varying graph sizes (~10 to ~40 nodes),
  * PERMUTATION INVARIANCE of the pooled graph vector under node relabeling
    (and equivariance of the per-node embeddings),
  * CONTENT-BLINDNESS -- proven two ways: injecting content into the observation
    changes nothing, and corrupting the source state's content changes neither the
    observation nor the encoding.

Graphs are built by driving the real environment with the uniform-random legal
agent from the env fuzzer (`fuzz_env_shell.sample_legal`), so the encoder is tested
on the exact observations it will see in training, at realistic sizes.
"""

from __future__ import annotations

import copy
import os
import random
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

from env import CDAFEnvironment
from policy import GraphEncoder
from policy.features import extract_features, CONT_DIM

from fuzz_env_shell import sample_legal


# --- graph generation --------------------------------------------------------

def _snapshots(seed, targets):
    """Drive one random-legal round; return {node_count: observation} snapshots for
    every distinct node count hit that lies in `targets`. Observations are deep-copied
    so later steps cannot mutate a captured one."""
    rng = random.Random(seed)
    env = CDAFEnvironment()
    obs = env.reset()
    out = {}
    n = len(env.state.nodes)
    if n in targets:
        out[n] = copy.deepcopy(obs)
    while not env.state.terminated:
        action, _ = sample_legal(env.state, rng)
        obs, _r, done, _info = env.step(action)
        if done:
            break
        n = len(env.state.nodes)
        if n in targets and n not in out:
            out[n] = copy.deepcopy(obs)
    return out


def _build_pool(lo=10, hi=40, seeds=range(2000)):
    """Scan random-legal rounds, capturing one observation per distinct node count in
    [lo, hi]. Node count grows by exactly one per node-creating action, so a round of
    final size F contributes every count up to F; high counts (mostly-introduce play)
    are rare under uniform-random sampling, so we scan many seeds and stop once we have
    a broad spread. Returns {node_count: obs}.

    The Ruling 1&2 legality masks (same-kind weigh; cross-side defensive-attack on a
    BallotDirective) shrank the random-legal branching, lowering the ceiling reachable
    under uniform play from ~33 to ~31, so the break targets 30 (not 33) and scans more
    seeds to hit it. See docs/judge_spec.md (Ruling 1 & 2)."""
    want = set(range(lo, hi + 1))
    pool = {}
    for s in seeds:
        for n, obs in _snapshots(s, want).items():
            pool.setdefault(n, obs)
        # Enough coverage: a wide spread reaching into the (post-ruling) realistic band.
        if len(pool) >= 12 and max(pool) >= 30:
            break
    return pool


def _spread(available, k):
    """Pick up to `k` evenly-spaced values from sorted `available`, including both ends
    -- a representative span of graph sizes for parametrization."""
    available = sorted(available)
    if len(available) <= k:
        return available
    idx = [round(i * (len(available) - 1) / (k - 1)) for i in range(k)]
    return sorted({available[i] for i in idx})


# Representative graphs spanning the ~10-40 range the spec calls out. Sizes are chosen
# from what the fuzzer actually produces (realistic rounds run ~30-40 nodes; exact 40 is
# reachable but uncommon under random play, so we test the span rather than fixed 40).
_POOL = _build_pool()
_SIZES = _spread(_POOL, 7)
assert len(_SIZES) >= 5 and min(_SIZES) <= 12 and max(_SIZES) >= 30, \
    f"fuzzer produced too narrow a size range for a meaningful test: {sorted(_POOL)}"
_GRAPHS = _POOL
# A small, small-to-large subset for the more expensive property tests.
_SUBSET = _spread(_SIZES, 3)


def _encoder():
    torch.manual_seed(0)
    enc = GraphEncoder()
    enc.eval()
    return enc


# --- shape correctness -------------------------------------------------------

@pytest.mark.parametrize("size", _SIZES)
def test_shapes_across_sizes(size):
    assert size in _GRAPHS, f"could not build a {size}-node graph from the fuzzer"
    obs = _GRAPHS[size]
    enc = _encoder()
    with torch.no_grad():
        out = enc.encode(obs)
    n = len(obs["graph"]["nodes"])
    assert n == size
    assert out.node_embeddings.shape == (n, enc.d_model)
    assert out.graph_embedding.shape == (enc.graph_dim,)
    assert len(out.node_ids) == n
    assert torch.isfinite(out.node_embeddings).all()
    assert torch.isfinite(out.graph_embedding).all()


def test_feature_dim_matches_layout():
    """The declared CONT_DIM matches the array the extractor actually produces."""
    obs = _GRAPHS[_SIZES[0]]
    feats = extract_features(obs)
    assert feats.cont.shape == (feats.n_nodes, CONT_DIM)


def test_empty_graph():
    """The reset() observation has zero nodes; the encoder must still return a
    well-formed (empty node set, global-only) graph vector."""
    env = CDAFEnvironment()
    obs = env.reset()
    assert obs["graph"]["nodes"] == []
    enc = _encoder()
    with torch.no_grad():
        out = enc.encode(obs)
    assert out.node_embeddings.shape == (0, enc.d_model)
    assert out.graph_embedding.shape == (enc.graph_dim,)
    assert torch.isfinite(out.graph_embedding).all()


# --- permutation invariance / equivariance -----------------------------------

def _relabel(obs, perm, rename):
    """Return a copy of `obs` with nodes reordered by `perm` (new row p is old row
    perm[p]) and every node id remapped through `rename` (a bijection). All id
    references -- edges, reachability, accrual, drop/extension-failure lists -- are
    remapped consistently, so the graph is identical up to relabeling."""
    obs = copy.deepcopy(obs)
    g = obs["graph"]
    old_nodes = g["nodes"]
    g["nodes"] = [{**old_nodes[perm[p]], "id": rename[old_nodes[perm[p]]["id"]]}
                  for p in range(len(old_nodes))]
    for e in g["edges"]:
        e["source"] = rename[e["source"]]
        e["target"] = rename[e["target"]]
    obs["reachability"] = {rename[k]: v for k, v in obs["reachability"].items()}
    acc = obs["accrual"]
    acc["sigma"] = {rename[k]: v for k, v in acc["sigma"].items()}
    acc["eff_pol"] = {rename[k]: v for k, v in acc["eff_pol"].items()}
    obs["closed_window_drops"] = [rename[x] for x in obs["closed_window_drops"]]
    obs["permanent_extension_failures"] = [rename[x] for x in obs["permanent_extension_failures"]]
    return obs


@pytest.mark.parametrize("size", _SUBSET)
def test_permutation_invariance(size):
    obs = _GRAPHS[size]
    ids = [n["id"] for n in obs["graph"]["nodes"]]
    n = len(ids)

    rng = random.Random(1234 + size)
    perm = list(range(n))
    rng.shuffle(perm)
    # A bijective id renaming disjoint from the originals, to prove ids are pure
    # labels the encoder never keys on.
    rename = {old: f"z{i}" for i, old in enumerate(ids)}

    perm_obs = _relabel(obs, perm, rename)

    enc = _encoder()
    with torch.no_grad():
        base = enc.encode(obs)
        other = enc.encode(perm_obs)

    # Pooled graph vector is permutation-INVARIANT.
    assert torch.allclose(base.graph_embedding, other.graph_embedding, atol=1e-4, rtol=1e-4)

    # Per-node embeddings are EQUIVARIANT: new row p is the old row perm[p].
    for p in range(n):
        assert other.node_ids[p] == rename[base.node_ids[perm[p]]]
        assert torch.allclose(other.node_embeddings[p], base.node_embeddings[perm[p]],
                              atol=1e-4, rtol=1e-4)


# --- content blindness -------------------------------------------------------

def _inject_content(obs):
    """Splatter fake content/text fields everywhere the encoder might conceivably
    look. None of these keys are ones the extractor reads."""
    obs = copy.deepcopy(obs)
    obs["content"] = "TOP-LEVEL SECRET"
    obs["graph"]["content"] = "GRAPH SECRET"
    for i, node in enumerate(obs["graph"]["nodes"]):
        node["content"] = f"secret argument text {i} " * 5
        node["label"] = f"label {i}"
        node["text"] = "should never be read"
    for e in obs["graph"]["edges"]:
        e["content"] = "edge prose"
        e["label"] = "edge label"
    return obs


@pytest.mark.parametrize("size", _SUBSET)
def test_content_blind_injection(size):
    """Injecting content fields into the observation cannot change the output."""
    obs = _GRAPHS[size]
    enc = _encoder()
    with torch.no_grad():
        clean = enc.encode(obs)
        dirty = enc.encode(_inject_content(obs))
    assert torch.equal(clean.graph_embedding, dirty.graph_embedding)
    assert torch.equal(clean.node_embeddings, dirty.node_embeddings)


def test_content_blind_source_state():
    """Corrupting the SOURCE state's node content changes neither the observation nor
    the encoding -- the observation carries no content, so the encoder cannot see it."""
    rng = random.Random(7)
    env = CDAFEnvironment()
    env.reset()
    # Play until a non-trivial graph exists.
    obs = None
    while not env.state.terminated:
        action, _ = sample_legal(env.state, rng)
        obs, _r, done, _info = env.step(action)
        if done:
            obs = None
            break
        if len(env.state.nodes) >= 12:
            break
    assert obs is not None and len(env.state.nodes) >= 12

    enc = _encoder()
    with torch.no_grad():
        before = enc.encode(obs)

    # Rewrite every node's content in the live state, then re-observe.
    from env.observation import observe
    for rec in env.state.nodes.values():
        rec.content = "COMPLETELY DIFFERENT PROSE " * 3
    obs_after = observe(env.state)

    # The observation is byte-identical (it never carried content) ...
    assert obs_after["graph"] == obs["graph"]
    # ... and so is the encoding.
    with torch.no_grad():
        after = enc.encode(obs_after)
    assert torch.equal(before.graph_embedding, after.graph_embedding)
    assert torch.equal(before.node_embeddings, after.node_embeddings)


# --- determinism -------------------------------------------------------------

def test_deterministic():
    """Same weights + same observation -> identical output on repeated calls."""
    obs = _GRAPHS[_SIZES[len(_SIZES) // 2]]
    enc = _encoder()
    with torch.no_grad():
        a = enc.encode(obs)
        b = enc.encode(obs)
    assert torch.equal(a.graph_embedding, b.graph_embedding)
    assert torch.equal(a.node_embeddings, b.node_embeddings)
