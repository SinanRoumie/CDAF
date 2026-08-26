"""Tests for the Phase-4 factored action heads + critic (`policy.heads`, `policy.masking`).

Verifies the milestone's required properties:
  * illegal actions/targets NEVER receive nonzero probability mass under the mask
    (masking, not post-hoc filtering) -- at the type, pointer-target, and edge_type
    stages, including the Support-cycle connect case;
  * the factored heads COMPOSE into a single valid action (every sample is
    structurally legal per the env's own generator);
  * POINTER-NET target selection works across the ~10-40 node range (logits scale with
    node count -- N for extend/connect, N+1 for introduce's NEW sentinel);
  * the CRITIC value head is scalar-shaped and reads ONLY the pooled graph embedding,
    never per-node embeddings.

Masks come from `LegalActionMask`, which delegates every legality decision to
`env.legal_actions.is_legal` -- the same entry point `CDAFEnvironment.step()` uses.
"""

from __future__ import annotations

import copy
import os
import random
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

from env import CDAFEnvironment, is_legal, NEW
from env.actions import Introduce, Extend, Concede, Weigh, Connect, EndSpeech
from env.observation import observe
from policy import GraphEncoder, ActorCritic, LegalActionMask, masked_probs
from policy.masking import ACTION_TYPES, EDGE_TYPE_ORDER

from fuzz_env_shell import sample_legal


# --- fixtures: states/observations at varying node counts --------------------

def _collect(lo=10, hi=40, seeds=range(2000)):
    """{node_count: (state, obs)} spanning [lo, hi], one representative per count.

    Post Ruling 1&2 the random-legal ceiling dropped from ~33 to ~31, so the break
    targets 30 (not 33) and scans more seeds to reach it."""
    want = set(range(lo, hi + 1))
    pool = {}
    for s in seeds:
        rng = random.Random(s)
        env = CDAFEnvironment()
        env.reset()
        while not env.state.terminated:
            action, _ = sample_legal(env.state, rng)
            _obs, _r, done, _info = env.step(action)
            if done:
                break
            n = len(env.state.nodes)
            if n in want and n not in pool:
                pool[n] = (copy.deepcopy(env.state), observe(env.state))
        if len(pool) >= 12 and max(pool) >= 30:
            break
    return pool


def _spread(available, k):
    available = sorted(available)
    if len(available) <= k:
        return available
    idx = [round(i * (len(available) - 1) / (k - 1)) for i in range(k)]
    return sorted({available[i] for i in idx})


_POOL = _collect()
_SIZES = _spread(_POOL, 7)
assert len(_SIZES) >= 5 and min(_SIZES) <= 12 and max(_SIZES) >= 30, \
    f"fuzzer produced too narrow a size range: {sorted(_POOL)}"
_SUBSET = _spread(_SIZES, 3)


def _model():
    torch.manual_seed(0)
    enc = GraphEncoder()
    ac = ActorCritic(enc)
    ac.eval()
    return enc, ac


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


# --- critic value head -------------------------------------------------------

def test_value_is_scalar():
    _, ac = _model()
    _state, obs = _POOL[_SIZES[len(_SIZES) // 2]]
    with torch.no_grad():
        out = ac.evaluate(obs)
        v = ac.value(out.graph_embedding)
    assert v.shape == torch.Size([])          # scalar
    assert torch.isfinite(v)


def test_value_reads_only_pooled():
    """The critic consumes ONLY the pooled graph vector: its input width is graph_dim,
    and value() runs given just that vector -- no per-node embeddings in the signature."""
    enc, ac = _model()
    assert ac.value_head[0].in_features == enc.graph_dim
    # Calling with a bare pooled-shaped vector (no node embeddings available) works.
    with torch.no_grad():
        v = ac.value(torch.zeros(enc.graph_dim))
    assert v.shape == torch.Size([])


def test_value_empty_graph():
    """Value is defined on the reset (zero-node) observation -- pooled vector exists."""
    enc, ac = _model()
    env = CDAFEnvironment()
    obs = env.reset()
    with torch.no_grad():
        out = ac.evaluate(obs)
        v = ac.value(out.graph_embedding)
    assert v.shape == torch.Size([])


# --- masking: illegal never gets probability mass ----------------------------

def test_type_mask_zeros_illegal_on_empty_graph():
    """On the reset graph only `introduce` and `end_speech` are legal; the other four
    types must receive exactly zero probability."""
    enc, ac = _model()
    env = CDAFEnvironment()
    obs = env.reset()
    mask = LegalActionMask(env.state)
    tm = mask.type_mask()
    # sanity on the mask itself (delegated to the generator)
    assert tm[ACTION_TYPES.index("introduce")] and tm[ACTION_TYPES.index("end_speech")]
    for t in ("extend", "concede", "weigh", "connect"):
        assert not tm[ACTION_TYPES.index(t)]
    with torch.no_grad():
        out = ac.evaluate(obs)
        probs = masked_probs(ac.type_head(out.graph_embedding),
                             torch.as_tensor(tm, dtype=torch.bool))
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-6)
    for t in ("extend", "concede", "weigh", "connect"):
        assert probs[ACTION_TYPES.index(t)].item() == 0.0


def test_weigh_second_target_excludes_self():
    """weigh's second-node mask forbids re-selecting the first node; that index must get
    zero probability under the pointer distribution."""
    enc, ac = _model()
    # any state with >= 2 nodes
    state, obs = _POOL[_SIZES[0]]
    mask = LegalActionMask(state)
    # Ruling 1 (same-kind weigh) makes an arbitrary pair need not be legal, so anchor on a
    # node that HAS at least one legal (same-kind) partner rather than assuming node 0/1 do.
    a_idx = next(i for i in range(len(mask.node_ids))
                 if any(mask.weigh_b_mask(mask.node_ids[i])))
    a_id = mask.node_ids[a_idx]
    bmask = mask.weigh_b_mask(a_id)
    assert not bmask[a_idx]                    # cannot weigh a node against itself
    b_idx = next(k for k, legal in enumerate(bmask) if legal)     # a legal same-kind partner
    assert is_legal(state, Weigh(a_id, mask.node_ids[b_idx], a_id))   # some pair IS legal
    with torch.no_grad():
        out = ac.evaluate(obs)
        q = ac._q("weigh_b", out.graph_embedding)
        probs = masked_probs(ac.pointer(out.node_embeddings, q),
                             torch.as_tensor(bmask, dtype=torch.bool))
    assert probs[a_idx].item() == 0.0
    # every zero-prob index is an illegal target and vice versa
    for k, legal in enumerate(bmask):
        assert (probs[k].item() > 0.0) == bool(legal)


def _build_cross_side_support_cycle():
    """A THREE-node support path s <- m <- t so that a `support` connect s -> t would close
    the directed cycle s -> t -> m -> s and is illegal, while the ATTACK edge_types between
    the CROSS-SIDE, offense-bearing endpoints s (AFF) and t (NEG) stay legal -- so
    `connect_edge_mask(s, t)` is [support=masked, defensive=legal, offensive=legal].

    The extra middle node `m` is deliberate: after the >=1-edge-per-unordered-pair ban
    (2026-08-13 edge-soup fix), a DIRECT s<->t edge would make the attack connects illegal
    too (the pair already carries an edge), collapsing the mask to all-False and hiding the
    cycle rule. Routing the cycle through `m` leaves the pair {s, t} edge-free, so this test
    isolates the Support-cycle rule exactly as before. (A same-side pair would have all three
    edge_types masked by the same-side rule, so a cross-side pair is still required.)"""
    env = CDAFEnvironment()
    env.reset()                                          # 1AC (AFF), budget 8
    env.step(Introduce("", "advocacy", NEW, None))       # AFF advocacy (legal root)
    adv = list(env.state.nodes)[-1]
    env.step(Introduce("", "link", adv, "support"))      # s = AFF link, attached to the advocacy
    s = list(env.state.nodes)[-1]
    env.step(Introduce("", "link", s, "support"))        # m = AFF link supports s: m -> s
    m = list(env.state.nodes)[-1]
    env.step(EndSpeech())                                # -> 1NC (NEG)
    env.step(Introduce("", "link", m, "support"))        # t = NEG link supports m: t -> m
    t = list(env.state.nodes)[-1]
    # t -> m -> s exists; connect s -> t (support) would close s -> t -> m -> s. No direct
    # s<->t edge, so the attack edge_types stay legal under the pair ban.
    return env, s, t


def test_connect_support_cycle_masked():
    """The one genuinely conditional structural rule: a `support` connect closing a
    Support cycle is masked out for that ordered pair, while defensive/offensive stay
    legal (cross-side, offense-bearing endpoints) -- and the masked edge_type gets zero
    probability."""
    enc, ac = _model()
    env, s_id, t_id = _build_cross_side_support_cycle()
    state = env.state
    mask = LegalActionMask(state)
    # ground truth straight from the generator
    assert not is_legal(state, Connect(s_id, t_id, "support"))       # closes a directed cycle
    assert is_legal(state, Connect(s_id, t_id, "defensive_attack"))  # cross-side -> legal
    assert is_legal(state, Connect(s_id, t_id, "offensive_attack"))  # cross-side, offense-bearing
    emask = mask.connect_edge_mask(s_id, t_id)
    assert list(emask) == [False, True, True]            # support masked; def/off legal

    obs = observe(state)
    isrc, itgt = mask.node_ids.index(s_id), mask.node_ids.index(t_id)
    with torch.no_grad():
        out = ac.evaluate(obs)
        ctx = torch.cat([out.graph_embedding, out.node_embeddings[isrc],
                         out.node_embeddings[itgt]], dim=0)
        probs = masked_probs(ac.connect_edge_head(ctx),
                             torch.as_tensor(emask, dtype=torch.bool))
    assert probs[EDGE_TYPE_ORDER.index("support")].item() == 0.0
    assert probs.sum().item() == pytest.approx(1.0, abs=1e-6)


# --- pointer-net target selection across sizes -------------------------------

@pytest.mark.parametrize("size", _SIZES)
def test_pointer_shapes_across_sizes(size):
    """Pointer logits scale with node count: N for a plain target, N+1 for introduce's
    NEW sentinel -- never a fixed index space."""
    enc, ac = _model()
    state, obs = _POOL[size]
    n = len(state.nodes)
    assert n == size
    with torch.no_grad():
        out = ac.evaluate(obs)
        extend_logits = ac.pointer(out.node_embeddings, ac._q("extend", out.graph_embedding))
        items = torch.cat([out.node_embeddings, ac.new_token.unsqueeze(0)], dim=0)
        intro_logits = ac.pointer(items, ac._q("introduce_target", out.graph_embedding))
    assert extend_logits.shape == (n,)
    assert intro_logits.shape == (n + 1,)
    assert torch.isfinite(extend_logits).all() and torch.isfinite(intro_logits).all()


# --- factored composition -> valid single action -----------------------------

@pytest.mark.parametrize("size", _SIZES)
def test_sampled_actions_always_legal(size):
    """Every factored, masked sample assembles into a structurally-legal env action."""
    enc, ac = _model()
    state, obs = _POOL[size]
    gen = _gen(size)
    with torch.no_grad():
        out = ac.evaluate(obs)
        for _ in range(60):
            sa = ac.sample_action(out, state, generator=gen)
            assert is_legal(state, sa.action), f"illegal sample: {sa.action}"
            assert sa.action_type in ACTION_TYPES


def test_sample_returns_concrete_action_types():
    """Sampling produces genuine env action objects across the vocabulary."""
    enc, ac = _model()
    seen = set()
    gen = _gen(1)
    with torch.no_grad():
        for size in _SIZES:
            state, obs = _POOL[size]
            out = ac.evaluate(obs)
            for _ in range(120):
                sa = ac.sample_action(out, state, generator=gen)
                seen.add(type(sa.action))
                assert isinstance(sa.action, (Introduce, Extend, Concede, Weigh, Connect, EndSpeech))
    # the pointer/param machinery for each multi-argument type actually fires
    assert {Introduce, Weigh, Connect} <= seen


def test_every_stage_distribution_is_valid():
    """For a full sample, each recorded stage distribution sums to 1, is non-negative,
    and puts zero mass on every masked (illegal) entry."""
    enc, ac = _model()
    state, obs = _POOL[_SIZES[-1]]
    gen = _gen(2)
    with torch.no_grad():
        out = ac.evaluate(obs)
        for _ in range(40):
            sa = ac.sample_action(out, state, generator=gen)
            for name, st in sa.stages.items():
                probs, m = st["probs"], torch.as_tensor(st["mask"], dtype=torch.bool)
                assert torch.all(probs >= 0)
                assert probs.sum().item() == pytest.approx(1.0, abs=1e-6)
                assert torch.all(probs[~m] == 0.0), f"mass on illegal entry in stage {name}"
                assert m[st["choice"]], f"chose a masked entry in stage {name}"


# --- content blindness of emitted actions ------------------------------------

def test_sampled_actions_are_content_blind():
    """The policy emits no content/justification -- assembled actions carry only the
    empty-string placeholder the judge never reads."""
    enc, ac = _model()
    gen = _gen(3)
    with torch.no_grad():
        for size in _SIZES:
            state, obs = _POOL[size]
            out = ac.evaluate(obs)
            for _ in range(80):
                sa = ac.sample_action(out, state, generator=gen)
                if isinstance(sa.action, Introduce):
                    assert sa.action.content == ""
                if isinstance(sa.action, Weigh):
                    assert sa.action.justification == ""


# --- shared encoder ----------------------------------------------------------

def test_single_shared_encoder_instance():
    """Actor and critic read one encoder; ActorCritic does not fork or duplicate it."""
    enc = GraphEncoder()
    ac = ActorCritic(enc)
    assert ac.encoder is enc
    encoders = [m for m in ac.modules() if isinstance(m, GraphEncoder)]
    assert len(encoders) == 1
