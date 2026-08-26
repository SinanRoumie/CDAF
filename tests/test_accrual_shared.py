"""Differential guard: the extracted `node_accrual` IS the judge's accrual.

`passes.node_accrual` is the single shared implementation of per-node DF-QuAD strength
(σ) and effective polarity, called by BOTH the judge (via `pass_accrual`) and the env
observation. This test pins that identity node-for-node on every oracle fixture: run the
judge's accrual over each round, then call `node_accrual` STANDALONE on the same
BD-reachable subgraph, and assert σ and eff_pol match exactly. If the two ever diverge
(e.g. someone reintroduces a second implementation on one path), this fails.

NOTE ON INPUTS (the deliberate split, see environment_shell_spec.md §Node-level
accrual): the JUDGE passes its BD-reachable subset; the ENV observation passes the WHOLE
graph. This test compares the two callers on IDENTICAL inputs (the reachable subset), so
it verifies the function, not the caller-scope choice -- that split is the spec's to make
legible, not this test's to check.
"""

import glob
import os

from model import serialize
from judge import passes

ORACLE_DIR = os.path.join(os.path.dirname(__file__), "oracle")


def _fixtures():
    return sorted(glob.glob(os.path.join(ORACLE_DIR, "*.json")))


def test_node_accrual_matches_judge_per_fixture():
    checked = 0
    for path in _fixtures():
        name = os.path.basename(path)
        rnd = serialize.load(path)
        # the judge's accrual (pass_accrual routes through node_accrual and copies into ctx)
        ctx = passes.build_context(rnd)
        passes.pass2_drops(ctx)
        passes.pass_accrual(ctx)
        # an INDEPENDENT standalone call on the same BD-reachable subgraph
        view_nodes = [ctx.nodes[i] for i in ctx.reachable]
        view_edges = [e for e in ctx.edges
                      if e.source in ctx.reachable and e.target in ctx.reachable]
        acc = passes.node_accrual(view_nodes, view_edges)
        assert acc.sigma == ctx.sigma, f"{name}: sigma mismatch"
        assert acc.eff_pol == ctx.eff_pol, f"{name}: eff_pol mismatch"
        checked += 1
    assert checked >= 40   # the whole oracle corpus, not a truncated glob


def test_observation_accrual_view_matches_model_objects():
    """The env's NodeView/EdgeView feed `node_accrual` byte-identically to model.Node/
    Edge on the SAME graph -- guards the view adapter (role->kind, owner->side, stamped
    liveness). Compares the observation's accrual to `node_accrual` over the materialized
    model objects (whole graph, matching the observation's caller-scope)."""
    from env import CDAFEnvironment, Introduce, NEW
    from env.actions import EndSpeech
    from env.observation import observe
    env = CDAFEnvironment(); env.reset()
    env.step(Introduce("adv", "advocacy", NEW)); adv = list(env.state.nodes)[-1]
    env.step(Introduce("lk", "link", adv, "support")); lk = list(env.state.nodes)[-1]
    env.step(Introduce("im", "impact", lk, "support"))
    env.step(Introduce("vote", "ballot_directive", lk, "support"))
    env.step(EndSpeech())                                          # -> 1NC (NEG)
    # a CROSS-SIDE offense on the AFF link (NEG link -> AFF link, both offense-bearing):
    # legal, and exercises the offense/eff_pol path. A same-side offense would now be
    # illegal (masking ruling).
    env.step(Introduce("no", "link", lk, "offensive_attack"))
    obs = observe(env.state)
    rnd = env.state.to_round()
    # same horizon on both sides -> this isolates the VIEW adapter, not the as_of gate
    acc = passes.node_accrual(list(rnd.nodes), list(rnd.edges), as_of=env.state.current_slot)
    for nid in env.state.nodes:
        assert obs["accrual"]["sigma"][nid] == acc.sigma.get(nid), f"sigma {nid}"
        assert obs["accrual"]["eff_pol"][nid] == acc.eff_pol.get(nid), f"eff_pol {nid}"
