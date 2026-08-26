"""Produce ONE builder-loadable sample round from the BC-warm-started actor alone.

This uses ONLY the already-completed imitation warm-start (behavior cloning) -- NO PPO, NO
self-play pool, NO real training run. It exists to get something showable into the existing
Dash + dash-cytoscape builder as fast as possible.

Pipeline:
  1. Warm-start a fresh ActorCritic on the fixture corpus (actor + critic; ruled config).
  2. Play ONE self-play episode -- the BC actor drives BOTH sides -- stepping the real env
     through the same legal-action masking the heads enforce, and CAPTURE the terminal
     graph. (rollout.collect_episode deliberately keeps only the learner's decisions for
     PPO and discards the final graph, so this plays its own episode loop to keep
     `env.state`, then materializes it with `state.to_round()`.)
  3. Serialize the terminal Round to the EXACT format the builder loads
     (`model.serialize` -> {"version": 2, "elements": [...]}), the same format the oracle
     fixtures use.

Content-blindness note: the policy emits no free-text content, so every node's `label`
is empty. For legibility in the builder we set each node's label to its POLICY-DECLARED
role (e.g. "link", "impact") -- that is the role the introduce action actually chose, not
fabricated argument text. Pass --raw-labels to keep labels empty instead.

Run:  python -m training.sample_round --epochs 20 --seed 0 --out samples/bc_sample_round.json
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import Counter, defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import serialize, SPEECH_ORDER, SPEECH_SIDE
from judge import judge as run_judge
from judge.config import AFF, NEG
from env import CDAFEnvironment, observe
from env.actions import (
    Introduce, Extend, Concede, Weigh, Connect, EndSpeech,
)
from policy import ActorCritic
from training.config import TrainingConfig
from training.checkpoint import new_actor_critic, EncoderSpec
from training.loop import warm_start


def _action_kind(a) -> str:
    return type(a).__name__


def play_capturing_round(actor: ActorCritic, *, seed: int = 0):
    """Play one self-play episode with `actor` on BOTH sides, capturing the terminal
    RoundState and a per-step action log. Every action is sampled through the heads' legal
    mask and applied via the real env, so the graph is legal + complete by construction."""
    gen = torch.Generator().manual_seed(seed)
    env = CDAFEnvironment()          # shaping OFF (default 0.0): honest terminal-only round
    env.reset()
    log = []                          # (slot, side, action_kind, detail)
    ended_by = {}                     # slot -> "end_speech" | "budget_exhausted"
    with torch.no_grad():
        while not env.state.terminated:
            slot = env.state.current_slot
            side = env.state.current_side
            budget_before = env.state.remaining_budget
            enc_out = actor.evaluate(observe(env.state))
            sa = actor.sample_action(enc_out, env.state, generator=gen)
            log.append((slot, side, sa.action_type, sa.action))
            if isinstance(sa.action, EndSpeech):
                ended_by[slot] = "end_speech"
            env.step(sa.action)
            # if the slot advanced without an explicit end_speech, budget ran out.
            if env.state.current_slot != slot and slot not in ended_by:
                ended_by[slot] = "budget_exhausted"
    return env.state, log, ended_by


def summarize(state, log, ended_by) -> dict:
    """Structural / degeneracy diagnostics for the produced round (honest reporting)."""
    rnd = state.to_round()
    ballot, _trace = run_judge(rnd)
    nodes = list(state.nodes.values())
    n_nodes = len(nodes)
    n_edges = len(state.edges)

    by_speech_types = defaultdict(Counter)
    for slot, _side, kind, _a in log:
        by_speech_types[slot][kind] += 1
    nodes_per_speech = Counter(n.introduction_speech for n in nodes)
    roles = Counter(n.role for n in nodes)
    edge_types = Counter(e.edge_type for e in state.edges)
    carriage = sum(1 for (_s, _sd, k, _a) in log if k in ("extend", "concede"))
    speeches_used = [s for s in SPEECH_ORDER if nodes_per_speech.get(s, 0) > 0]

    return {
        "winner": ballot, "n_nodes": n_nodes, "n_edges": n_edges,
        "total_actions": len(log), "carriage_actions": carriage,
        "roles": dict(roles), "edge_types": dict(edge_types),
        "nodes_per_speech": {s: nodes_per_speech.get(s, 0) for s in SPEECH_ORDER},
        "speeches_with_nodes": speeches_used,
        "action_types_by_speech": {s: dict(by_speech_types.get(s, {})) for s in SPEECH_ORDER},
        "ended_by": {s: ended_by.get(s, "n/a") for s in SPEECH_ORDER},
    }


def _label_by_role(rnd):
    """Set each node's label to its policy-declared role/type for builder legibility (the
    role the introduce action chose -- not fabricated content)."""
    for el in rnd.elements:
        if hasattr(el, "ntype") and not getattr(el, "label", None):
            el.label = getattr(el, "kind", None) or el.ntype
    return rnd


def build_sample(*, epochs: int, seed: int, out_path: str, raw_labels: bool = False,
                 encoder_spec: EncoderSpec = None, config: TrainingConfig = None):
    config = config or TrainingConfig()          # ruled semantics (γ etc.); shaping OFF
    encoder_spec = encoder_spec or EncoderSpec()
    ac = new_actor_critic(encoder_spec, seed=seed)
    # actor + critic behavior cloning on the whole corpus (critic uses the ruled discount).
    warm_start(ac, config, train_critic=True, epochs=epochs, seed=seed)
    ac.eval()

    state, log, ended_by = play_capturing_round(ac, seed=seed)
    summary = summarize(state, log, ended_by)

    rnd = state.to_round()
    if not raw_labels:
        _label_by_role(rnd)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    serialize.save(rnd, out_path)
    return out_path, summary


def _print_summary(out_path, summary, epochs, seed):
    print("=" * 72)
    print("BC-ONLY SAMPLE ROUND  (imitation warm-start only; NO PPO/self-play)")
    print("=" * 72)
    print(f"warm-start epochs={epochs}  seed={seed}")
    print(f"winner (judge verdict): {summary['winner']}")
    print(f"nodes={summary['n_nodes']}  edges={summary['n_edges']}  "
          f"total_actions={summary['total_actions']}  carriage(extend/concede)={summary['carriage_actions']}")
    print(f"roles: {summary['roles']}")
    print(f"edge_types: {summary['edge_types']}")
    print(f"nodes per speech: {summary['nodes_per_speech']}")
    print(f"speeches with >=1 node: {summary['speeches_with_nodes']}")
    print("action types by speech:")
    for s in SPEECH_ORDER:
        print(f"   {s:8s} ({SPEECH_SIDE[s]}): {summary['action_types_by_speech'][s]}  "
              f"[ended: {summary['ended_by'][s]}]")
    print(f"\nsaved: {out_path}")
    print("=" * 72)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Produce one BC-only builder-loadable sample round.")
    ap.add_argument("--epochs", type=int, default=20, help="warm-start epochs over the corpus")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="samples/bc_sample_round.json")
    ap.add_argument("--raw-labels", action="store_true",
                    help="keep node labels empty (content-blind) instead of role labels")
    args = ap.parse_args(argv)
    out_path, summary = build_sample(
        epochs=args.epochs, seed=args.seed, out_path=args.out, raw_labels=args.raw_labels)
    _print_summary(out_path, summary, args.epochs, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
