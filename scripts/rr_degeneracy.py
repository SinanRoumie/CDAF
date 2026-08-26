"""Degeneracy probe for the cross-seed round robin.

For each seed, play K self-play rounds (seed vs itself, the diagonal setting) capturing the
terminal graph, and report per seed:
  - mean nodes / edges per round
  - node-kind histogram (mean per round)
  - fraction of rounds with ANY advocacy node, and with an AFF-owned advocacy node
  - self-play reason_class distribution and AFF/NEG offense-established rates

This distinguishes a COLLAPSED policy (never establishes offense on either side -- structurally
degenerate) from a merely WEAK AFF policy. Evaluation only; runs locally off the banked
checkpoints. K=200 default (~cheap).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from judge.config import AFF, NEG
from training.checkpoint import load_checkpoint
from training.rollout import collect_episode

import random


def ballot_rec(traj):
    for r in traj.diagnostics:
        if getattr(r, "kind", None) == "BALLOT":
            return r
    return None


def probe_seed(ac, *, k, base_seed):
    nodes_tot = edges_tot = 0
    kind_tot = Counter()
    any_adv = aff_adv = 0
    reasons = Counter()
    aff_off = neg_off = 0
    for t in range(k):
        cs = base_seed + t
        rng = random.Random(cs)
        gen = torch.Generator().manual_seed(cs)
        traj, rnd = collect_episode(ac, ac, learner_side=AFF, rng=rng,
                                    torch_generator=gen, capture_round=True)
        nodes = [e for e in rnd.elements if hasattr(e, "ntype")]
        edges = [e for e in rnd.elements if hasattr(e, "source")]
        nodes_tot += len(nodes)
        edges_tot += len(edges)
        advs = [n for n in nodes if getattr(n, "kind", None) == "advocacy"]
        kind_tot.update(getattr(n, "kind", None) for n in nodes)
        if advs:
            any_adv += 1
        if any(getattr(n, "side", None) == AFF for n in advs):
            aff_adv += 1
        rec = ballot_rec(traj)
        if rec is not None:
            reasons[rec.reason_class] += 1
            if rec.reason_class == "AFF offense":
                aff_off += 1
            if rec.reason_class == "NEG offense":
                neg_off += 1
    return {
        "k": k,
        "mean_nodes": nodes_tot / k,
        "mean_edges": edges_tot / k,
        "mean_kinds": {kk: kind_tot[kk] / k for kk in sorted(kind_tot)},
        "frac_any_advocacy": any_adv / k,
        "frac_aff_advocacy": aff_adv / k,
        "reason_class": dict(reasons),
        "aff_offense_rate": aff_off / k,
        "neg_offense_rate": neg_off / k,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--ckpt-template", default="runs_phase2/W3_B4/seed{seed}/ckpt_u0499.pt")
    ap.add_argument("--out", default="runs_rr/degeneracy.json")
    args = ap.parse_args(argv)
    torch.set_num_threads(1)

    out = {}
    for s in range(5):
        ac, _ = load_checkpoint(args.ckpt_template.format(seed=s))
        ac.eval()
        r = probe_seed(ac, k=args.k, base_seed=777000 + s * 1000)
        out[s] = r
        print(f"seed{s}: nodes={r['mean_nodes']:.1f} edges={r['mean_edges']:.1f} "
              f"anyAdv={r['frac_any_advocacy']:.3f} affAdv={r['frac_aff_advocacy']:.3f} "
              f"AFFoff={r['aff_offense_rate']:.3f} NEGoff={r['neg_offense_rate']:.3f} "
              f"reasons={r['reason_class']}", flush=True)
        print(f"        kinds/round={ {k: round(v,2) for k,v in r['mean_kinds'].items()} }",
              flush=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
