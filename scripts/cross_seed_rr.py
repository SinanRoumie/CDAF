"""Cross-seed round-robin evaluation (EVALUATION ONLY, no training).

One invocation computes a single AFF row of the round-robin matrix: a fixed AFF seed
plays `--n` rounds against each NEG seed (including itself, the self-play diagonal).
Every round is played by `training.rollout.collect_episode` -- the same masked-legal
self-play path training uses -- so the judge scores exactly as in training. Sampling is
stochastic (the policy's own categorical, not argmax), so `--n` rounds yield a genuine
win-rate estimate.

We record, per (aff, neg) cell: AFF win count, and the judge's `reason_class` / net-offense
`N` distribution (from the terminal BALLOT trace record). The reason_class tally is what
lets the report show *why* NEG won -- presumption/tie vs NEG offense vs AFF structural
failure -- which is the evidence that 50% is not the presumption-adjusted neutral point.

Deterministic: cell (aff, neg) is seeded `SEED_BASE + aff*1000 + neg`, so the whole matrix
is reproducible and each pod's row is independent.

Usage:
  python -m scripts.cross_seed_rr --aff-seed 0 --neg-seeds 0,1,2,3,4 --n 1000 \
      --ckpt-template runs_phase2/W3_B4/seed{seed}/ckpt_u0499.pt --out rr_aff0.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from judge.config import AFF, NEG
from training.checkpoint import load_checkpoint
from training.rollout import collect_episode

SEED_BASE = 20260826


def _ballot_record(traj):
    """The terminal BALLOT trace record (reason_class, N) from a trajectory's diagnostics,
    or None if absent."""
    for r in traj.diagnostics:
        if getattr(r, "kind", None) == "BALLOT":
            return r
    return None


def play_cell(aff_ac, neg_ac, *, n, cell_seed):
    """Play `n` rounds with aff_ac on AFF, neg_ac on NEG. Returns a result dict with the
    AFF win count and the reason_class / N distribution."""
    rng = random.Random(cell_seed)
    gen = torch.Generator().manual_seed(cell_seed)
    aff_wins = 0
    reason_counts = {}
    n_sum = 0.0
    for _ in range(n):
        traj = collect_episode(aff_ac, neg_ac, learner_side=AFF, rng=rng,
                               torch_generator=gen)
        if traj.winner == AFF:
            aff_wins += 1
        rec = _ballot_record(traj)
        if rec is not None:
            reason_counts[rec.reason_class] = reason_counts.get(rec.reason_class, 0) + 1
            n_sum += float(rec.N)
    return {
        "n": n,
        "aff_wins": aff_wins,
        "aff_win_rate": aff_wins / n,
        "reason_class_counts": reason_counts,
        "mean_N": n_sum / n,
        "cell_seed": cell_seed,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Cross-seed round-robin (one AFF row).")
    ap.add_argument("--aff-seed", type=int, required=True)
    ap.add_argument("--neg-seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--ckpt-template", type=str,
                    default="runs_phase2/W3_B4/seed{seed}/ckpt_u0499.pt")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    neg_seeds = [int(x) for x in args.neg_seeds.split(",")]

    def load(seed):
        ac, meta = load_checkpoint(args.ckpt_template.format(seed=seed))
        ac.eval()
        return ac, meta

    aff_ac, aff_meta = load(args.aff_seed)
    neg_acs = {j: load(j) for j in neg_seeds}

    results = {}
    t0 = time.time()
    for j in neg_seeds:
        cell_seed = SEED_BASE + args.aff_seed * 1000 + j
        r = play_cell(aff_ac, neg_acs[j][0], n=args.n, cell_seed=cell_seed)
        results[str(j)] = r
        print(f"[aff{args.aff_seed} vs neg{j}] win={r['aff_win_rate']:.4f} "
              f"({r['aff_wins']}/{args.n}) meanN={r['mean_N']:+.4f} "
              f"reasons={r['reason_class_counts']}  ({time.time()-t0:.0f}s elapsed)",
              flush=True)

    out = {
        "aff_seed": args.aff_seed,
        "neg_seeds": neg_seeds,
        "n": args.n,
        "ckpt_template": args.ckpt_template,
        "aff_meta": aff_meta,
        "results": results,
        "wall_seconds": time.time() - t0,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] wrote {args.out} in {out['wall_seconds']:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
