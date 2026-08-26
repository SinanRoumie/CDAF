"""Re-verify the three churned rows (s1, s2, s4) with a DIFFERENT RNG.

During collection, seeds 1/2/4's rows were killed and re-run once (process incident). The
independence argument (each process runs a complete n=1000 row and writes atomically at the end)
says the banked values are clean, but this rules it out empirically: re-run one representative
cell from each of those rows with a fresh RNG stream (cell_seed + OFFSET) and confirm the new
win rate lands inside the banked value's 95% Wilson CI.
"""
from __future__ import annotations

import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.checkpoint import load_checkpoint
from scripts.cross_seed_rr import play_cell, SEED_BASE

OFFSET = 500000                       # fresh RNG stream, disjoint from the banked run
N = 1000
CELLS = [(1, 3), (2, 0), (4, 3)]      # (aff, neg): one representative cell per churned row


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main():
    torch.set_num_threads(1)
    tmpl = "runs_phase2/W3_B4/seed{seed}/ckpt_u0499.pt"
    cache = {}

    def load(s):
        if s not in cache:
            ac, _ = load_checkpoint(tmpl.format(seed=s))
            ac.eval()
            cache[s] = ac
        return cache[s]

    print(f"Re-verification (n={N}, RNG offset +{OFFSET})")
    allok = True
    for aff, neg in CELLS:
        banked = json.load(open(f"runs_rr/rr_aff{aff}.json"))["results"][str(neg)]
        bp = banked["aff_win_rate"]
        blo, bhi = wilson(banked["aff_wins"], banked["n"])
        cell_seed = SEED_BASE + aff * 1000 + neg + OFFSET
        r = play_cell(load(aff), load(neg), n=N, cell_seed=cell_seed)
        rp = r["aff_win_rate"]
        inside = blo <= rp <= bhi
        allok &= inside
        print(f"  aff{aff} vs neg{neg}: banked={bp:.3f} (95%CI {blo:.3f}-{bhi:.3f})  "
              f"rerun={rp:.3f} ({r['aff_wins']}/{N})  -> {'INSIDE' if inside else 'OUTSIDE'}")
    print("ALL INSIDE CI" if allok else "SOME OUTSIDE CI -- investigate")
    json.dump({"offset": OFFSET, "n": N, "cells": CELLS, "all_inside": allok},
              open("runs_rr/verify.json", "w"), indent=2)


if __name__ == "__main__":
    main()
