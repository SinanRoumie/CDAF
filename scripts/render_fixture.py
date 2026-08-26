#!/usr/bin/env python3
"""M0 render CLI (spec §8). Deterministic, no LLM.

Usage:
    python scripts/render_fixture.py <name-or-path> [<name-or-path> ...]
    python scripts/render_fixture.py --stats <name ...>     # stats only

A bare name (e.g. `r1`, `A5_affwin_seed3_u25_ep079`) is resolved against the
oracle fixtures and the run-export directories. Read-only.
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from render.adapter import analyze, load_round
from render.linearize import render, summary_stats

_SEARCH_DIRS = [
    "tests/oracle",
    "runs/round_export_matrix_A_20260812", "runs/round_export_matrix_A2_20260812",
    "runs/round_export_matrix_A3_20260812", "runs/round_export_matrix_A4_20260812",
    "runs/round_export_matrix_A5_20260812",
    "runs_verify/B4post/exports", "runs_verify/B4edge/exports",
]


def resolve(name: str) -> str:
    if os.path.exists(name):
        return name
    stem = name[:-5] if name.endswith(".json") else name
    for d in _SEARCH_DIRS:
        cand = os.path.join(d, stem + ".json")
        if os.path.exists(cand):
            return cand
    hits = []
    for d in _SEARCH_DIRS:
        hits += glob.glob(os.path.join(d, stem + "*.json"))
    hits = [h for h in hits if not h.endswith(".sidecar.json")]
    if len(hits) == 1:
        return hits[0]
    raise FileNotFoundError(f"could not resolve {name!r} (matches: {hits})")


def main(argv):
    stats_only = "--stats" in argv
    names = [a for a in argv if not a.startswith("--")]
    for name in names:
        path = resolve(name)
        base = os.path.basename(path)[:-5]
        rnd = load_round(path)
        an = analyze(rnd)
        if not stats_only:
            print(render(an, rnd, base))
        print()
        print(f"# STATS {base}: {summary_stats(an, rnd)}")
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
