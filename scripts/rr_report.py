"""Build the cross-seed round-robin report from the per-row JSONs.

Reads rr_aff{0..4}.json (each a single AFF row produced by cross_seed_rr.py) and emits:

  1. Directional win-rate matrix M[i][j] = P(seed i wins | i=AFF, j=NEG), with Wilson 95% CIs.
  2. The presumption-adjusted DIRECTIONAL baseline p_pres = mean of the self-play diagonal
     M[i][i], with the reason_class decomposition showing WHY NEG wins the rest (ties /
     framework lock-out / AFF structural failure -- i.e. AFF failed to establish offense,
     which presumption banks -- vs NEG actually establishing offense).
  3. The side-averaged DOMINANCE matrix S[A][B] = 0.5*M[A][B] + 0.5*(1 - M[B][A]), whose
     equal-skill baseline is EXACTLY 0.5 (presumption cancels across the role swap), with a
     z-test of S vs 0.5 per pair.
  4. Copeland dominance ranking + any 3-cycle (A>B>C>A), a hard disproof of one equilibrium.

Usage: python -m scripts.rr_report --dir <dir with rr_aff*.json> --out rr_analysis.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

SEEDS = [0, 1, 2, 3, 4]
# NEG-win reason classes that mean "AFF failed to establish prevailing offense" (presumption
# banks these). Only "NEG offense" is NEG actually winning on its own offense.
PRESUMPTION_FAMILY = {"presumption", "framework lock-out", "AFF structural failure"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def load_rows(d):
    rows = {}
    for f in sorted(glob.glob(os.path.join(d, "rr_aff*.json"))):
        j = json.load(open(f))
        rows[int(j["aff_seed"])] = j
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default="rr_analysis.json")
    args = ap.parse_args(argv)

    rows = load_rows(args.dir)
    seeds = sorted(rows.keys())
    n = rows[seeds[0]]["n"]

    # M[i][j] win rate, wins[i][j] count
    M = {i: {} for i in seeds}
    wins = {i: {} for i in seeds}
    reasons = {i: {} for i in seeds}
    for i in seeds:
        for j in seeds:
            cell = rows[i]["results"][str(j)]
            M[i][j] = cell["aff_win_rate"]
            wins[i][j] = cell["aff_wins"]
            reasons[i][j] = cell["reason_class_counts"]

    # --- Baseline: diagonal self-play ---
    diag = {i: M[i][i] for i in seeds}
    p_pres = sum(diag.values()) / len(seeds)
    # decompose diagonal NEG wins
    diag_reason_tot = {}
    for i in seeds:
        for rc, c in reasons[i][i].items():
            diag_reason_tot[rc] = diag_reason_tot.get(rc, 0) + c
    diag_total = sum(diag_reason_tot.values())
    neg_wins_diag = diag_total - diag_reason_tot.get("AFF offense", 0)
    presum_family_diag = sum(diag_reason_tot.get(rc, 0) for rc in PRESUMPTION_FAMILY)
    neg_offense_diag = diag_reason_tot.get("NEG offense", 0)

    # --- Dominance S[A][B] and z vs 0.5 ---
    S = {a: {} for a in seeds}
    Z = {a: {} for a in seeds}
    for a in seeds:
        for b in seeds:
            if a == b:
                S[a][b] = 0.5
                Z[a][b] = 0.0
                continue
            p1, p2 = M[a][b], M[b][a]
            s = 0.5 * p1 + 0.5 * (1 - p2)
            se = 0.5 * math.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n)
            S[a][b] = s
            Z[a][b] = (s - 0.5) / se if se > 0 else 0.0

    # Copeland: count pairs where A dominates B at |z|>=2
    copeland = {a: 0 for a in seeds}
    dom_edges = []      # a beats b
    for a in seeds:
        for b in seeds:
            if a == b:
                continue
            if S[a][b] > 0.5 and Z[a][b] >= 2.0:
                copeland[a] += 1
                dom_edges.append((a, b))

    # 3-cycles in the significant-dominance tournament
    edgeset = set(dom_edges)
    cycles = []
    for a in seeds:
        for b in seeds:
            for c in seeds:
                if len({a, b, c}) == 3 and (a, b) in edgeset and (b, c) in edgeset and (c, a) in edgeset:
                    cyc = tuple(sorted([a, b, c]))
                    if (a, b, c) not in [x[1] for x in cycles]:
                        cycles.append((cyc, (a, b, c)))
    # dedup cycles by set
    seen = set()
    uniq_cycles = []
    for cyc, order in cycles:
        if cyc not in seen:
            seen.add(cyc)
            uniq_cycles.append(order)

    # ---- print report ----
    def fmt_row(d): return "  ".join(f"{d[j]:.3f}" for j in seeds)
    print("=" * 78)
    print(f"CROSS-SEED ROUND ROBIN  (n={n}/cell, W3_B4 seed{seeds} u0499)")
    print("=" * 78)
    print("\n[1] DIRECTIONAL WIN-RATE MATRIX  M[AFF][NEG] = P(AFF-seed wins)")
    print("      NEG:   " + "     ".join(f"s{j}" for j in seeds))
    for i in seeds:
        star = "  <- diagonal(self-play)" if False else ""
        print(f"  AFF s{i}:  " + fmt_row(M[i]))
    print("  (diagonal = self-play baseline per seed)")

    print("\n[2] PRESUMPTION-ADJUSTED DIRECTIONAL BASELINE (why 50% is not neutral)")
    for i in seeds:
        p, lo, hi = wilson(wins[i][i], n)
        print(f"  seed{i} self-play AFF win = {p:.3f}  (95% CI {lo:.3f}-{hi:.3f})")
    print(f"  --> p_pres = mean diagonal = {p_pres:.3f}")
    print(f"  Diagonal decomposition over {diag_total} self-play rounds:")
    for rc in ["AFF offense", "NEG offense"] + sorted(PRESUMPTION_FAMILY):
        c = diag_reason_tot.get(rc, 0)
        print(f"     {rc:24s}: {c:5d}  ({c/diag_total:.3f})")
    print(f"  Of {neg_wins_diag} NEG self-play wins: {presum_family_diag} "
          f"({presum_family_diag/max(1,neg_wins_diag):.3f}) are AFF-failed-to-establish "
          f"(presumption family), only {neg_offense_diag} are NEG offense.")
    print(f"  => NEG's edge is presumption on undecided rounds, not superior play. "
          f"Equal-skill AFF rate is {p_pres:.3f}, not 0.5.")

    print("\n[3] DOMINANCE MATRIX  S[A][B] = 0.5*M[A][B] + 0.5*(1-M[B][A])  (equal-skill = 0.500)")
    print("      vs:    " + "     ".join(f"s{j}" for j in seeds))
    for a in seeds:
        cells = []
        for b in seeds:
            tag = "*" if (a != b and abs(Z[a][b]) >= 2.0) else " "
            cells.append(f"{S[a][b]:.3f}{tag}")
        print(f"  s{a} :  " + " ".join(cells))
    print("  (* = |z| >= 2 vs 0.5; S[A][B]>0.5 means A beats B side-averaged)")

    print("\n[4] DOMINANCE RANKING (Copeland: # seeds significantly beaten, |z|>=2)")
    for a in sorted(seeds, key=lambda x: -copeland[x]):
        beaten = [f"s{b}" for b in seeds if (a, b) in edgeset]
        print(f"  seed{a}: Copeland={copeland[a]}  beats {beaten}")
    print("\n  Significant directed dominances (A beats B):")
    for a, b in dom_edges:
        print(f"    s{a} > s{b}   (S={S[a][b]:.3f}, z={Z[a][b]:+.1f})")

    print("\n[5] TRANSITIVITY / CYCLES")
    if uniq_cycles:
        for order in uniq_cycles:
            a, b, c = order
            print(f"  CYCLE: s{a} > s{b} > s{c} > s{a}  "
                  f"(S {S[a][b]:.3f}, {S[b][c]:.3f}, {S[c][a]:.3f}) "
                  f"-- hard disproof of a single stable equilibrium")
    else:
        print("  No 3-cycle among significant dominances (tournament is transitive at |z|>=2).")

    out = {
        "n": n, "seeds": seeds,
        "M": M, "wins": wins,
        "diagonal": diag, "p_pres": p_pres,
        "diag_reason_totals": diag_reason_tot,
        "S": S, "Z": Z, "copeland": copeland,
        "dominance_edges": dom_edges, "cycles": uniq_cycles,
    }
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n[wrote] {args.out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
