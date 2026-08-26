#!/usr/bin/env python3
"""Random-agent crash test for the Phase 1 CDAF environment shell.

WHY HERE (not a pytest test): this is a long-running stochastic fuzzer, not an
assertion unit test. It lives under tests/ so it sits with the env suite and the
oracle fixtures it exercises, but it is deliberately NOT named test_* so a normal
`pytest tests/` never collects the 500-round run. Invoke it directly:

    python tests/fuzz_env_shell.py                 # 500 rounds, master seed 0
    python tests/fuzz_env_shell.py --rounds 500 --seed 0
    python tests/fuzz_env_shell.py --replay <round_seed>   # replay one round, verbose

WHAT IT DOES: an agent that samples UNIFORMLY from the legal action set each step
and plays to termination (a full 7-speech / up-to-52-action round), repeated N
times. The shell has never played a full round end to end; the point is to surface
latent Phase-1 bugs -- exceptions, and (both should be UNREACHABLE) a
structural-admission assertion or scope-guard trip, either of which means the
legal-action generator has a gap.

REPRODUCIBILITY: each round is driven by `random.Random(round_seed)`. Round seeds
are drawn from `random.Random(master_seed)`, and every round records its seed, so
any failure replays exactly with `--replay <round_seed>` (independent of the master
seed or round order). Enumeration order is fully deterministic (sorted roles/edge
types, dict-insertion node order, i<j weigh pairs), so the same seed reproduces the
identical action sequence.

LEGAL-SET ENUMERATION: the flat legal set is enumerated each step and one member is
chosen uniformly. For speed the enumeration Fence-A-checks only the one family that
can be rejected -- SUPPORT-edge introduces (attacks add no Support edge, NEW nodes
are isolated, extend/concede/weigh have no Fence-A, so all are legal when well
formed). This shortcut is not trusted blindly: `CDAFEnvironment.step()` independently
re-runs `check_legality` on every executed action, so if the shortcut ever admits an
illegal action the step raises and the round is reported as a failure.
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
import traceback
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

from env import (
    CDAFEnvironment, Introduce, Extend, Concede, Weigh, Connect, EndSpeech,
    NEW, is_legal,
)
from env.actions import ROLES, RELATIONSHIP_EDGE_TYPES

_ROLES = sorted(ROLES)
_REL = sorted(RELATIONSHIP_EDGE_TYPES)                 # support / defensive_attack / offensive_attack
_ATTACK_REL = [e for e in _REL if e != "support"]
_CONTENT = "x"                                         # never read by the judge; fixed for determinism

ACTION_TYPES = ("Introduce", "Extend", "Concede", "Weigh", "Connect", "EndSpeech")


def _atype(a) -> str:
    return type(a).__name__


# The legal action set factors into families whose sizes are known from the node
# count n, so we sample uniformly over the flat SUPERSET without enumerating it:
#   EndSpeech 1 | NEW-introduce 6 | attack-introduce 12n | support-introduce 6n
#   | extend n | concede n | weigh n(n-1) | connect 3n(n-1)
#   (weigh = unordered pair x 2 favors; connect = ordered pair x 3 edge_types)
# We then `is_legal`-check the ONE sampled candidate and resample on rejection.
# Rejection sampling from a uniform superset yields a uniform draw over the legal
# subset -- identical to a full enumerate-and-choose -- at O(1) legality checks per
# step. We make NO assumption about which families are rejectable (a support-introduce
# or a support-connect can fail the connect cycle rule; every other well-formed
# candidate is legal). EndSpeech is always legal, so the resample loop terminates.

def _unrank_pair(idx, n):
    """The idx-th unordered pair (i<j) over range(n), deterministically."""
    i = 0
    while True:
        cnt = n - 1 - i
        if idx < cnt:
            return i, i + 1 + idx
        idx -= cnt
        i += 1


def _unrank_ordered(idx, n):
    """The idx-th ORDERED pair (i, j) with i != j over range(n), deterministically."""
    i, rem = divmod(idx, n - 1)
    j = rem if rem < i else rem + 1
    return i, j


def superset_size(n):
    return 4 * n * n + 16 * n + 7           # 1+6+12n+6n+n+n + n(n-1) + 3n(n-1)


def available_types(n):
    """Action types with at least one candidate at node-count n (superset-level)."""
    t = {"EndSpeech", "Introduce"}          # EndSpeech + NEW-introduce always present
    if n >= 1:
        t |= {"Extend", "Concede"}
    if n >= 2:
        t |= {"Weigh", "Connect"}
    return t


def sample_legal(state, rng):
    """Uniformly sample one legal action at `state` (rejection resampling on any illegal
    candidate). Returns (action, action_type_name). Assumes not terminated."""
    ids = list(state.nodes)                 # dict-insertion order (deterministic)
    n = len(ids)
    while True:
        cand, atype = _candidate(rng.randrange(superset_size(n)), ids, n)
        if is_legal(state, cand):           # check the sampled candidate; no family assumptions
            return cand, atype
        # rejected -> resample uniformly over the whole superset


def _candidate(k, ids, n):
    """Map a superset index k in [0, superset_size(n)) to a concrete (action, type)."""
    c_end, c_new, c_atk, c_sup, c_ext, c_con = 1, 6, 12 * n, 6 * n, n, n
    c_weigh = n * (n - 1)
    if k < c_end:
        return EndSpeech(), "EndSpeech"
    k -= c_end
    if k < c_new:
        return Introduce(_CONTENT, _ROLES[k], NEW, None), "Introduce"
    k -= c_new
    if k < c_atk:
        node, w = divmod(k, 12)
        role, et = w // 2, _ATTACK_REL[w % 2]
        return Introduce(_CONTENT, _ROLES[role], ids[node], et), "Introduce"
    k -= c_atk
    if k < c_sup:
        node, role = divmod(k, 6)
        return Introduce(_CONTENT, _ROLES[role], ids[node], "support"), "Introduce"
    k -= c_sup
    if k < c_ext:
        return Extend(ids[k]), "Extend"
    k -= c_ext
    if k < c_con:
        return Concede(ids[k]), "Concede"
    k -= c_con
    if k < c_weigh:                         # weigh: unordered pair x 2 favors
        i, j = _unrank_pair(k // 2, n)
        a, b = ids[i], ids[j]
        return Weigh(a, b, b if (k % 2) else a), "Weigh"
    k -= c_weigh
    i, j = _unrank_ordered(k // 3, n)       # connect: ordered pair x 3 edge_types
    return Connect(ids[i], ids[j], _REL[k % 3]), "Connect"


def run_round(round_seed: int, *, verbose: bool = False) -> dict:
    """Play one full round with a uniform-random legal agent. Returns a result dict;
    never raises (captures any failure into the dict)."""
    rng = random.Random(round_seed)
    env = CDAFEnvironment()
    env.reset()

    log = []                                           # compact, replayable action records
    exec_counts = Counter()                            # action type -> times executed
    avail_counts = Counter()                           # action type -> steps where >=1 candidate
    legalset_empty = 0                                 # steps with an empty superset (should be 0)
    non_endspeech_empty = 0                            # steps whose only legal action is EndSpeech
    min_superset = None                                # smallest superset size seen this round
    steps = 0

    result = {"seed": round_seed, "ok": True, "error_kind": None, "error": None,
              "traceback": None, "log": log}

    try:
        while not env.state.terminated:
            n = len(env.state.nodes)
            size = superset_size(n)
            min_superset = size if min_superset is None else min(min_superset, size)
            if size == 0:                              # impossible (>=7); recorded for completeness
                legalset_empty += 1
                result.update(ok=False, error_kind="EMPTY_LEGAL_SET",
                              error=f"empty superset at step {steps}, slot {env.state.current_slot}")
                break
            types_here = available_types(n)
            for t in types_here:
                avail_counts[t] += 1
            if types_here == {"EndSpeech"}:
                non_endspeech_empty += 1

            action, atype = sample_legal(env.state, rng)
            log.append(_record(action))
            exec_counts[atype] += 1
            if verbose:
                print(f"  step {steps:2d} slot={env.state.current_slot} "
                      f"budget={env.state.remaining_budget} superset={size}  -> {_record(action)}")
            _obs, reward, done, info = env.step(action)
            steps += 1
            if done:
                result["verdict"] = info["winner"]
                result["reward"] = reward
    except AssertionError as e:
        msg = str(e)
        if "STRUCTURAL ADMISSION" in msg:
            kind = "STRUCTURAL_ADMISSION_ASSERT"
        elif "SCOPE GUARD" in msg:
            kind = "SCOPE_GUARD_ASSERT"
        else:
            kind = "OTHER_ASSERT"
        result.update(ok=False, error_kind=kind, error=msg,
                      traceback=traceback.format_exc())
    except Exception as e:                              # noqa: BLE001 -- fuzzer catches everything
        result.update(ok=False, error_kind=type(e).__name__, error=str(e),
                      traceback=traceback.format_exc())

    result.update(steps=steps, nodes=len(env.state.nodes),
                  exec_counts=dict(exec_counts), avail_counts=dict(avail_counts),
                  legalset_empty=legalset_empty, non_endspeech_empty=non_endspeech_empty,
                  min_superset=min_superset)
    return result


def _record(a):
    if isinstance(a, Introduce):
        return ("Introduce", a.role, a.target, a.edge_type)
    if isinstance(a, Extend):
        return ("Extend", a.node_id)
    if isinstance(a, Concede):
        return ("Concede", a.node_id)
    if isinstance(a, Weigh):
        return ("Weigh", a.node_a, a.node_b, a.favors)
    if isinstance(a, Connect):
        return ("Connect", a.source_id, a.target_id, a.edge_type)
    return ("EndSpeech",)


# --- reporting ---------------------------------------------------------------

def _hist(values, buckets=10):
    if not values:
        return "  (no data)"
    lo, hi = min(values), max(values)
    if lo == hi:
        return f"  all = {lo}  (n={len(values)})"
    width = (hi - lo) / buckets
    counts = Counter(min(buckets - 1, int((v - lo) / width)) for v in values)
    peak = max(counts.values()) or 1
    lines = []
    for b in range(buckets):
        b_lo = lo + b * width
        b_hi = lo + (b + 1) * width
        c = counts.get(b, 0)
        bar = "#" * int(40 * c / peak)
        lines.append(f"  [{b_lo:6.1f},{b_hi:6.1f})  {c:5d} {bar}")
    return "\n".join(lines)


def report(results, master_seed, rounds):
    failures = [r for r in results if not r["ok"]]
    lengths = [r["steps"] for r in results]
    nodes = [r["nodes"] for r in results]
    verdicts = Counter(r.get("verdict") for r in results if r.get("verdict") is not None)

    exec_total = Counter()
    avail_total = Counter()
    empty_steps = 0
    non_end_empty_steps = 0
    min_superset_overall = None
    for r in results:
        exec_total.update(r.get("exec_counts", {}))
        avail_total.update(r.get("avail_counts", {}))
        empty_steps += r.get("legalset_empty", 0)
        non_end_empty_steps += r.get("non_endspeech_empty", 0)
        ms = r.get("min_superset")
        if ms is not None:
            min_superset_overall = ms if min_superset_overall is None else min(min_superset_overall, ms)

    p = print
    p("=" * 72)
    p(f"RANDOM-AGENT CRASH TEST  master_seed={master_seed}  rounds={rounds}")
    p("=" * 72)

    p(f"\n[FAILURES]  {len(failures)} / {rounds}")
    kinds = Counter(r["error_kind"] for r in failures)
    for k, c in kinds.items():
        p(f"  {k}: {c}")
    for r in failures:
        p(f"\n  --- FAILURE  round_seed={r['seed']}  kind={r['error_kind']} ---")
        p(f"      error: {r['error']}")
        p(f"      replay: python tests/fuzz_env_shell.py --replay {r['seed']}")
        p(f"      action sequence ({len(r['log'])} actions):")
        for i, rec in enumerate(r["log"]):
            p(f"        {i:2d}: {rec}")
        if r["traceback"]:
            p("      traceback:")
            for line in r["traceback"].rstrip().splitlines():
                p(f"        {line}")

    p(f"\n[ROUND LENGTH]  (actions to termination)")
    p(f"  min={min(lengths)} max={max(lengths)} mean={statistics.mean(lengths):.1f} "
      f"median={statistics.median(lengths)}")
    p(_hist(lengths))

    p(f"\n[FINAL NODE COUNT]")
    p(f"  min={min(nodes)} max={max(nodes)} mean={statistics.mean(nodes):.1f} "
      f"median={statistics.median(nodes)}")
    p(_hist(nodes))

    p(f"\n[VERDICT DISTRIBUTION]")
    tot = sum(verdicts.values()) or 1
    for v, c in verdicts.most_common():
        p(f"  {v}: {c}  ({100*c/tot:.1f}%)")
    missing = rounds - sum(verdicts.values())
    if missing:
        p(f"  (no verdict -- round errored before termination): {missing}")

    p(f"\n[LEGAL-SET COVERAGE]")
    p(f"  steps with EMPTY legal set (should be 0): {empty_steps}")
    p(f"  steps whose ONLY legal action was EndSpeech: {non_end_empty_steps}")
    p(f"  smallest candidate-superset size seen: {min_superset_overall}   "
      f"(legal set >= 7 always: EndSpeech + 6 NEW-introduces are unconditionally legal)")
    total_steps = sum(lengths)
    p(f"  per action type -- executed / steps-available (of {total_steps} steps):")
    for t in ACTION_TYPES:
        p(f"    {t:10s}  exec={exec_total.get(t,0):7d}   available_in_steps={avail_total.get(t,0):7d}")

    p("\n" + "=" * 72)
    p("RESULT: " + ("PASS -- no exceptions, no assertion/scope-guard trips"
                    if not failures else f"FAIL -- {len(failures)} round(s) failed (see above)"))
    p("=" * 72)
    return 0 if not failures else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="Random-agent crash test for the CDAF env shell.")
    ap.add_argument("--rounds", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0, help="master seed")
    ap.add_argument("--replay", type=int, default=None,
                    help="replay a single round by its round_seed, verbose, and re-raise on failure")
    args = ap.parse_args(argv)

    if args.replay is not None:
        print(f"REPLAY round_seed={args.replay}")
        r = run_round(args.replay, verbose=True)
        print(f"\nresult: ok={r['ok']} kind={r['error_kind']} steps={r['steps']} "
              f"nodes={r['nodes']} verdict={r.get('verdict')}")
        if not r["ok"]:
            print(r["traceback"] or r["error"])
            return 1
        return 0

    master = random.Random(args.seed)
    round_seeds = [master.randrange(2**63) for _ in range(args.rounds)]
    results = []
    for i, rs in enumerate(round_seeds):
        results.append(run_round(rs))
        if (i + 1) % 50 == 0:
            fails = sum(1 for r in results if not r["ok"])
            print(f"  ... {i+1}/{args.rounds} rounds  ({fails} failures so far)", file=sys.stderr)
    return report(results, args.seed, args.rounds)


if __name__ == "__main__":
    sys.exit(main())
