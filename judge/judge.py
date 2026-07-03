"""judge(round) -> (ballot, trace): the deterministic CDAF verdict.

Runs the six passes in §9 order over a model.Round and returns a binary ballot
(AFF / NEG) plus the ordered decision trace. Pure: imports only model/ and the
judge package; never touches the app, filesystem, network, or clock; never
raises on a well-typed Round (§0).
"""

from __future__ import annotations

from typing import List, Tuple

from model import Advocacy, BallotDirective

from . import passes
from . import trace as T
from .config import AFF, NEG, EPSILON
from .qpn import UNRESOLVED


def judge(rnd) -> Tuple[str, List]:
    """Evaluate a finished argument graph. Returns (ballot, trace)."""
    ctx = passes.build_context(rnd)        # Pass 1: discovery + indexing
    passes.pass2_drops(ctx)                # Pass 2: drop detection
    passes.pass3_resolve(ctx)              # Pass 3: accrual, polarity, chains (+ extension)
    passes.pass5a_framework(ctx)           # Pass 5a: framework gate
    passes.pass5b_weighing(ctx)            # Pass 5b: weighing preferences
    winner = _ballot(ctx)                  # Pass 6: BD validation + net offense
    return winner, ctx.trace


def _favored_side(ctx: passes.Context, ch: dict):
    """Which side an anchored argument resolves in favor of, or None if it
    establishes no offense (unresolved sign, failed extension, out of scope, or
    collapsed magnitude)."""
    if ch["sign"] == UNRESOLVED or ch["unresolved"]:
        return None
    if not ch["extended"] or not ch.get("in_scope", True):
        return None
    if ch["mag"] <= EPSILON:
        return None
    return ch["side"] if ch["sign"] > 0 else passes._opposing(ch["side"])


def _ballot(ctx: passes.Context) -> str:
    """Pass 6 (§7): validate BDs, sum net offense with weighing preferences,
    apply the asymmetric win condition, drain the indeterminate to presumption."""
    valid = []          # chains that passed BD validation
    valid_ids = set()
    bds = [n for n in ctx.nodes.values()
           if isinstance(n, BallotDirective) and n.id in ctx.reachable]

    for bd in bds:
        incident = {nbr for nbr, _e in ctx.adj.get(bd.id, [])}
        anchored = [ch for ch in ctx.chains if ch["members"] & incident]
        if not anchored:
            ctx.trace.append(T.BdValidate(bd_id=bd.id, result="fail",
                                          reason="no anchored argument", side=bd.side))
            continue
        for ch in anchored:
            fav = _favored_side(ctx, ch)
            if fav is None:
                result, reason = "fail", "anchored argument establishes no offense ('?')"
            elif fav != bd.side:
                result, reason = "fail", f"offense favors {fav}, BD claims {bd.side}"
            else:
                result, reason = "pass", f"resolves {bd.side}"
                if ch["id"] not in valid_ids:
                    valid.append(ch)
                    valid_ids.add(ch["id"])
            ctx.trace.append(T.BdValidate(bd_id=bd.id, result=result, reason=reason, side=bd.side))

    excluded = _weighing_excluded(ctx, valid)

    # Net offense N = sum(AFF deltas) - sum(NEG deltas), over validated chains,
    # honoring won-weighing preferences. delta already carries any polarity flip.
    contributed_ids = {ch["id"] for ch in valid if ch["id"] not in excluded}
    aff_sum = neg_sum = 0.0
    decomposition = []
    for ch in ctx.chains:
        contributed = ch["id"] in contributed_ids
        d = 0.0 if ch["sign"] == UNRESOLVED else ch["sign"] * ch["mag"]
        decomposition.append({"chain_id": ch["id"], "side": ch["side"],
                              "delta": d, "contributed": contributed})
        if contributed:
            if ch["side"] == AFF:
                aff_sum += d
            else:
                neg_sum += d
    N = aff_sum - neg_sum

    # AFF structural gates
    aff_valid = [ch for ch in valid if ch["side"] == AFF and ch["id"] not in excluded]
    advocacy_present = any(isinstance(ctx.nodes[m], Advocacy)
                           for ch in aff_valid for m in ch["members"])
    complete_chain = any(ch["mag"] > EPSILON for ch in aff_valid)
    inscope_impact = any(ch.get("in_scope", True) and ch["impacts"] for ch in aff_valid)

    gates = []
    if advocacy_present:
        gates.append("advocacy")
    if complete_chain:
        gates.append("complete_chain")
    if inscope_impact:
        gates.append("in_scope_impact")
    if N > EPSILON:
        gates.append("N>eps")

    aff_wins = advocacy_present and complete_chain and inscope_impact and N > EPSILON
    winner = AFF if aff_wins else NEG     # everything indeterminate drains to NEG presumption
    reason_class = _reason_class(ctx, winner, advocacy_present, complete_chain,
                                 inscope_impact, N)
    ctx.trace.append(T.Ballot(N=N, gates_passed=gates, winner=winner,
                              reason_class=reason_class, aff_sum=aff_sum,
                              neg_sum=neg_sum, decomposition=decomposition))
    return winner


def _reason_class(ctx, winner, advocacy_present, complete_chain, inscope_impact, N):
    """Label the decision (descriptive only -- does not change the verdict).

    "AFF structural failure" means AFF mounted a case that fell short of a gate;
    a round with no AFF case at all (e.g. an empty graph) is "presumption", not
    a structural failure."""
    if winner == AFF:
        return "AFF offense"
    if ctx.winning_framework is not None and not inscope_impact:
        return "framework lock-out"
    if N < -EPSILON:
        return "NEG offense"
    aff_attempted = (any(isinstance(n, Advocacy) for n in ctx.nodes.values())
                     or any(ch["side"] == AFF for ch in ctx.chains))
    if aff_attempted and not (advocacy_present and complete_chain and inscope_impact):
        return "AFF structural failure"
    return "presumption"


def _weighing_excluded(ctx: passes.Context, valid: list) -> set:
    """Resolve won-weighing preferences to a set of excluded chain ids: a
    preference for one member of a compared pair excludes chains anchored on the
    dispreferred member (preference overrides raw delta, §7)."""
    impact_to_chain = {}
    for ch in valid:
        for imp in ch["impacts"]:
            impact_to_chain[imp] = ch["id"]
        for m in ch["members"]:
            impact_to_chain.setdefault(m, ch["id"])

    excluded = set()
    for _side, preferred, pair in ctx.preferences:
        if preferred is None:
            continue
        for member in pair:
            if member != preferred and member in impact_to_chain:
                excluded.add(impact_to_chain[member])
    return excluded
