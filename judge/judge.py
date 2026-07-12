"""judge(round) -> (ballot, trace): the deterministic CDAF verdict.

Runs the six passes in §9 order over a model.Round and returns a binary ballot
(AFF / NEG) plus the ordered decision trace. Pure: imports only model/ and the
judge package; never touches the app, filesystem, network, or clock; never
raises on a well-typed Round (§0).
"""

from __future__ import annotations

from typing import List, Tuple

from model import Advocacy, BallotDirective, Impact

from . import passes
from . import trace as T
from .config import AFF, NEG, EPSILON
from .qpn import UNRESOLVED
from .resolve import resolve


def judge(rnd) -> Tuple[str, List]:
    """Evaluate a finished argument graph. Returns (ballot, trace). Passes run in
    §9 order: weighing towers resolve BEFORE clash resolution, so a determinate
    weigh can decide a link/turn polarity clash without a dependency cycle."""
    ctx = passes.build_context(rnd)        # Pass 1: discovery + weighing index
    passes.pass2_drops(ctx)                # Pass 2: drop + extension
    passes.pass3_accrual(ctx)              # Pass 3: DF-QuAD sigma (no polarity yet)
    passes.pass4_weighing_towers(ctx)      # Pass 4: resolve weighing towers (§6.5)
    passes.pass5_clashes(ctx)              # Pass 5: polarity (via resolve) + chains
    passes.pass6_framework(ctx)            # Pass 6: framework gate
    winner = _ballot(ctx)                  # Pass 7: BD validation + net offense
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
    a structural failure.

    "framework lock-out" is RESERVED for the case where NEG wins FOR WANT of AFF
    offense (v6): a framework governs, AFF has no in-scope impact, AND NEG has no
    offense of its own (N >= -EPSILON). When NEG carries in-scope offense
    (N < -EPSILON) the reason is "NEG offense", even if a framework also shut AFF
    out -- lock-out and NEG-offense are otherwise the same judged state (a
    governing framework with AFF out of scope), and the tabula-rasa judge cannot
    tell "AFF was locked out" from "NEG's offense won" except by whether NEG
    actually has offense. This narrows the lock-out branch; it never widens it (a
    wash, winning_framework is None, still never locks out)."""
    if winner == AFF:
        return "AFF offense"
    if ctx.winning_framework is not None and not inscope_impact and N >= -EPSILON:
        return "framework lock-out"
    if N < -EPSILON:
        return "NEG offense"
    # AFF structural failure (§7, redrawn): AFF ESTABLISHED offense that then
    # failed -- a COMPLETE, EXTENDED, still-AFF-favoring chain (sign +1, resolved)
    # existed but was driven to zero magnitude or gated out of scope, so it reached
    # the ballot contributing nothing. "advocacy present" is too coarse a proxy for
    # "offense established" (it fires for a bare advocacy or an unresolved impact),
    # so we test for the chain itself. A chain merely TURNED to the opponent (sign
    # flipped) is NOT a structural failure: turned offense becomes the opponent's --
    # scored if that side anchored a BD (r4), else orphaned to presumption (r10) --
    # so a flip never collapses AFF's own offense. When such a chain exists AND is
    # complete/in-scope but the round nets to a tie (|N| <= eps), that is presumption
    # (offense reached the ballot; it just did not prevail), not structural failure.
    aff_established = any(ch["side"] == AFF and ch["extended"]
                         and not ch["unresolved"] and ch["sign"] == 1
                         for ch in ctx.chains)
    if aff_established and not (advocacy_present and complete_chain and inscope_impact):
        return "AFF structural failure"
    return "presumption"


def _weighing_excluded(ctx: passes.Context, valid: list) -> set:
    """Rank surviving offense with the recursive clash-breaker (§6.5): for each
    impact-pair weigh, `resolve` says whether it determinately decides; if so, the
    dispreferred impact's chain is excluded from the tally (preference overrides
    raw delta). Indeterminate weighs leave both chains in (fall to raw delta)."""
    impact_to_chain = {}
    for ch in valid:
        for imp in ch["impacts"]:
            impact_to_chain[imp] = ch["id"]
        for m in ch["members"]:
            impact_to_chain.setdefault(m, ch["id"])

    excluded = set()
    for w in ctx.weighings:
        pair = ctx.weigh_pair.get(w.id)
        if not pair or not all(isinstance(ctx.nodes[m], Impact) for m in pair):
            continue                            # ballot ranking is over impact clashes
        determinate, winner, _ = resolve(ctx, pair)
        if determinate:
            for member in pair:
                if member != winner and member in impact_to_chain:
                    excluded.add(impact_to_chain[member])
    return excluded
