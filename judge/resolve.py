"""Recursive weighing clash-resolution (§6.5) -- the general clash-breaker.

Weighing is not impact-only. It is the general clash-breaker over ANY two
same-type nodes (two impacts, two links = a turn's polarity clash, two
uniquenesses, two frameworks, ...). It never edits a node's strength; it decides
*how to break a clash*. Magnitude (DF-QuAD sigma / raw delta) is only the
fallback for a clash the debaters did not resolve.

The rule is ONE recursion at every depth (never a fixed depth-2 check):

    resolve(clash):
        P = the weighing layer immediately above this clash
        if P has exactly ONE surviving preference:   -> that preference decides
        else (P empty or tied -> indeterminate):      -> caller uses magnitude

    a preference "survives" iff its weighing is not dropped (won its own
    accrual: extended and sigma >= threshold) AND not defeated by a higher
    weighing clash -- itself resolve()'d one level up (meta-weighing over
    weighing, and so on).

Levels top-to-bottom: meta-weighing -> weighing -> magnitude. Magnitude is the
base case / floor: no level above, always a comparison, never punts upward, so
the recursion terminates. Well-founded: each step climbs to strictly higher
weighing nodes in a finite graph, and a visited-guard blocks any malformed
meta-cycle.

ANTI-CYCLE (§9): resolve reads ONLY the weighing tower -- weighing sigma/extension
(from accrual), sides, and Comparison pairs. It NEVER reads a main-chain link's
effective polarity. So the towers settle independently, before the clash
resolution that consumes them; there is no weighing<->polarity cycle. If this
module ever needs a main-chain polarity, the wiring is wrong.
"""

from __future__ import annotations

from typing import FrozenSet, Optional, Tuple

from model import Comparison, Weighing

from .config import POLARITY_THRESHOLD, TAU


def weigh_pair(ctx, w_id: str) -> Optional[FrozenSet]:
    """The unordered pair (frozenset of exactly two node ids) a weighing RANKS,
    or None if it is not a well-formed pairwise weigh.

    A Comparison edge points FROM the ranking weighing TO a member of the pair
    (source = the weighing). This direction is load-bearing here -- unlike
    support/attack (§2.2) -- because it is what separates 'this weigh ranks {L,T}'
    from 'a meta-weigh ranks this weigh': the meta's edge has the meta as source,
    so it does not pollute this weigh's pair."""
    members = [nbr for nbr, e in ctx.adj.get(w_id, [])
               if isinstance(e, Comparison) and e.source == w_id and nbr in ctx.nodes]
    members = sorted(set(members))
    return frozenset(members) if len(members) == 2 else None


def preferred_node(ctx, w) -> Optional[str]:
    """The pair member a weighing prefers = the one on the weighing's own side
    (an AFF weigh of {AFF-link, NEG-turn} prefers the AFF link). None if the pair
    is not exactly one own-side node (e.g. two same-side nodes -> no directional
    preference is derivable from the model, so the weigh is treated as inert)."""
    pair = ctx.weigh_pair.get(w.id)
    if not pair:
        return None
    same = [m for m in pair if ctx.nodes[m].side == w.side]
    return same[0] if len(same) == 1 else None


def _won_accrual(ctx, w) -> bool:
    """The weighing survived its OWN sub-debate: extended (§6) and not attacked
    down (sigma >= threshold). A conceded (unanswered) weigh stands at sigma 1."""
    from .passes import node_extension_ok
    return ctx.sigma.get(w.id, TAU) >= POLARITY_THRESHOLD and node_extension_ok(w)[0]


def _weighs_over(ctx, pair: FrozenSet):
    """The weighing layer immediately above `pair`: weighings whose Comparison
    pair is exactly `pair`."""
    return [w for w in ctx.weighings if ctx.weigh_pair.get(w.id) == pair]


def resolve(ctx, pair: FrozenSet, _stack: FrozenSet = frozenset()) -> Tuple[bool, Optional[str], Optional[str]]:
    """Resolve the clash over `pair` (§6.5). Returns
    (determinate, winner_node_id, deciding_weigh_id).

      determinate=True  -> the layer above has exactly one surviving preference;
                           winner_node_id is that preferred node.
      determinate=False -> indeterminate (layer absent or tied); the CALLER
                           breaks the clash by magnitude (sigma threshold for a
                           polarity clash, raw delta for an impact clash).
    """
    if pair in _stack:                      # malformed meta-cycle guard (well-foundedness)
        return (False, None, None)
    stack = _stack | {pair}

    survivors = []                          # (weigh_id, preferred_node)
    preferences = set()
    for w in _weighs_over(ctx, pair):
        if _survives(ctx, w, stack):
            p = preferred_node(ctx, w)
            if p is not None:
                survivors.append((w.id, p))
                preferences.add(p)

    if len(preferences) == 1:               # determinate: a lone surviving preference
        winner = next(iter(preferences))
        decider = next(wid for wid, p in survivors if p == winner)
        return (True, winner, decider)
    return (False, None, None)              # indeterminate -> magnitude fallback (caller)


def _survives(ctx, w, stack: FrozenSet) -> bool:
    """A weighing survives iff it won its own accrual AND is not defeated by a
    higher weighing clash -- the meta-clash {w, w2} resolve()'d one level up. A
    higher clash defeats w only when it is DETERMINATE and decides against w; an
    indeterminate meta-clash leaves both weighs standing (so the clash below it
    stays tied -> magnitude)."""
    if not _won_accrual(ctx, w):
        return False
    for w2 in ctx.weighings:
        if w2.id == w.id:
            continue
        meta_pair = frozenset({w.id, w2.id})
        if not _weighs_over(ctx, meta_pair):        # no meta-weigh over {w, w2}
            continue
        det, winner, _ = resolve(ctx, meta_pair, stack)
        if det and winner == w2.id:                 # a higher weigh decided against w
            return False
    return True
