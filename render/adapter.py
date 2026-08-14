"""M0 renderer — graph adapter (spec §8; RS10 / RS10b / RS10c / RS13 / RS16b).

Read-only bridge from a `model.Round` to the judge's OWN component/anchor
computation. Reuses `judge.passes` verbatim — `node_accrual` -> `resolve_chains`
for components (RS16b), `_framework_anchors` for framework reach. The ONLY
renderer-side traversal is advocacy reach (RS10b), which is `_framework_anchors`'
walk with the collector changed to Advocacy.

RS10c (v1.3): RS10b mirrors `_framework_anchors` exactly; divergence from
`_neg_offense_rooted` is expected and merely recorded (`reach_matches_rooted`),
not a failure — the judge itself disagrees about whether a BallotDirective
conducts, and BD absorbs for register.

Nothing here mutates judge / env / model state. Private judge imports are by
design (RS10 consume-only discipline). `judge/` files are untouched and
treated read-only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from model import serialize, Advocacy, BallotDirective
from model.round import Round
from model.speeches import speech_index

from judge.passes import (
    node_accrual,
    resolve_chains,
    _framework_anchors,
    _terminal_impacts,
    _neg_offense_rooted,
)
from judge.config import NEG

# Register labels (RS11 / RS13c). RS13's assert is retired in v1.4: every
# no-register component is INCOMPLETE and renders with a marker — no exceptions,
# no dependence on what else the round contains.
SUBSTANTIVE = "substantive"          # advocacy reach non-empty
FRAMEWORK = "framework"              # advocacy reach empty, framework reach non-empty
INCOMPLETE = "incomplete"           # RS13c: no register; legal unfinished argument

# Trace record kinds the renderer is permitted to consume (RS27 / RS27b).
_INVIS_KINDS = ("INERT_ATTACK", "WINDOW_CLOSED")


def load_round(path: str) -> Round:
    """Load a fixture / export through the canonical serialize path (v1->v2 aware)."""
    return serialize.load(path)


def _advocacy_reach(ctx, ch) -> Set[str]:
    """RS10b — advocacy reach: `_framework_anchors`' traversal, collector changed
    to Advocacy.

    Mirrors `judge.passes._framework_anchors` (passes.py:1303-1319) EXACTLY:
    identical seeding (`_terminal_impacts`), identical absorption (Advocacy and
    BallotDirective arrived at, not expanded through), identical support-only,
    side-agnostic walk. The SOLE difference is the collected type — Advocacy, not
    Framework. Any other divergence is drift (RS10c) and a hard failure.
    """
    reach: Set[str] = set()
    seen: Set[str] = set()
    stack = list(_terminal_impacts(ctx, ch))     # identical seed to _framework_anchors
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        node = ctx.nodes.get(cur)
        if isinstance(node, Advocacy):               # collector changed: Advocacy
            reach.add(cur)
        if isinstance(node, (Advocacy, BallotDirective)):
            continue                                 # absorbing: arrive, do not expand
        for nbr, e in ctx.adj.get(cur, []):
            if e.kind == "support" and nbr not in seen:
                stack.append(nbr)
    return reach


@dataclass
class Component:
    """One RS16b component: the chains sharing a union-find root (`members`)."""
    root: frozenset
    members: Set[str]
    chains: List[dict]
    side: str
    terminals: List[str]            # terminal impact ids, id-sorted (RS16c)
    adv_reach: Set[str]             # RS10b advocacy reach
    fw_reach: Set[str]              # _framework_anchors framework reach
    register: str                   # SUBSTANTIVE / FRAMEWORK / INCOMPLETE / UNANCHORED
    first_speech_idx: int           # RS15 first appearance
    order: List[str]                # RS16c anchor->terminal node order over members
    rooted: Optional[bool] = None            # _neg_offense_rooted (NEG only)
    reach_matches_rooted: Optional[bool] = None  # RS10c recorded divergence (NEG only)

    @property
    def multi_terminal(self) -> bool:
        return len(self.terminals) > 1

    @property
    def incomplete(self) -> bool:
        return self.register == INCOMPLETE


@dataclass
class Analysis:
    ctx: object
    chains: List[dict]
    components: List[Component]                      # RS15 order
    node2comp: Dict[str, Component]                  # member id -> its component
    invisible_edges: Dict[str, Tuple[str, str]]      # edge_id -> (kind, reason)
    round_has_advocacy: bool
    rs10c_divergences: List[Component] = field(default_factory=list)


def _rs16c_order(members: Set[str], ctx, anchor_ids: Set[str]) -> List[str]:
    """RS16c: DFS over the support-subgraph restricted to `members`, FROM the
    register anchor outward to terminal, neighbors id-ordered, each node exactly
    once. The shared spine renders once, then branches.

    Seeds are the register anchor(s) in `members` first (the advocacy for a
    substantive component, the framework for a framework one), then any remaining
    members by (speech, id). For a NEG disad whose advocacy is cross-side (not in
    `members`), `anchor_ids ∩ members` is empty and the fallback seeds the earliest
    spine root — still anchor-outward toward the terminal impact.
    """
    def key(m):
        return (speech_index(ctx.nodes[m].speech), m)

    seeds = sorted(anchor_ids & members, key=key) + sorted(members - anchor_ids, key=key)
    order: List[str] = []
    seen: Set[str] = set()
    for seed in seeds:
        if seed in seen:
            continue
        stack = [seed]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            order.append(cur)
            nbrs = sorted(
                (nbr for nbr, e in ctx.adj.get(cur, [])
                 if e.kind == "support" and nbr in members and nbr not in seen),
                reverse=True,                        # reverse -> smallest id popped first
            )
            stack.extend(nbrs)
    return order


def analyze(rnd: Round) -> Analysis:
    """Full-graph analysis (as_of=None): components, register, RS10c record,
    RS27 invisibility records. Whole-graph scope, no BD-reachability gate — the
    read-only path `env.observation` uses for Φ. No ballot is run (RS7).
    """
    trace: list = []
    ctx = node_accrual(rnd.nodes, rnd.edges, as_of=None, trace=trace)
    chains = resolve_chains(ctx, emit_trace=False)

    # RS27 / RS27b: consume ONLY the judge's own invisibility records; the renderer
    # never computes visibility. INERT_ATTACK / WINDOW_CLOSED are emitted by
    # `_classify_attacks` inside `node_accrual` (passes.py:331/338/359/374).
    invisible: Dict[str, Tuple[str, str]] = {}
    for rec in trace:
        kind = getattr(rec, "kind", None)
        if kind in _INVIS_KINDS:
            reason = getattr(rec, "reason", None) or "response window closed"
            invisible[rec.edge_id] = (kind, reason)

    round_has_advocacy = any(isinstance(n, Advocacy) for n in ctx.nodes.values())

    # RS16b: group chains into components by shared union-find root (`members`).
    groups: Dict[frozenset, List[dict]] = {}
    for ch in chains:
        groups.setdefault(frozenset(ch["members"]), []).append(ch)

    comps: List[Component] = []
    node2comp: Dict[str, Component] = {}
    divergences: List[Component] = []
    for root, chs in groups.items():
        members = set(root)
        side = ctx.nodes[next(iter(members))].side
        adv_reach: Set[str] = set()
        fw_reach: Set[str] = set()
        for ch in chs:
            adv_reach |= _advocacy_reach(ctx, ch)
            fw_reach |= _framework_anchors(ctx, ch)

        # RS11 precedence; every no-register component is RS13c INCOMPLETE (v1.4:
        # the RS13 assert is retired — staleness is provenance, not structure).
        if adv_reach:
            register = SUBSTANTIVE
        elif fw_reach:
            register = FRAMEWORK
        else:
            register = INCOMPLETE          # RS13c: legal unfinished argument

        terminals = sorted(ch["id"].split("chain:", 1)[1] for ch in chs)
        first_idx = min(speech_index(ctx.nodes[m].speech) for m in members)
        # RS16c anchor = the register anchor (advocacy for substantive, framework
        # for framework); empty for incomplete/unanchored -> seed by (speech, id).
        anchor_ids = adv_reach if adv_reach else (fw_reach if fw_reach else set())
        order = _rs16c_order(members, ctx, anchor_ids)

        comp = Component(
            root=root, members=members, chains=chs, side=side,
            terminals=terminals, adv_reach=adv_reach, fw_reach=fw_reach,
            register=register, first_speech_idx=first_idx, order=order,
        )
        # RS10c (v1.3): record divergence from _neg_offense_rooted for NEG
        # components. It is EXPECTED (BD absorbs for register; the judge conducts
        # through BD), not a failure. Recorded, never corrected.
        if side == NEG:
            comp.rooted = _neg_offense_rooted(ctx, list(members))
            comp.reach_matches_rooted = (bool(adv_reach or fw_reach) == comp.rooted)
            if not comp.reach_matches_rooted:
                divergences.append(comp)

        comps.append(comp)
        for m in members:
            node2comp[m] = comp

    # RS15: components ordered by first appearance in the round, fixed thereafter.
    # Deterministic tie-break on sorted members (id-stable).
    comps.sort(key=lambda c: (c.first_speech_idx, sorted(c.members)))

    return Analysis(
        ctx=ctx, chains=chains, components=comps, node2comp=node2comp,
        invisible_edges=invisible, round_has_advocacy=round_has_advocacy,
        rs10c_divergences=divergences,
    )


def analyze_path(path: str) -> Analysis:
    return analyze(load_round(path))
