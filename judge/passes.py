"""The six passes (§9), operating on a model.Round.

Each pass mutates a shared `Context` and appends J1 trace records. The passes
call the J2 channels (dfquad, qpn, chain) for all numeric work and never
re-implement the channel math.

Direction-agnostic throughout (§2.2): edges are read as undirected connectivity.
Chains are oriented by node TYPE (Uniqueness root, Impact terminal, BD sink);
clashes and re-assertions are oriented by SPEECH RECENCY (the later-speech node
is the attacker / the re-assertion). Edge `source`/`target` direction is never
load-bearing.

V1 modeling choices, grounded in the real authoring convention where a
`SupportEdge` spine runs Advocacy -> Uniqueness -> ... -> Link -> Impact -> BD:

  * Support edges are SPINE CONNECTIVITY and the magnitude channel (their node
    strengths multiply along the chain). They are NOT fed into DF-QuAD accrual
    as supporters -- doing so would let a full spine support cancel a conceded
    attacker, contradicting the §11 oracles (conceded defense -> link sigma 0).
    DF-QuAD accrual at a node therefore uses ATTACKS ONLY in V1; restoration of
    a contested node comes from answering its attacker (lowering the attacker's
    sigma), exactly as oracles 2-3 describe. The {s_k} supporter slot is kept in
    the math (dfquad.accrue) for a future authoring convention that draws
    explicit restorative supports distinct from the spine.
  * A "chain" is a same-side connected component over Support edges that
    contains an Impact; the chain magnitude is the product of its spine nodes'
    sigmas, the sign the product of link/impact effective polarities.

EXTENSION IS READ FROM LIVENESS (Model C, §6), not from edges. Each node carries
a `liveness` record (speech -> contested/conceded). A spine node is extended iff
its record covers every one of its own side's speeches from introduction onward
(node_extension_ok). There is no ExtensionEdge and no re-assertion-duplicate
collapsing; each spine node is a single object.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from model import (
    Node, Edge,
    Uniqueness, Link, Impact, Advocacy, Framework, Weighing, BallotDirective,
    Support, DefensiveAttack, OffensiveAttack, Comparison,
)

from . import dfquad, qpn, chain as chainmod
from . import trace as T
from .config import (
    TAU, POLARITY_THRESHOLD, EPSILON, AFF, NEG, SPEECH_ORDER, SPEECH_SIDE,
)

ATTACK_TYPES = (DefensiveAttack, OffensiveAttack)
SPINE_TYPES = (Uniqueness, Link, Impact, Advocacy)
OFFENSE_BEARING = (Link, Impact)
# Rebuttal speeches: a NEW chain first introduced here does not count (§6).
REBUTTAL_SPEECHES = frozenset({"1AR", "2NR", "2AR"})


# --- small speech/side helpers ------------------------------------------------

def _sidx(speech: str) -> Optional[int]:
    """Index in SPEECH_ORDER, or None for an unknown speech string."""
    try:
        return SPEECH_ORDER.index(speech)
    except ValueError:
        return None


def _opposing(side: str) -> str:
    return NEG if side == AFF else AFF


def side_speeches(side: str) -> List[str]:
    """The speeches owned by `side`, in order."""
    return [s for s in SPEECH_ORDER if SPEECH_SIDE.get(s) == side]


def response_window(speech: str, side: str) -> Optional[str]:
    """The node's response window (§4): the next speech in SPEECH_ORDER owned by
    the opposing side after `speech`. None if no later opposing speech exists
    (the node was introduced too late for the opponent to answer)."""
    i = _sidx(speech)
    if i is None:
        return None
    opp = _opposing(side)
    for j in range(i + 1, len(SPEECH_ORDER)):
        if SPEECH_SIDE.get(SPEECH_ORDER[j]) == opp:
            return SPEECH_ORDER[j]
    return None


# --- context ------------------------------------------------------------------

@dataclass
class Context:
    round: object
    trace: List = field(default_factory=list)

    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)
    adj: Dict[str, List[Tuple[str, Edge]]] = field(default_factory=lambda: defaultdict(list))

    reachable: set = field(default_factory=set)

    attackers_by_target: Dict[str, List[Tuple[str, Edge]]] = field(default_factory=lambda: defaultdict(list))
    offense_on: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))

    status: Dict[str, str] = field(default_factory=dict)          # node -> answered/dropped/unresolved
    sigma: Dict[str, float] = field(default_factory=dict)         # raw DF-QuAD surviving strength
    mag_sigma: Dict[str, float] = field(default_factory=dict)     # sigma used in the chain magnitude
    eff_pol: Dict[str, object] = field(default_factory=dict)      # link/impact id -> +1/-1/'?'
    chains: List[dict] = field(default_factory=list)
    winning_framework: Optional[Node] = None

    # Weighing towers (§6.5): the reachable Weighing nodes and the pair each one
    # ranks. Populated in build_context; consumed by judge.resolve.
    weighings: List = field(default_factory=list)
    weigh_pair: Dict[str, object] = field(default_factory=dict)   # weigh id -> frozenset(pair)


# --- Pass 1: discovery + structural indexing ----------------------------------

def build_context(rnd) -> Context:
    """Pass 1 (§4): index the round, walk the undirected BD-anchored subgraph,
    and classify attacks by speech recency."""
    ctx = Context(round=rnd)
    ctx.nodes = {n.id: n for n in rnd.nodes}
    ctx.edges = list(rnd.edges)

    adj: Dict[str, List[Tuple[str, Edge]]] = defaultdict(list)
    for e in ctx.edges:
        if e.source in ctx.nodes and e.target in ctx.nodes:   # ignore dangling refs (robustness)
            adj[e.source].append((e.target, e))
            adj[e.target].append((e.source, e))
    ctx.adj = adj

    _discover(ctx)
    _classify_attacks(ctx)
    _index_weighings(ctx)
    return ctx


def _index_weighings(ctx: Context) -> None:
    """Index the reachable weighing tower: the Weighing nodes and the pair each
    ranks (from its Comparison edges). Structural only -- no strengths yet."""
    from .resolve import weigh_pair
    ctx.weighings = [n for n in ctx.nodes.values()
                     if isinstance(n, Weighing) and n.id in ctx.reachable]
    ctx.weigh_pair = {w.id: weigh_pair(ctx, w.id) for w in ctx.weighings}


def _discover(ctx: Context) -> None:
    """Undirected reachability from every BallotDirective (§4, Pass 1)."""
    seen: set = set()
    bds = [n.id for n in ctx.nodes.values() if isinstance(n, BallotDirective)]
    for start in bds:
        stack = [start]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            for nbr, _e in ctx.adj.get(cur, []):
                if nbr not in seen:
                    stack.append(nbr)
    ctx.reachable = seen


def _union_find(ids):
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    return find, union, parent


def _prior_opposing_speech(speech: str, side: str) -> Optional[str]:
    """The most recent speech BEFORE `speech` owned by the opposing side -- the
    'immediately prior opposing speech' the §4 final-speech refinement reads."""
    i = _sidx(speech)
    if i is None:
        return None
    opp = _opposing(side)
    for j in range(i - 1, -1, -1):
        if SPEECH_SIDE.get(SPEECH_ORDER[j]) == opp:
            return SPEECH_ORDER[j]
    return None


def _classify_attacks(ctx: Context) -> None:
    """Orient each attack edge by speech recency (§2.2): the later-speech node is
    the attacker, the earlier is the target. Inert attacks (§3.4) are dropped and
    recorded. Direction of the drawn edge is ignored."""
    attackers_by_target: Dict[str, List[Tuple[str, Edge]]] = defaultdict(list)
    for e in ctx.edges:
        if not isinstance(e, ATTACK_TYPES):
            continue
        a = ctx.nodes.get(e.source)
        b = ctx.nodes.get(e.target)
        if a is None or b is None:
            continue
        ia, ib = _sidx(a.speech), _sidx(b.speech)
        if ia is None or ib is None:
            ctx.trace.append(T.InertAttack(edge_id=e.id, reason="unorderable speech"))
            continue
        if ia > ib:
            attacker, target = a, b
        elif ib > ia:
            attacker, target = b, a
        else:
            ctx.trace.append(T.InertAttack(edge_id=e.id, reason="same-speech clash (incoherent)"))
            continue

        reason = None
        if attacker.side == target.side:
            reason = "same-side attack (incoherent)"
        elif isinstance(e, OffensiveAttack) and not (
                isinstance(attacker, OFFENSE_BEARING) and isinstance(target, OFFENSE_BEARING)):
            # Turn-eligibility (§3.4): an OffensiveAttack is a competing-polarity
            # claim; it can only flip a node that carries polarity -- a Link or an
            # Impact. If EITHER endpoint is a Uniqueness/Advocacy/Framework/Weighing/
            # BD there is no polarity to flip and the edge is inert. This one rule
            # subsumes every enumerated inert case (offense at a uniqueness, an
            # advocacy, a framework, ...). It is DIRECTION-AGNOSTIC (§2.2): keyed on
            # the two endpoint types, never on which end is `source`. DefensiveAttack
            # is NOT governed by this -- it lowers magnitude and never flips, so it
            # stays coherent against a Framework's sigma or a link's (the non-unique);
            # only OffensiveAttack is guarded here.
            bad = attacker if not isinstance(attacker, OFFENSE_BEARING) else target
            reason = f"offense requires offense-bearing endpoints; {bad.ntype} bears no polarity (§3.4)"
        if reason:
            ctx.trace.append(T.InertAttack(edge_id=e.id, reason=reason))
            continue

        # Attacker-liveness gate (§3.1, §6 -- v4). An attack contributes to its
        # target's accrual ONLY while the attack itself is LIVE -- extended by its
        # maker (side-agnostic union, node_live_by_any_side, so a turn kept live by
        # either side still counts). An attack its maker ABANDONED lapses: it is
        # removed from the target's attacker set (and offense set) and contributes
        # nothing. It is NOT scored "conceded" merely because the opposing side did
        # not answer it -- concession/full strength holds only for a LIVE attack.
        # This does not touch the mitigation path: an attack the TARGET answered is
        # still live (the maker extended it) and is reduced by its own attacker via
        # the leaves-first DF-QuAD recursion below, exactly as before.
        live, _ = node_live_by_any_side(attacker)
        if not live:
            _, maker_missing = node_extension_ok(attacker)   # maker-side gap, for the message
            ctx.trace.append(T.InertAttack(
                edge_id=e.id,
                reason=f"lapsed: attack not extended by its maker (missing {maker_missing})"))
            continue

        attackers_by_target[target.id].append((attacker.id, e))
        if isinstance(e, OffensiveAttack) and isinstance(target, OFFENSE_BEARING):
            ctx.offense_on[target.id].append(attacker.id)
    ctx.attackers_by_target = attackers_by_target


# --- Pass 2: drop detection + extension ---------------------------------------

def pass2_drops(ctx: Context) -> None:
    """Drop detection (§4): per reachable node, classify answered / dropped /
    unresolved against its response window. The §4 final-speech refinement reads
    liveness status. Extension (§6) is checked per node from the liveness record
    in pass 3 (node_extension_ok)."""
    for nid, n in ctx.nodes.items():
        if nid not in ctx.reachable:
            continue
        if isinstance(n, (BallotDirective, Weighing, Framework)):
            continue  # structural / sub-debate nodes are not offense nodes
        win = response_window(n.speech, n.side)
        if win is None:
            # No window: introduced in the final speech of its side. Baseline =
            # UNRESOLVED (inert). Refinement (§4): it engages only if it continues
            # a clash that was CONTESTED entering that speech -- read off the
            # attachment point's liveness for the immediately prior opposing
            # speech. Conceded-live is NOT enough; a fresh spike is inert.
            prior_opp = _prior_opposing_speech(n.speech, n.side)
            continues = prior_opp is not None and any(
                (m.liveness or {}).get(prior_opp) == "contested"
                for nbr, _e in ctx.adj.get(nid, [])
                for m in [ctx.nodes.get(nbr)] if m is not None
            )
            if continues:
                ctx.status[nid] = "answered"   # legitimate continuation -> resolves normally
            else:
                ctx.status[nid] = "unresolved"
                ctx.trace.append(T.Unresolved(node_id=nid, owner=n.side, intro_speech=n.speech))
            continue
        answered = False
        for nbr, e in ctx.adj.get(nid, []):
            m = ctx.nodes.get(nbr)
            if m is not None and isinstance(e, ATTACK_TYPES) \
                    and m.side == _opposing(n.side) and m.speech == win:
                answered = True
                break
        if answered:
            ctx.status[nid] = "answered"
        else:
            ctx.status[nid] = "dropped"
            ctx.trace.append(T.Drop(node_id=nid, owner=n.side, intro_speech=n.speech, window_speech=win))


def node_extension_ok(node: Node, carrying_side: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """§6, read from the LIVENESS record: a spine node is extended iff its
    liveness covers every one of the CARRYING side's speeches from its
    introduction onward. Returns (ok, missing_speech).

    LIVENESS IS SIDE-AGNOSTIC (§6): a node's record is the UNION of every party's
    extensions through it, so a node kept live by the opponent is live. This reads
    the record, never edges. `carrying_side` is the side whose speech schedule the
    node must be extended through; it defaults to the node's own introducing side
    (the ordinary, non-turned case). For a TURNED chain a node need only be live by
    *someone* -- that union check is `node_live_by_any_side`, which calls this with
    each side in turn. The "no new chains in rebuttals" rule is applied at the chain
    level, so a weighing/framework introduced in a rebuttal is not penalized here."""
    liveness = node.liveness or {}
    intro_idx = _sidx(node.speech)
    if intro_idx is None:
        return True, None
    for s in side_speeches(carrying_side or node.side):
        if _sidx(s) >= intro_idx and s not in liveness:
            return False, s
    return True, None


def node_live_by_any_side(node: Node) -> Tuple[bool, Optional[str]]:
    """§6 side-agnostic union: is this node live -- carried by ANY side? A turned
    chain's nodes count iff their (union) liveness record is sustained by someone,
    NOT specifically by the turning side (spec §6). This yields both turn win-paths
    with no special-casing (§3.5): (a) the introducing side keeps the link/impact
    live while the other side carries the turn, or (b) the turning side carries the
    whole chain -- either way the node was kept live by *someone*. A node no side
    kept alive (a dead impact) fails. Returns (ok, missing_speech); the reported
    gap is the opponent-side gap (the carrying side one would expect for a turn),
    falling back to the own-side gap."""
    ok_own, miss_own = node_extension_ok(node, carrying_side=node.side)
    if ok_own:
        return True, None
    ok_opp, miss_opp = node_extension_ok(node, carrying_side=_opposing(node.side))
    if ok_opp:
        return True, None
    return False, miss_opp or miss_own


# --- Pass 3: node accrual (sigma only -- NO polarity yet, §9) -----------------

def pass3_accrual(ctx: Context) -> None:
    """Pass 3 (§9): DF-QuAD surviving strength per node. No polarity here -- that
    is set in clash resolution (pass 5), after the weighing towers settle."""
    _resolve_strengths(ctx)


# --- Pass 4: weighing towers resolve first (§6.5 / §9 anti-cycle) -------------

def pass4_weighing_towers(ctx: Context) -> None:
    """Resolve each weighing sub-debate via the recursive clash-breaker (§6.5) and
    emit a WEIGH record. These read ONLY the weighing nodes' own accrual
    (sigma/extension) and Comparison pairs -- never main-chain polarity -- so they
    settle before the clash resolution that consumes them and cannot cycle (§9)."""
    from .resolve import resolve
    for w in ctx.weighings:
        pair = ctx.weigh_pair.get(w.id)
        if pair is None:
            ctx.trace.append(T.Weigh(weighing_id=w.id, outcome="symmetric",
                                     preferred_node=None, via="malformed", pair=[]))
            continue
        determinate, winner, _decider = resolve(ctx, pair)
        if determinate:
            ctx.trace.append(T.Weigh(weighing_id=w.id, outcome="resolved",
                                     preferred_node=winner, via="preference",
                                     pair=sorted(pair)))
        else:
            ctx.trace.append(T.Weigh(weighing_id=w.id, outcome="symmetric",
                                     preferred_node=None, via="magnitude",
                                     pair=sorted(pair)))

    # The towers are resolved -- now CONSUME them against accrual (§6.5, v5): a
    # determinate weigh defeats the dispreferred member of a clash, and a defeated
    # attacker does not attack the winner. This is the same rule the polarity
    # channel applies to a defeated turn (§3.2), here applied to every attacker so
    # a won uniqueness-weigh (or framework/defensive weigh) actually removes the
    # defeated non-unique from the winner's DF-QuAD accrual. Runs after the towers
    # settle (they read only their own sub-debate, §9) and re-accrues below.
    if _apply_weigh_defeat(ctx):
        ctx.trace[:] = [r for r in ctx.trace if r.kind != "MAGNITUDE"]
        _resolve_strengths(ctx)   # recompute sigma with defeated attackers removed


def _apply_weigh_defeat(ctx: Context) -> bool:
    """Drop every attacker DEFEATED by a determinate weigh over its clash with its
    target: resolve({attacker, target}) determinate AND winner is the target -> the
    attacker forfeits and is removed from the target's attacker set (it no longer
    reduces the winner's accrual). Emits INERT_ATTACK 'defeated by weigh'. Returns
    True if anything was pruned.

    `offense_on` is intentionally left untouched: a defeated TURN on an offense-
    bearing link is still consumed by the polarity channel (_resolve_polarity),
    which reads offense_on to emit POLARITY_FLIP and preserve magnitude (§3.2).
    Removing the turn here from `attackers_by_target` only zeroes its magnitude
    contribution, which the polarity channel already did -- so the two agree and
    the v2 link-weigh behavior (r9) is unchanged."""
    from .resolve import resolve
    pruned = False
    for target, atts in list(ctx.attackers_by_target.items()):
        keep = []
        for a, e in atts:
            if a != target:
                determinate, winner, _ = resolve(ctx, frozenset({a, target}))
                if determinate and winner == target:
                    ctx.trace.append(T.InertAttack(edge_id=e.id, reason="defeated by weigh"))
                    pruned = True
                    continue
            keep.append((a, e))
        ctx.attackers_by_target[target] = keep
    return pruned


# --- Pass 5: clash resolution (polarity via resolve) + chain products ---------

def pass5_clashes(ctx: Context) -> None:
    """Pass 5 (§9): set effective polarity by resolving each link/turn clash
    (§3.2, consuming a determinate weigh; else the 0.5 sigma threshold), then
    build chain sign/magnitude/delta (§3.3)."""
    _resolve_polarity(ctx)
    _build_chains(ctx)


def _resolve_strengths(ctx: Context) -> None:
    """DF-QuAD accrual per node (§3.1), leaves first. Implemented as a damped
    fixpoint over the reachable nodes so resolution order (attacker before
    target) and any noisy cycles converge deterministically. V1 feeds ATTACKS
    ONLY into DF-QuAD (see module docstring)."""
    sigma = {nid: TAU for nid in ctx.reachable}
    order = sorted(ctx.reachable, key=lambda nid: -(_sidx(ctx.nodes[nid].speech) or 0))
    for _ in range(len(order) + 2):
        changed = False
        for nid in order:
            atts = [sigma.get(a, TAU) for a, _e in ctx.attackers_by_target.get(nid, [])
                    if a in ctx.reachable]
            new = dfquad.accrue(TAU, atts, [])   # supporters reserved; empty in V1
            if abs(new - sigma[nid]) > 1e-12:
                sigma[nid] = new
                changed = True
        if not changed:
            break
    ctx.sigma = sigma
    ctx.mag_sigma = dict(sigma)   # default; clash resolution may drop a defeated turn

    for nid in ctx.reachable:
        attacks = ctx.attackers_by_target.get(nid, [])
        if attacks:   # only emit for nodes that actually accrued something
            ctx.trace.append(T.Magnitude(
                node_id=nid, base_tau=TAU, surviving_sigma=sigma[nid],
                attackers=[a for a, _e in attacks], supporters=[],
            ))


def _resolve_polarity(ctx: Context) -> None:
    """Effective polarity per offense-bearing node (§3.2), CONSUMING a determinate
    weigh over the link/turn clash (§6.5).

    THE FLIP IS GATED ON THE PRESENCE OF AN OFFENSIVE ATTACK. Only a link that is
    the target of an OffensiveAttack (in ctx.offense_on) is eligible to flip. For
    such a link the {link, turn} clash is resolved by `resolve`:
      * a DETERMINATE won link-weigh decides the sign outright (via 'preference',
        regardless of raw sigma): winner == link keeps +1, winner == turn flips;
      * absent/indeterminate -> the 0.5 sigma threshold (via 'dfquad'), as before,
        reading ctx.sigma (which INCLUDES the offensive attacker -- that is what
        pushes sigma below 0.5 and triggers the flip).
    A defensively-only link keeps +1 no matter how low sigma falls (defense
    reduces magnitude, never reverses direction).

    MAGNITUDE PRESERVATION IS SYMMETRIC (§3.2, §3.5). A polarity flip NEVER changes
    magnitude -- magnitude changes ONLY through defensive attack. The offensive
    attack is a SIGN-channel operation: it decides which way the link cuts, not how
    strong it is. So the link's CHAIN magnitude (mag_sigma) is recomputed over its
    DEFENSIVE attackers ONLY, dropping every offensive attacker, whether the link
    wins (keeps +1 at its surviving magnitude) or the turn wins (the flipped link
    carries that same surviving magnitude into the turned chain at the opponent's
    sign). v2 did this only for a LINK that won its weigh; v3 generalizes it to the
    turn-wins case too, so a winning turn GENERATES offense instead of merely
    driving its target to sigma 0."""
    from .resolve import resolve
    eff: Dict[str, object] = {}
    for nid in ctx.reachable:
        n = ctx.nodes[nid]
        if not isinstance(n, OFFENSE_BEARING):
            continue
        if nid not in ctx.offense_on:
            eff[nid] = 1            # defensive-only or unattacked: keeps +1, skip flip path
            continue

        preference = None
        for t in ctx.offense_on.get(nid, []):
            determinate, winner, _ = resolve(ctx, frozenset({nid, t}))
            if determinate:
                preference = 1 if winner == nid else -1   # link wins -> keeps; turn wins -> flips
                break

        pol, via = chainmod.effective_polarity(1, ctx.sigma[nid], preference=preference)
        eff[nid] = pol
        # Preserve magnitude across the flip: accrue over DEFENSIVE attackers only,
        # excluding every offensive attacker (they are the sign channel, not the
        # magnitude channel). This carries the surviving magnitude into the chain
        # at the resolved sign -- so a turned link contributes real offense, not 0.
        offensive = set(ctx.offense_on.get(nid, []))
        defensive = [ctx.sigma.get(a, TAU) for a, _e in ctx.attackers_by_target.get(nid, [])
                     if a in ctx.reachable and a not in offensive]
        ctx.mag_sigma[nid] = dfquad.accrue(TAU, defensive, [])
        ctx.trace.append(T.PolarityFlip(
            link_id=nid, from_sign=1, to_sign=pol, sigma=ctx.mag_sigma[nid], via=via,
        ))
    ctx.eff_pol = eff


def _terminals(ctx: Context, members, impacts) -> list:
    """Terminal impacts of a component (§2): impacts with nothing chaining forward
    to a later offense-bearing node. Mirrors `_terminal_impacts` but works from raw
    members/impacts (before the chain dict exists), for the §3.3.1 path aggregation."""
    out = []
    for imp in impacts:
        forwards = any(
            isinstance(e, Support) and nbr in members
            and isinstance(ctx.nodes.get(nbr), OFFENSE_BEARING) and nbr != imp
            and (_sidx(ctx.nodes[nbr].speech) or 0) > (_sidx(ctx.nodes[imp].speech) or 0)
            for nbr, e in ctx.adj.get(imp, []))
        if not forwards:
            out.append(imp)
    return out


def _spine_paths(ctx: Context, spine_set, impact, roots) -> list:
    """§3.3.1(a): enumerate the distinct simple spine paths converging on one shared
    `impact` -- each a node-id list [impact, ..., root] over the undirected Support
    subgraph restricted to `spine_set`. A path halts at the first root reached (the
    premise terminus). Graphs are small; DFS over simple paths."""
    root_set = set(roots)
    out = []

    def dfs(cur, path, seen):
        if cur in root_set and len(path) > 1:
            out.append(list(path))          # root is the premise terminus; do not pass it
            return
        for nbr, e in ctx.adj.get(cur, []):
            if isinstance(e, Support) and nbr in spine_set and nbr not in seen:
                seen.add(nbr); path.append(nbr)
                dfs(nbr, path, seen)
                path.pop(); seen.discard(nbr)

    dfs(impact, [impact], {impact})
    return out or [[impact]]                 # degenerate: impact is the whole spine


def _path_stats(ctx: Context, path, side):
    """(sign, mag, extended, ext_fail_node) for ONE root->impact path (§3.3.1a).
    A turned path (composed sign favors the opponent) checks liveness side-agnostic
    (§6); an ordinary path checks its own side. mag is the path's own σ product."""
    ob = [n for n in path if isinstance(ctx.nodes[n], OFFENSE_BEARING)]
    sign = qpn.sign_product([ctx.eff_pol.get(n, 1) for n in ob])
    favored = None if sign == qpn.UNRESOLVED else (side if sign > 0 else _opposing(side))
    turned = favored is not None and favored != side
    mag = 1.0
    for n in path:
        mag *= ctx.mag_sigma.get(n, TAU)
    for n in path:
        node = ctx.nodes[n]
        ok, _m = node_live_by_any_side(node) if turned else node_extension_ok(node)
        if not ok:
            return sign, mag, False, n
    return sign, mag, True, None


def _aggregate_impact(ctx: Context, spine_set, impact, roots, side):
    """§3.3.1(a-c): fold the paths converging on one shared impact into a single
    (sign, mag, extended, ext_fail_node) for the ONE component chain. Per-path
    liveness (a: impact survives iff >=1 complete path is extended); same-sign
    redundancy -> MAX not product (b); equal-magnitude sign-conflict -> wash to
    UNRESOLVED (c). The ballot never sees a path twice -- one object per impact."""
    stats = [_path_stats(ctx, p, side) for p in _spine_paths(ctx, spine_set, impact, roots)]
    ext_fail_node = next((s[3] for s in stats if not s[2]), None)
    # live carriers: complete path, non-zero magnitude, resolved sign
    live = [(sg, mg) for (sg, mg, ext, _f) in stats if ext and mg > EPSILON and sg != qpn.UNRESOLVED]
    if live:
        pos = [mg for sg, mg in live if sg > 0]
        neg = [mg for sg, mg in live if sg < 0]
        if pos and neg:                                   # (c) sign conflict at the impact
            hp, hn = max(pos), max(neg)
            if abs(hp - hn) <= EPSILON:                   # equal magnitude -> WASH (non-resolved-+1)
                return qpn.UNRESOLVED, max(hp, hn), True, None
            # unequal magnitude is OUT OF SCOPE for V1 (ruling 4): provisional --
            # the strongest single surviving path carries the impact. No current
            # fixture exercises this; its isolating round is authored later.
            return (1, hp, True, None) if hp > hn else (-1, hn, True, None)
        if pos:                                           # (b) same-sign redundancy -> max
            return 1, max(pos), True, None
        return -1, max(neg), True, None
    # No surviving carrier: emit a representative dead path so the single-path
    # collapse semantics survive (defensive_kill / unresolved / extension_fail).
    extended_any = [s for s in stats if s[2]]
    if extended_any:                                      # extended but zeroed / unresolved
        rep = max(extended_any, key=lambda s: s[1])
        return rep[0], rep[1], True, None
    rep = stats[0]                                        # nothing extended -> extension_fail
    return rep[0], rep[1], False, ext_fail_node or rep[3]


def _build_chains(ctx: Context) -> None:
    """Enumerate same-side Support-edge components containing an Impact; propagate
    sign (QPN) and magnitude (§3.3), aggregating redundant root->impact paths per
    §3.3.1 (per-path liveness, same-sign max, equal-magnitude convergence wash).
    Apply the binary extension gate (§6) by reading each spine node's LIVENESS."""
    ids = [nid for nid in ctx.reachable]
    find, union, _parent = _union_find(ids)
    for e in ctx.edges:
        if isinstance(e, Support):
            a = ctx.nodes.get(e.source)
            b = ctx.nodes.get(e.target)
            if a and b and a.id in ctx.reachable and b.id in ctx.reachable and a.side == b.side:
                union(a.id, b.id)

    comps: Dict[str, List[str]] = defaultdict(list)
    for nid in ids:
        comps[find(nid)].append(nid)

    for root, members in comps.items():
        impacts = [m for m in members if isinstance(ctx.nodes[m], Impact)]
        if not impacts:
            continue
        side = ctx.nodes[members[0]].side
        chain_id = "chain:" + root

        # Under Model C there are no re-assertion duplicates: each spine node is a
        # single object contributing once. `spine_reps` stays the whole component
        # spine (descriptive: dict + collapse-reason naming); mag/sign/extended are
        # computed per-path (§3.3.1) so a dead node on one redundant path cannot
        # collapse an impact a clean sibling path still carries.
        spine_reps = sorted(
            (m for m in members if isinstance(ctx.nodes[m], SPINE_TYPES)),
            key=lambda x: (_sidx(ctx.nodes[x].speech) or 0, x),
        )
        spine_set = set(spine_reps)
        roots = ([m for m in spine_reps if isinstance(ctx.nodes[m], Advocacy)]
                 or [m for m in spine_reps if isinstance(ctx.nodes[m], Uniqueness)]
                 or [m for m in spine_reps if not isinstance(ctx.nodes[m], Impact)]
                 or list(impacts))
        terminals = _terminals(ctx, members, impacts)

        intro_idx = min((_sidx(ctx.nodes[m].speech) or 0) for m in members)
        intro_speech = SPEECH_ORDER[intro_idx]

        if len(terminals) == 1:
            # §3.3.1 per-path aggregation over the shared terminal impact. ONE chain
            # object per component -- the ballot sees the impact once (r35 guard).
            sign, mag, extended, ext_fail_node = _aggregate_impact(
                ctx, spine_set, terminals[0], roots, side)
        else:
            # Fallback (malformed / genuinely multi-terminal): the flat series
            # product (§3.3), with the component-level turned check.
            mag = 1.0
            for r in spine_reps:
                mag *= ctx.mag_sigma.get(r, TAU)   # weigh may have dropped a defeated turn
            sign = qpn.sign_product(
                [ctx.eff_pol.get(r, 1) for r in spine_reps if isinstance(ctx.nodes[r], OFFENSE_BEARING)])
            fav0 = None if sign == qpn.UNRESOLVED else (side if sign > 0 else _opposing(side))
            turned0 = fav0 is not None and fav0 != side
            extended, ext_fail_node = True, None
            for r in spine_reps:
                ok, _m = (node_live_by_any_side(ctx.nodes[r]) if turned0
                          else node_extension_ok(ctx.nodes[r]))
                if not ok:
                    extended, ext_fail_node = False, r
                    break

        # The side the composed sign favors -- the side that OWNS this chain's
        # offense. Equal to `side` reads normally; the opponent means TURNED (§3.5);
        # UNRESOLVED (incl. a §3.3.1c wash) favors nobody.
        favored = None
        if sign != qpn.UNRESOLVED:
            favored = side if sign > 0 else _opposing(side)
        turned = favored is not None and favored != side

        # "No new chains in rebuttals" (§6) stays a chain-level guard, over the
        # component's earliest introduction.
        if intro_speech in REBUTTAL_SPEECHES:
            extended, ext_fail_node = False, (spine_reps[0] if spine_reps else root)
        if not extended:
            node = ctx.nodes.get(ext_fail_node) if ext_fail_node else None
            if intro_speech in REBUTTAL_SPEECHES:
                missing = intro_speech
            else:
                missing = node_extension_ok(node)[1] if node is not None else None
            ctx.trace.append(T.ExtensionFail(
                chain_id=chain_id, missing_speech=missing, spine_node_id=ext_fail_node))

        delta = chainmod.delta(sign, mag)

        impact_reps = impacts
        unresolved = any(ctx.status.get(r) == "unresolved" for r in impact_reps)

        collapse_reason, responsible = _collapse_reason(
            ctx, extended, ext_fail_node, sign, mag, spine_reps, unresolved)

        # Owning side (descriptive, non-load-bearing): the side the composed sign
        # favors -- the introducing `side` for a normal chain, its opponent for a
        # turned chain (§3.5), "" when the sign is unresolved. `side` stays the
        # introducing side (load-bearing for the ballot's aff_sum/neg_sum routing).
        owner = favored or ""
        # Anchor-membership (§7): a turning link joins the chain it CAPTURED for BD
        # anchoring only -- a BD may direct the ballot at any node on the chain it
        # points to, and a turn that flipped one of the chain's links is on it. The
        # link is joined to the captured chain solely by its OffensiveAttack edge, so
        # union-find (same-side Support, §3.3) never made it a `members` element;
        # `anchor_members` records it WITHOUT touching aggregation (side/sign/mag/
        # owner read `members` only). Gated on LIVE CAPTURE -- `eff_pol[m] == -1`
        # means the attack actually flipped m; a defeated/washed turn (m kept +1)
        # stays in `offense_on` but earns NO anchor reach (offense_on is populated
        # pre-resolution, so it alone is not evidence of capture).
        anchor_members = set(members) | {
            a for m in members
            for a in ctx.offense_on.get(m, [])
            if ctx.eff_pol.get(m) == -1
        }
        ctx.chains.append({
            "id": chain_id, "side": side, "owner": owner, "members": set(members),
            "anchor_members": anchor_members,
            "spine_reps": spine_reps, "impacts": impact_reps,
            "mag": mag, "sign": sign, "delta": delta,
            "intro_speech": intro_speech, "extended": extended,
            "in_scope": True, "unresolved": unresolved,
            "collapse_reason": collapse_reason, "responsible": responsible,
        })
        ctx.trace.append(T.Chain(
            chain_id=chain_id, sign=sign, mag=mag, delta=delta,
            side=side, owner=owner, extended=extended, in_scope=True,
            collapse_reason=collapse_reason, responsible=responsible))


def _collapse_reason(ctx, extended, ext_fail_node, sign, mag, spine_reps, unresolved):
    """Why a chain establishes no (own-side) offense, and the node most
    responsible -- descriptive only, derived from already-resolved state. None
    means the chain stands (positive own-side offense)."""
    if not extended:
        return "extension_fail", ext_fail_node
    if unresolved:
        return "unresolved_sign", None
    if sign == qpn.UNRESOLVED:
        return "unresolved_sign", None
    if sign == -1:
        flipped = next((r for r in spine_reps
                        if isinstance(ctx.nodes[r], OFFENSE_BEARING) and ctx.eff_pol.get(r) == -1), None)
        return "sign_flip", flipped
    if mag <= EPSILON:
        dead = next((r for r in spine_reps if ctx.sigma.get(r, TAU) <= EPSILON), None)
        responsible = dead
        if dead is not None:
            # prefer naming the conceded (dropped) attacker that did the killing
            killer = next((a for a, _e in ctx.attackers_by_target.get(dead, [])
                           if ctx.status.get(a) == "dropped"), None)
            responsible = killer or dead
        return "defensive_kill", responsible
    return None, None


# --- Pass 5a: framework gating ------------------------------------------------

def _terminal_impacts(ctx: Context, ch: dict) -> list:
    """The chain's terminal impacts (§2): Impact members with nothing chaining
    forward out of them within the chain (they do not Support a later
    offense-bearing node). These seed the §5.3 anchor walk. Genuinely independent
    scored terminals are separate chains at discovery, so a V1 chain has exactly
    one; the fallback to all impacts keeps this robust for a malformed graph."""
    members = ch["members"]
    terminals = []
    for imp in ch["impacts"]:
        forwards = any(
            isinstance(e, Support) and nbr in members
            and isinstance(ctx.nodes.get(nbr), OFFENSE_BEARING) and nbr != imp
            and _sidx(ctx.nodes[nbr].speech) is not None
            and _sidx(ctx.nodes[nbr].speech) > (_sidx(ctx.nodes[imp].speech) or 0)
            for nbr, e in ctx.adj.get(imp, [])
        )
        if not forwards:
            terminals.append(imp)
    return terminals or list(ch["impacts"])


def _framework_anchors(ctx: Context, ch: dict) -> set:
    """The chain's framework ANCHORS (§5.3, revised): the Frameworks reachable
    from the chain's IMPACT TERMINAL(s) over `Support` paths whose interior nodes
    are never an `Advocacy` or a `BallotDirective`. Anchoring is 'this impact is
    evaluable under this framework' -- the impact reaching a framework through the
    conductive spine (Uniqueness/Link/Impact).

    `Advocacy` and `BallotDirective` are NOT conductive spine, so they are
    ABSORBING, NOT TRAVERSABLE: the walk may ARRIVE at one (a direct impact->
    advocacy or impact->BD edge is a legal arrival) but may not EXPAND outward from
    it (arrival != traversal). This is exactly what stops a NEG disad whose
    uniqueness links off the shared `Advocacy` from reaching the AFF framework by
    walking backward through its own premises and sideways through the advocacy
    into the AFF spine -- the walk arrives at the advocacy and halts. A DIRECT
    cross-side impact->framework edge (§11.26, 'I win even under their framework')
    still anchors, because it never passes through an absorbing interior node.

    Only `Support` edges are followed, so a framework kritik's `DefensiveAttack`
    onto the framework it criticizes is NEVER traversed: the kritik anchors to its
    own framework by the direct Support edge (§5.3), never spuriously to the
    attacked one.

    SINGLE source of truth for gating: `in_scope(chain)` is `winning in
    _framework_anchors(...)`, and the same set is the FRAMEWORK_GATE's descriptive
    `anchors[]`, so the identity `in_scope == (winning is None or winning in
    anchors)` holds BY CONSTRUCTION -- never two walks that happen to agree (FLAG 2)."""
    anchors: set = set()
    seen: set = set()
    stack = list(_terminal_impacts(ctx, ch))     # impact-terminal seed (§5.3)
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        node = ctx.nodes.get(cur)
        if isinstance(node, Framework):
            anchors.add(cur)
        if isinstance(node, (Advocacy, BallotDirective)):
            continue                              # absorbing: arrive, do not expand
        for nbr, e in ctx.adj.get(cur, []):
            if isinstance(e, Support) and nbr not in seen:
                stack.append(nbr)
    return anchors


def _terminal_impact(ctx: Context, ch: dict):
    """The chain's terminal impact -- the one carrying its SCORED delta (§8, FLAG
    1), never 'first terminal in iteration order'. Deterministic id-sorted tiebreak
    keeps it order-independent if a chain ever has more than one terminal."""
    terminals = _terminal_impacts(ctx, ch)
    return sorted(terminals)[0] if terminals else None


def pass6_framework(ctx: Context) -> None:
    """Framework gating (§5). Select the governing framework by LIVE-SET
    CARDINALITY -- never element order (§5.1) -- then binary-gate chains by
    BD-blocking anchoring (§5.3). Emit FRAMEWORK_SELECT once, FRAMEWORK_DEFEAT per
    defeat, FRAMEWORK_GATE per chain (§8)."""
    from .resolve import resolve

    frameworks = [n for n in ctx.nodes.values()
                  if isinstance(n, Framework) and n.id in ctx.reachable]

    # Live set (§5.1, §3.6, §5.4): survived accrual AND its own maker extended it.
    # maker-extension is own-side node_extension_ok -- frameworks have no union
    # liveness (§5.4). `live_pre` is the set BEFORE weigh-defeat; kept to
    # distinguish none_survived from all_defeated for FRAMEWORK_SELECT.via.
    live = {f.id for f in frameworks
            if ctx.sigma.get(f.id, TAU) >= POLARITY_THRESHOLD and node_extension_ok(f)[0]}
    live_pre = set(live)

    # Weigh-defeat through the SAME resolve() impacts use (§6.5). A determinate
    # framework weigh removes the dispreferred framework from the live set; an
    # indeterminate one yields NO defeat (the framework channel has no magnitude
    # floor, §3.6). Iterate distinct framework pairs a Weighing ranks.
    defeated = []
    seen_pairs = set()
    for w in ctx.weighings:
        pair = ctx.weigh_pair.get(w.id)
        if not pair or pair in seen_pairs:
            continue
        if not all(isinstance(ctx.nodes.get(m), Framework) for m in pair):
            continue
        seen_pairs.add(pair)
        determinate, winner, decider = resolve(ctx, pair)
        if determinate:
            loser = next(m for m in pair if m != winner)
            if loser in live or loser in live_pre:
                live.discard(loser)
                if loser not in [d[0] for d in defeated]:
                    defeated.append((loser, decider, winner))

    # Cardinality selection (§5.1): exactly one live framework governs; zero or
    # many is a wash. ORDER-PROOF: len(live) != 1  =>  winning_framework is None.
    winning = ctx.nodes[next(iter(live))] if len(live) == 1 else None
    ctx.winning_framework = winning
    assert (winning is not None) == (len(live) == 1)      # §5.1 order-proof invariant

    winning_id = winning.id if winning is not None else None
    via, reason = _framework_via(len(frameworks), live_pre, live, defeated)

    ctx.trace.append(T.FrameworkSelect(
        winning_framework_id=winning_id, live=sorted(live),
        defeated=sorted(d[0] for d in defeated), via=via, reason=reason))
    for loser, decider, winner in defeated:
        ctx.trace.append(T.FrameworkDefeat(
            framework_id=loser, weighing_id=decider, preferred_id=winner))

    # Gate (§5.3): ONE impact-rooted anchor walk (Advocacy/BD absorbing) feeds
    # both the boolean and the descriptive anchors[]. A wash (winning is None)
    # gates nothing -- every chain in scope. Defeat excludes no chain directly: a
    # chain is out only when the WINNER is not among its anchors, exactly as a
    # chain anchored to nothing is.
    for ch in ctx.chains:
        anchors = _framework_anchors(ctx, ch)
        in_scope = winning_id is None or winning_id in anchors
        ch["in_scope"] = in_scope
        ctx.trace.append(T.FrameworkGate(
            chain_id=ch["id"], impact_id=_terminal_impact(ctx, ch),
            framework_id=winning_id, in_scope=in_scope, anchors=sorted(anchors)))


def _framework_via(n_authored: int, live_pre: set, live_post: set, defeated: list):
    """Classify FRAMEWORK_SELECT.via (§5.2, §8) from the live-set snapshots.
    Distinguishes none_survived (frameworks authored, none maker-extended) from
    no_frameworks (none authored) from multiple_live / all_defeated."""
    if n_authored == 0:
        return "no_frameworks", "no framework was read"
    if len(live_post) == 1:
        if len(live_pre) == 1:
            return "sole_survivor", "one framework was ever live; no weigh needed"
        return "weigh", "a determinate weigh reduced the live set to one"
    if len(live_post) == 0:
        if len(live_pre) == 0:
            return "none_survived", "every framework fell below threshold or failed maker-extension"
        return "all_defeated", "a cycle of determinate weighs defeated every framework"
    return "multiple_live", "two or more frameworks live with no determinate weigh separating them"


# Ballot-stage weighing (ranking surviving offense via `resolve`) lives in
# judge._ballot, which consumes the same recursive clash-breaker as polarity.
