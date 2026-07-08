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
        elif isinstance(e, OffensiveAttack) and isinstance(target, Uniqueness):
            reason = "offense aimed at pre-world uniqueness"
        if reason:
            ctx.trace.append(T.InertAttack(edge_id=e.id, reason=reason))
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


def node_extension_ok(node: Node) -> Tuple[bool, Optional[str]]:
    """§6, read from the LIVENESS record: a spine node is extended iff its
    liveness covers every one of its own side's speeches from its introduction
    onward. Returns (ok, missing_speech). Liveness is side-agnostic (a node kept
    live by the opponent is live); this reads the record, never edges. The
    "no new chains in rebuttals" rule is applied at the chain level, so a
    weighing/framework introduced in a rebuttal is not penalized here."""
    liveness = node.liveness or {}
    intro_idx = _sidx(node.speech)
    if intro_idx is None:
        return True, None
    for s in side_speeches(node.side):
        if _sidx(s) >= intro_idx and s not in liveness:
            return False, s
    return True, None


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
      * a DETERMINATE won link-weigh keeps the preferred side's polarity outright
        (via 'preference', regardless of raw sigma). When the LINK wins, its
        offensive attacker (the turn) lost the clash, so it is defeated -- dropped
        from the link's CHAIN magnitude (mag_sigma), which is why a won weigh
        saves a turned link's chain;
      * absent/indeterminate -> the 0.5 sigma threshold (via 'dfquad'), as before.
    A defensively-only link keeps +1 no matter how low sigma falls (defense
    reduces magnitude, never reverses direction)."""
    from .resolve import resolve
    eff: Dict[str, object] = {}
    for nid in ctx.reachable:
        n = ctx.nodes[nid]
        if not isinstance(n, OFFENSE_BEARING):
            continue
        if nid not in ctx.offense_on:
            eff[nid] = 1            # defensive-only or unattacked: keeps +1, skip flip path
            continue

        preference, defeated = None, set()
        for t in ctx.offense_on.get(nid, []):
            determinate, winner, _ = resolve(ctx, frozenset({nid, t}))
            if determinate:
                if winner == nid:
                    preference, defeated = 1, {t}    # link wins -> keeps, turn defeated
                else:
                    preference = -1                  # turn wins -> link flips
                break

        pol, via = chainmod.effective_polarity(1, ctx.sigma[nid], preference=preference)
        eff[nid] = pol
        if defeated:
            # The defeated competing claim (the turn that lost the weigh) does not
            # reduce the winner: recompute the link's chain sigma without it.
            live = [ctx.sigma.get(a, TAU) for a, _e in ctx.attackers_by_target.get(nid, [])
                    if a in ctx.reachable and a not in defeated]
            ctx.mag_sigma[nid] = dfquad.accrue(TAU, live, [])
        ctx.trace.append(T.PolarityFlip(
            link_id=nid, from_sign=1, to_sign=pol, sigma=ctx.mag_sigma[nid], via=via,
        ))
    ctx.eff_pol = eff


def _build_chains(ctx: Context) -> None:
    """Enumerate same-side Support-edge components containing an Impact; multiply
    spine sigmas (mag) and effective polarities (sign) -> delta (§3.3). Apply the
    binary extension gate (§6) by reading each spine node's LIVENESS record."""
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
        # single object contributing once to the product.
        spine_reps = sorted(
            (m for m in members if isinstance(ctx.nodes[m], SPINE_TYPES)),
            key=lambda x: (_sidx(ctx.nodes[x].speech) or 0, x),
        )
        mag = 1.0
        for r in spine_reps:
            mag *= ctx.mag_sigma.get(r, TAU)   # weigh may have dropped a defeated turn
        sign = qpn.sign_product(
            [ctx.eff_pol.get(r, 1) for r in spine_reps if isinstance(ctx.nodes[r], OFFENSE_BEARING)]
        )
        delta = chainmod.delta(sign, mag)

        intro_idx = min((_sidx(ctx.nodes[m].speech) or 0) for m in members)
        intro_speech = SPEECH_ORDER[intro_idx]

        # extension gate (§6): every spine node extended (read from liveness); a
        # chain first introduced in a rebuttal does not count.
        extended = True
        ext_fail_node = None
        if intro_speech in REBUTTAL_SPEECHES:
            extended = False
            ext_fail_node = spine_reps[0] if spine_reps else root
            ctx.trace.append(T.ExtensionFail(
                chain_id=chain_id, missing_speech=intro_speech, spine_node_id=ext_fail_node))
        else:
            for r in spine_reps:
                ok, missing = node_extension_ok(ctx.nodes[r])
                if not ok:
                    extended = False
                    ext_fail_node = r
                    ctx.trace.append(T.ExtensionFail(
                        chain_id=chain_id, missing_speech=missing, spine_node_id=r))
                    break

        impact_reps = impacts
        unresolved = any(ctx.status.get(r) == "unresolved" for r in impact_reps)

        collapse_reason, responsible = _collapse_reason(
            ctx, extended, ext_fail_node, sign, mag, spine_reps, unresolved)

        ctx.chains.append({
            "id": chain_id, "side": side, "members": set(members),
            "spine_reps": spine_reps, "impacts": impact_reps,
            "mag": mag, "sign": sign, "delta": delta,
            "intro_speech": intro_speech, "extended": extended,
            "in_scope": True, "unresolved": unresolved,
            "collapse_reason": collapse_reason, "responsible": responsible,
        })
        ctx.trace.append(T.Chain(
            chain_id=chain_id, sign=sign, mag=mag, delta=delta,
            side=side, extended=extended, in_scope=True,
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

def pass6_framework(ctx: Context) -> None:
    """A won framework (unattacked-or-restored and extended) binary-gates impacts:
    an impact with no support path to it is out of scope (§5)."""
    fws = [n for n in ctx.nodes.values()
           if isinstance(n, Framework) and n.id in ctx.reachable]
    winning = None
    for fw in fws:
        if ctx.sigma.get(fw.id, TAU) >= POLARITY_THRESHOLD and node_extension_ok(fw)[0]:
            winning = fw
            break
    ctx.winning_framework = winning
    if winning is None:
        return   # no framework debate -> every impact stays in scope

    # Nodes reachable from the framework over Support edges (undirected). The
    # walk does not pass THROUGH a BallotDirective: the ballot sink is not part
    # of framework scope, and traversing it would falsely connect every impact
    # that anchors to the same BD.
    reach = set()
    stack = [winning.id]
    while stack:
        cur = stack.pop()
        if cur in reach:
            continue
        reach.add(cur)
        if isinstance(ctx.nodes.get(cur), BallotDirective) and cur != winning.id:
            continue
        for nbr, e in ctx.adj.get(cur, []):
            if isinstance(e, Support) and nbr not in reach:
                stack.append(nbr)

    for ch in ctx.chains:
        in_scope = False
        for imp in ch["impacts"]:
            imp_in = imp in reach
            in_scope = in_scope or imp_in
            ctx.trace.append(T.FrameworkGate(impact_id=imp, framework_id=winning.id, in_scope=imp_in))
        ch["in_scope"] = in_scope


# Ballot-stage weighing (ranking surviving offense via `resolve`) lives in
# judge._ballot, which consumes the same recursive clash-breaker as polarity.
