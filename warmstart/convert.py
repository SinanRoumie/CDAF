"""Fixture -> legal action sequence inversion (docs/warm_start_data_spec.md).

Turns a finished, static argument graph (a `model.Round`, e.g. a `tests/oracle/*.json`
fixture) into an ordered list of legal env actions such that replaying them through a
fresh `CDAFEnvironment` reproduces a **judge-equivalent** graph (ruling A). The
`(observation, action)` pairs the training loop consumes are produced by that replay
(observe-before-each-step), per the spec's consumption contract.

Rulings implemented (spec §Resolved rulings):
  A  judge-equivalent, not byte-identical -- validated by comparing the replayed
     round's (winner, reason_class) to the fixture's oracle verdict.
  B  any-legal-path -- no plausibility modeling; a fixed deterministic ordering.
  C  connect placement: earliest speech both endpoints exist AND the move is legal /
     affordable, performed by that speech's side.
  D  extend/concede verb: `concede` iff carriage side != node owner, else `extend`.
  E  natural-forward-with-flips orientation: introduction edges are built `new ->
     target` in whatever direction natural forward construction produces (never the
     fixture's authored orientation); connect orientation is the cycle-safe one.
  F  atomic per-node carriage, speech-wide `ceil(count/K)` batched cost -- one carriage per
     chain, not per node (respects speech budgets).

Determinism (spec §Underdetermination): global introduction order is
`(speech_index, id)`; a node's introduction edge is the incident structural edge to the
earliest-`(speech_index, id)` already-introduced endpoint; the rest of its incident
edges become `connect`s.

Non-negotiable (spec §Governing principle): this module never fabricates state -- it
only emits actions the environment applies, checking `is_legal` before every step. If a
reconstruction cannot reproduce the oracle verdict, that is reported as a real result
(`ConversionResult.ok is False`), never silently patched.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from model import SPEECH_ORDER, SPEECH_SIDE, speech_index, serialize
from judge import judge

from env import CDAFEnvironment, check_legality, observe
from env.actions import Introduce, Extend, Concede, Weigh, Connect, EndSpeech, NEW


class ConversionError(Exception):
    """A fixture could not be inverted into a legal sequence at all (e.g. an action
    the generator rejects, or budget exhaustion before a mandatory move). Distinct
    from a *verdict mismatch*, which is reported via `ConversionResult.ok = False`."""


@dataclass
class ConversionResult:
    name: str
    ok: bool                                   # verdict-equivalent to the oracle?
    actions: List[object] = field(default_factory=list)          # ordered env actions
    pairs: List[Tuple[dict, object]] = field(default_factory=list)  # (observation, action)
    oracle_verdict: Tuple[str, str] = ("", "")     # (winner, reason_class) of the fixture
    replay_verdict: Tuple[str, str] = ("", "")     # (winner, reason_class) of the rebuild
    error: Optional[str] = None                    # set iff a ConversionError was raised
    diag: dict = field(default_factory=dict)       # per-fixture reconstruction facts


def _reason_class(trace) -> str:
    for r in trace:
        if getattr(r, "kind", None) == "BALLOT":
            return r.reason_class
    return ""


def oracle_verdict(rnd) -> Tuple[str, str]:
    """The fixture's authoritative (winner, reason_class)."""
    ballot, trace = judge(rnd)
    return ballot, _reason_class(trace)


def _resolve_favors(nodes, wid, pair) -> str:
    """The `favors` the env `Weigh` must carry so the built weigh scores identically to
    the fixture's (judge.resolve.preferred_node). Explicit favors -> use it; a cross-side
    pair with no explicit favors -> the lone own-side member (the judge's legacy default,
    byte-identical); a same-side pair -> inert (preferred_node returns None regardless),
    so any pair member is safe."""
    w = nodes[wid]
    fav = getattr(w, "favors", None)
    if fav is not None and fav in pair:
        return fav
    same = [m for m in pair if nodes[m].side == w.side]
    if len(same) == 1:                          # cross-side: legacy default
        return same[0]
    return pair[0]                              # same-side (inert) / malformed


def convert(rnd, name: str = "") -> ConversionResult:
    """Invert `rnd` and self-validate. Returns a `ConversionResult`; raises nothing for
    a verdict mismatch (reported as `ok=False`), but a genuine build failure sets
    `error` and `ok=False`."""
    nodes = {n.id: n for n in rnd.nodes}
    oracle = oracle_verdict(rnd)

    weigh_ids = {i for i, n in nodes.items() if n.kind == "weighing"}
    reg_ids = [i for i in nodes if i not in weigh_ids]
    comparison_edges = [e for e in rnd.edges if e.kind == "comparison"]
    structural_edges = [e for e in rnd.edges if e.kind != "comparison"]

    # weighing -> its compared pair (Comparison edges are authored weighing -> member)
    pair_of: Dict[str, List[str]] = defaultdict(list)
    for e in comparison_edges:
        w, m = (e.source, e.target) if e.source in weigh_ids else (e.target, e.source)
        pair_of[w].append(m)

    diag = {"n_nodes": len(nodes), "n_weigh": len(weigh_ids), "roots": 0,
            "intro_edges": [], "flipped_support": 0, "connects": [],
            "extends": 0, "concedes": 0}

    try:
        order, intro_meta, connects = _build_forest(
            nodes, reg_ids, weigh_ids, structural_edges, diag)
        pos = {i: k for k, i in enumerate(order)}
        result = _drive(name, nodes, order, pos, weigh_ids, order_meta=intro_meta,
                        connects=connects, pair_of=pair_of, diag=diag)
    except ConversionError as exc:
        return ConversionResult(name=name, ok=False, oracle_verdict=oracle,
                                error=str(exc), diag=diag)

    actions, pairs, replay = result
    return ConversionResult(
        name=name, ok=(replay == oracle), actions=actions, pairs=pairs,
        oracle_verdict=oracle, replay_verdict=replay, diag=diag)


def _build_forest(nodes, reg_ids, weigh_ids, structural_edges, diag):
    """Build a spanning forest over the regular nodes and their structural edges, and
    the introduction order. Each node attaches (`introduce`, `new -> parent`) to an
    ALREADY-INTRODUCED neighbor when one exists -- so forest edges become introduction
    edges and only genuinely non-forest edges become `connect`s. A node with no
    already-introduced neighbor is a `NEW` root.

    Speech order is the primary layer (a node is introduced in its own speech). Within a
    speech the incremental attach keeps the sub-forest a tree (determinism: at each step
    place the lowest-id attachable node, else the lowest-id remaining node as a root),
    which is what a naive fixed id-order would miss -- stranding a would-be child as a
    root and mis-typing its forest edge as a `connect` (and, via the extra move, blowing
    the speech budget).

    Returns (order, intro_meta, connects):
      order       -- global introduction order (list of node ids), speech-bucketed and
                     topological (a parent precedes its child).
      intro_meta  -- node -> {parent, kind, authored, flipped}; absent for roots.
      connects    -- list of (u, v, kind) for the non-forest edges (orientation chosen
                     later, E)."""
    adj: Dict[str, List[tuple]] = defaultdict(list)
    for e in structural_edges:
        if e.source in weigh_ids or e.target in weigh_ids:
            raise ConversionError(
                f"structural edge {e.id} ({e.kind}) is incident to a weighing node")
        adj[e.source].append((e.target, e.kind, e))
        adj[e.target].append((e.source, e.kind, e))

    order: List[str] = []
    seen: set = set()
    intro_meta: Dict[str, dict] = {}
    consumed: set = set()                       # edge ids used as introduction edges

    for slot in SPEECH_ORDER:
        remaining = {n for n in reg_ids if nodes[n].speech == slot}
        while remaining:
            placed = None
            for n in sorted(remaining):
                cand = [(nbr, kind, e) for (nbr, kind, e) in adj[n] if nbr in seen]
                if cand:
                    cand.sort(key=lambda c: (speech_index(nodes[c[0]].speech), c[0]))
                    nbr, kind, e = cand[0]
                    intro_meta[n] = {"parent": nbr, "kind": kind,
                                     "authored": (e.source, e.target),
                                     "flipped": (e.source, e.target) != (n, nbr)}
                    consumed.add(e.id)
                    placed = n
                    break
            if placed is None:                  # no attachable node -> a NEW root
                placed = min(remaining)
            order.append(placed); seen.add(placed); remaining.discard(placed)

    connects = [(e.source, e.target, e.kind) for e in structural_edges
                if e.id not in consumed]

    diag["roots"] = sum(1 for n in reg_ids if n not in intro_meta)
    for n in order:
        m = intro_meta.get(n)
        if m is None:
            continue
        if m["kind"] == "support" and m["flipped"]:
            diag["flipped_support"] += 1
        diag["intro_edges"].append({"child": n, "parent": m["parent"], "kind": m["kind"],
                                    "authored": m["authored"], "flipped": m["flipped"]})
    return order, intro_meta, connects


def _drive(name, nodes, order, pos, weigh_ids, order_meta, connects, pair_of, diag):
    """Drive a fresh env speech-by-speech, emitting the reconstructed actions."""
    intro_meta = order_meta

    env = CDAFEnvironment(); env.reset()
    fx2env: Dict[str, str] = {}
    actions: List[object] = []
    pairs: List[Tuple[dict, object]] = []

    def emit(action):
        ok, reason = check_legality(env.state, action)
        if not ok:
            raise ConversionError(
                f"illegal {type(action).__name__} at slot {env.state.current_slot}: {reason}")
        pairs.append((observe(env.state), action))
        actions.append(action)
        env.step(action)

    def newest_env_id() -> str:
        return list(env.state.nodes)[-1]

    pending = list(connects)                    # (a_fx, b_fx, kind) not yet placed

    for slot in SPEECH_ORDER:
        if env.state.terminated:
            break
        if env.state.current_slot != slot:      # env auto-advanced past an empty/full slot
            continue
        side = SPEECH_SIDE[slot]

        # (a) introductions for nodes whose intro speech is this slot, in the forest's
        #     topological order (a parent precedes its child, so fx2env[parent] exists).
        for nid in order:
            if nodes[nid].speech != slot:
                continue
            if env.state.current_slot != slot:
                raise ConversionError(f"budget exhausted at {slot} before introducing {nid}")
            m = intro_meta.get(nid)
            if m is None:
                emit(Introduce("", nodes[nid].kind, NEW, None))
            else:
                emit(Introduce("", nodes[nid].kind, fx2env[m["parent"]], m["kind"]))
            fx2env[nid] = newest_env_id()

        # (b) connects: place any whose endpoints both now exist, earliest-legal (C).
        pending = _place_connects(env, emit, fx2env, pending, slot, diag)

        # (c) weighs introduced this slot.
        for wid in sorted(weigh_ids):
            if nodes[wid].speech != slot:
                continue
            pair = pair_of[wid]
            if len(pair) != 2 or any(m not in fx2env for m in pair):
                raise ConversionError(f"weigh {wid} pair {pair} not both present at {slot}")
            fav = _resolve_favors(nodes, wid, pair)
            emit(Weigh(fx2env[pair[0]], fx2env[pair[1]], fx2env[fav]))
            fx2env[wid] = newest_env_id()

        # (d) carriage: one atomic extend/concede per node that must be live this slot
        #     but was introduced earlier (atomic stamps only that node; speech-wide
        #     batching prices N carriages at ceil(N / K)).
        _carry(env, emit, fx2env, nodes, pos, slot, side, diag)

        # (e) close the turn if the env has not already auto-advanced.
        if not env.state.terminated and env.state.current_slot == slot:
            emit(EndSpeech())

    while not env.state.terminated:             # advance any trailing empty speeches
        emit(EndSpeech())

    if pending:
        raise ConversionError(f"{len(pending)} connect(s) never placed: {pending}")

    replay = oracle_verdict(env.state.to_round())
    return actions, pairs, replay


def _place_connects(env, emit, fx2env, pending, slot, diag):
    """Place every pending connect whose endpoints both exist now, if legal/affordable
    at this slot (C). Orientation is the cycle-safe one (E): try both, take the legal
    one. Returns the still-pending list."""
    still: List[Tuple[str, str, str]] = []
    for a_fx, b_fx, kind in pending:
        if (a_fx not in fx2env or b_fx not in fx2env
                or env.state.current_slot != slot or env.state.remaining_budget < 1):
            still.append((a_fx, b_fx, kind))
            continue
        a_env, b_env = fx2env[a_fx], fx2env[b_fx]
        placed = False
        for s, t in ((a_env, b_env), (b_env, a_env)):
            if check_legality(env.state, Connect(s, t, kind))[0]:
                emit(Connect(s, t, kind))
                diag["connects"].append(
                    {"pair": (a_fx, b_fx), "kind": kind, "slot": slot,
                     "side": SPEECH_SIDE[slot]})
                placed = True
                break
        if not placed:
            still.append((a_fx, b_fx, kind))
    return still


def _carry(env, emit, fx2env, nodes, pos, slot, side, diag):
    """Emit ONE ATOMIC `extend`/`concede` per node that must be live this slot but was
    introduced earlier. Extend/concede stamp only the named node (no path-walk), so this
    reproduces exactly the fixture's per-node liveness -- including a chain that is
    deliberately only partially carried. Verb by ownership (D). Speech-wide batching
    means N carriages cost ceil(N / K) slots."""
    need = [i for i in nodes
            if speech_index(nodes[i].speech) < speech_index(slot)
            and slot in (nodes[i].liveness or {})]
    for nid in sorted(need, key=lambda i: (pos.get(i, 1 << 30), i)):
        if env.state.current_slot != slot:
            raise ConversionError(f"budget exhausted at {slot} before carriage of {nid}")
        verb = Extend if nodes[nid].side == side else Concede
        emit(verb(fx2env[nid]))
        diag["extends" if verb is Extend else "concedes"] += 1
