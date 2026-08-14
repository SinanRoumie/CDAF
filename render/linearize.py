"""M0 linearizer (spec §4; RS14-RS18, RS16b/c, RS17, RS23, RS26, RS13/RS13c).

Turns an `adapter.Analysis` into a per-speech transcript skeleton. Topology-
derived order only (RS14): identical graphs render identically regardless of
construction order. Component partition and flow order come from the judge
(RS16b); this module only linearizes.

Public: `render(analysis, rnd, name) -> str` and `summary_stats(...)`.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from model.round import Round
from model.speeches import SPEECH_ORDER, SPEECH_SIDE, speech_index

from . import templates as T
from .adapter import Analysis, Component, INCOMPLETE

_ATTACK_KINDS = ("defensive_attack", "offensive_attack")


# --- edge classification -----------------------------------------------------

def _edge_speech(edge, ctx) -> str:
    """The speech an edge belongs to: the later of its endpoints' introductions
    (RS25 — a bare edge is by definition later than both endpoints; absent an
    edge-level speech field, the later endpoint is the best available anchor)."""
    si = speech_index(ctx.nodes[edge.source].speech)
    ti = speech_index(ctx.nodes[edge.target].speech)
    return SPEECH_ORDER[max(si, ti)]


def _attack_orientation(edge, ctx):
    """(attacker, target): later-speech endpoint attacks the earlier (judge §2.2).
    Same-speech tie -> source is the target (the clash is inert either way)."""
    a, b = ctx.nodes[edge.source], ctx.nodes[edge.target]
    if speech_index(a.speech) >= speech_index(b.speech):
        return a, b            # a later or equal -> a attacks b
    return b, a


def _edge_owner_id(edge, ctx) -> str:
    """The node whose component owns this edge (RS17: cross-side attacks render in
    the TARGET's component). Support/Comparison -> the target endpoint."""
    if edge.kind in _ATTACK_KINDS:
        _attacker, target = _attack_orientation(edge, ctx)
        return target.id
    return edge.target


def _edge_line(edge, ctx):
    """(text, budget) for one edge, by kind (RS23/RS25)."""
    src, tgt = ctx.nodes[edge.source], ctx.nodes[edge.target]
    if edge.kind == "support":
        if speech_index(src.speech) == speech_index(tgt.speech):
            return T.edge_same_line(edge, src, tgt)      # same speech -> 0w syntax
        return T.edge_xapp_line(edge, src, tgt)          # cross-application 50w
    if edge.kind == "defensive_attack":
        attacker, target = _attack_orientation(edge, ctx)
        return T.edge_refute_line(edge, attacker, target)
    if edge.kind == "offensive_attack":
        attacker, target = _attack_orientation(edge, ctx)
        return T.edge_attack_line(edge, attacker, target)
    if edge.kind == "comparison":
        return T.edge_compare_line(edge, src, tgt)
    return (f"[edge {edge.kind} {edge.source}→{edge.target}]", 0)


def _edge_sort_key(edge, ctx) -> Tuple:
    """Deterministic within-speech edge order: kind, then owned-target id, source."""
    return (edge.kind, _edge_owner_id(edge, ctx), edge.source)


# --- transcript assembly -----------------------------------------------------

@dataclass
class _Line:
    text: str
    budget: int


def _component_label(comp: Component, flow_idx: int) -> str:
    label = f"[C{flow_idx} · {comp.side} · {comp.register}"
    if comp.multi_terminal:
        label += f" · {len(comp.terminals)} terminals"
    label += "]"
    if comp.register == INCOMPLETE:
        label += " " + T.incomplete_marker()
    return label


def render(analysis: Analysis, rnd: Round, name: str = "round") -> str:
    ctx = analysis.ctx

    # v1.4: no assert. Every no-register component renders as RS13c INCOMPLETE;
    # the renderer does not halt on any input.
    comp_index = {id(c): i + 1 for i, c in enumerate(analysis.components)}
    owner = analysis.node2comp

    # Bucket every event by (speech, component-or-None).
    # nodes: introduced in node.speech; extensions: each later liveness speech.
    new_nodes: Dict[Tuple[str, Optional[int]], List[str]] = defaultdict(list)
    extensions: Dict[Tuple[str, Optional[int]], List[str]] = defaultdict(list)
    for n in rnd.nodes:
        comp = owner.get(n.id)
        ckey = id(comp) if comp else None
        new_nodes[(n.speech, ckey)].append(n.id)
        intro = speech_index(n.speech)
        for sp in n.liveness:
            if speech_index(sp) > intro:
                extensions[(sp, ckey)].append(n.id)

    edges_by: Dict[Tuple[str, Optional[int]], List[object]] = defaultdict(list)
    for e in rnd.edges:
        if e.source not in ctx.nodes or e.target not in ctx.nodes:
            continue
        sp = _edge_speech(e, ctx)
        oid = _edge_owner_id(e, ctx)
        comp = owner.get(oid)
        edges_by[(sp, id(comp) if comp else None)].append(e)

    # RS16c order lookup per component (member id -> rank).
    order_rank: Dict[int, Dict[str, int]] = {
        id(c): {nid: i for i, nid in enumerate(c.order)} for c in analysis.components}

    out: List[str] = []
    stats_words: Dict[str, int] = {}
    out.append("=" * 72)
    out.append(f"ROUND: {name}   |   {len(rnd.nodes)} nodes, {len(rnd.edges)} edges, "
               f"{len(analysis.components)} components")
    out.append("=" * 72)

    for speech in SPEECH_ORDER:
        # Does this speech have any events at all?
        keys = [(speech, id(c)) for c in analysis.components] + [(speech, None)]
        has = any(new_nodes.get(k) or extensions.get(k) or edges_by.get(k) for k in keys)
        if not has:
            continue

        out.append("")
        out.append(f"--- {speech} ({SPEECH_SIDE.get(speech, '?')}) ---")
        speech_budget = 0

        # Components in fixed flow order (RS15), then the unattached bucket.
        buckets: List[Tuple[Optional[Component], Optional[int]]] = \
            [(c, id(c)) for c in analysis.components] + [(None, None)]
        for comp, ckey in buckets:
            nn = new_nodes.get((speech, ckey), [])
            ex = extensions.get((speech, ckey), [])
            eg = edges_by.get((speech, ckey), [])
            if not (nn or ex or eg):
                continue

            if comp is not None:
                out.append("")
                out.append(_component_label(comp, comp_index[ckey]))
                rank = order_rank[ckey]
                nn = sorted(nn, key=lambda x: (rank.get(x, 1 << 30), x))
                ex = sorted(ex, key=lambda x: (rank.get(x, 1 << 30), x))
            else:
                out.append("")
                out.append("[unattached — no impact-bearing component]")
                nn = sorted(nn)
                ex = sorted(ex)

            for nid in nn:
                text, b = T.node_line(ctx.nodes[nid])
                out.append("    " + text)
                speech_budget += b
            for e in sorted(eg, key=lambda x: _edge_sort_key(x, ctx)):
                text, b = _edge_line(e, ctx)
                out.append("    " + text)
                speech_budget += b
                if e.id in analysis.invisible_edges:
                    kind, reason = analysis.invisible_edges[e.id]
                    out.append(T.invis_marker(kind, reason))
            for nid in ex:
                text, b = T.extension_line(ctx.nodes[nid])
                out.append("    " + text)
                speech_budget += b

        stats_words[speech] = speech_budget
        out.append("")
        out.append(f"    [speech budget: {speech_budget} words]")

    return "\n".join(out)


def summary_stats(analysis: Analysis, rnd: Round) -> Dict:
    """Per-round structural stats for the M0 report."""
    reg = defaultdict(int)
    for c in analysis.components:
        reg[c.register] += 1

    # words per speech (RS23 budget sum), recomputed independently of render text.
    ctx = analysis.ctx
    words: Dict[str, int] = defaultdict(int)
    for n in rnd.nodes:
        words[n.speech] += T.BUDGET_NODE
        intro = speech_index(n.speech)
        for sp in n.liveness:
            if speech_index(sp) > intro:
                words[sp] += T.BUDGET_EXTENSION
    for e in rnd.edges:
        if e.source not in ctx.nodes or e.target not in ctx.nodes:
            continue
        _text, b = _edge_line(e, ctx)
        words[_edge_speech(e, ctx)] += b

    return {
        "components": len(analysis.components),
        "register": {k: reg[k] for k in sorted(reg)},
        "rs13c_markers": reg.get(INCOMPLETE, 0),
        "rs27_markers": len(analysis.invisible_edges),
        "rs10c_divergences": len(analysis.rs10c_divergences),
        "words_per_speech": {s: words[s] for s in SPEECH_ORDER if words.get(s)},
    }
