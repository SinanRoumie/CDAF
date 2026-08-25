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


@dataclass
class SpeechBlock:
    """One speech's M0 skeleton — the unit M1 turns into prose (one call/speech)."""
    speech: str
    side: str
    lines: List[str]      # component headers + typed placeholder/event lines
    budget: int           # RS23 budget sum for this speech


def _speech_blocks(analysis: Analysis, rnd: Round) -> List[SpeechBlock]:
    """Bucket the graph into per-speech M0 skeletons (v1.4: no assert)."""
    ctx = analysis.ctx
    comp_index = {id(c): i + 1 for i, c in enumerate(analysis.components)}
    owner = analysis.node2comp

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
        edges_by[(_edge_speech(e, ctx), _bucket_key(e, ctx, owner))].append(e)

    order_rank: Dict[int, Dict[str, int]] = {
        id(c): {nid: i for i, nid in enumerate(c.order)} for c in analysis.components}

    # RS19 correction note: per-node advocacy claim stems. The disambiguating
    # ordinal is scoped per side, per ROUND — an advocacy is a commitment in the
    # round, not the speech, so a 2AC advocacy is as distinct from the 1AC ones as
    # they are from each other (the boundary is temporal, not structural). Per side,
    # not global, so NEG's first advocacy does not inherit AFF's count. Ordered by
    # round appearance (speech, then id). Computed once so an extension in a later
    # speech restates the same stem (RS26).
    adv_by_side: Dict[str, List[str]] = defaultdict(list)
    for n in rnd.nodes:
        if n.kind == "advocacy":
            adv_by_side[n.side].append(n.id)
    adv_stem: Dict[str, str] = {}
    for side, ids in adv_by_side.items():
        ordered = sorted(ids, key=lambda i: (speech_index(ctx.nodes[i].speech), i))
        for ordv, nid in enumerate(ordered, start=1):
            adv_stem[nid] = T.advocacy_stem(side, ordv)

    blocks: List[SpeechBlock] = []
    for speech in SPEECH_ORDER:
        lines: List[str] = []
        budget = 0
        buckets: List[Tuple[Optional[Component], Optional[int]]] = \
            [(c, id(c)) for c in analysis.components] + [(None, None)]
        for comp, ckey in buckets:
            nn = new_nodes.get((speech, ckey), [])
            ex = extensions.get((speech, ckey), [])
            eg = edges_by.get((speech, ckey), [])
            if not (nn or ex or eg):
                continue
            if comp is not None:
                lines.append(_component_label(comp, comp_index[ckey]))
                rank = order_rank[ckey]
                nn = sorted(nn, key=lambda x: (rank.get(x, 1 << 30), x))
                ex = sorted(ex, key=lambda x: (rank.get(x, 1 << 30), x))
            else:
                lines.append("[unattached — no impact-bearing component]")
                nn, ex = sorted(nn), sorted(ex)

            for nid in nn:
                text, b = T.node_line(ctx.nodes[nid], stem_override=adv_stem.get(nid))
                lines.append("    " + text)
                budget += b
            for e in sorted(eg, key=lambda x: _edge_sort_key(x, ctx)):
                text, b = _edge_line(e, ctx)
                lines.append("    " + text)
                budget += b
                if e.id in analysis.invisible_edges:
                    kind, reason = analysis.invisible_edges[e.id]
                    lines.append(T.invis_marker(kind, reason))
            for nid in ex:
                text, b = T.extension_line(ctx.nodes[nid], stem_override=adv_stem.get(nid))
                lines.append("    " + text)
                budget += b
        if lines:
            blocks.append(SpeechBlock(
                speech=speech, side=SPEECH_SIDE.get(speech, "?"), lines=lines, budget=budget))
    return blocks


def _bucket_key(e, ctx, owner) -> Optional[int]:
    comp = owner.get(_edge_owner_id(e, ctx))
    return id(comp) if comp else None


def speech_skeletons(analysis: Analysis, rnd: Round) -> List[SpeechBlock]:
    """Public: the per-speech M0 skeletons, in forward speech order. This is the
    M1 prompt unit — one call per block (render/llm.py)."""
    return _speech_blocks(analysis, rnd)


def render(analysis: Analysis, rnd: Round, name: str = "round") -> str:
    out: List[str] = ["=" * 72,
                      f"ROUND: {name}   |   {len(rnd.nodes)} nodes, {len(rnd.edges)} edges, "
                      f"{len(analysis.components)} components",
                      "=" * 72]
    for blk in _speech_blocks(analysis, rnd):
        out.append("")
        out.append(f"--- {blk.speech} ({blk.side}) ---")
        # blank line before the first component header, matching prior layout
        for ln in blk.lines:
            if not ln.startswith("    ") and not ln.startswith("  ⟨"):
                out.append("")
            out.append(ln)
        out.append("")
        out.append(f"    [speech budget: {blk.budget} words]")
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
