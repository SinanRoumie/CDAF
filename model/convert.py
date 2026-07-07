"""One-time converter: normalize a v1-style Round to v2 (§5 of the extension
migration spec). Pure Round -> Round; imports nothing app-side.

v1 encodes extension as duplicate same-claim nodes joined by `ExtensionEdge`.
This re-encodes that structure losslessly as per-node liveness:

  1. collapse each ExtensionEdge-connected component to ONE node,
  2. introduction `speech` = the earliest speech in the component,
  3. `liveness` = every speech present in the component, each tagged CONTESTED
     if an opposing attack targeted that duplicate that speech, else CONCEDED
     (§7 -- inferred from structure, not a flat default),
  4. rewire edges that pointed at a collapsed duplicate to the survivor,
  5. discard the duplicates and every `ExtensionEdge`.

The old structure holds exactly the information the new one needs, so this is a
re-encoding, not a guess. The same function is called by the file loader (for v1
files) and, later, by the app's judge path.
"""

from __future__ import annotations

from collections import defaultdict

from .edges import Edge, Extension, DefensiveAttack, OffensiveAttack
from .nodes import Node, CONTESTED, CONCEDED
from .round import Round, SCHEMA_VERSION
from .speeches import speech_index

_ATTACK_TYPES = (DefensiveAttack, OffensiveAttack)


def convert(rnd: Round) -> Round:
    """Return a new v2 Round with extension collapsed into liveness. Does not
    mutate the input. Intended for v1-style input; the loader calls it only for
    rounds below the current schema version."""
    nodes = rnd.nodes
    by_id = {n.id: n for n in nodes}

    # 1. Union-find over ExtensionEdge components (same-claim duplicates).
    parent = {n.id: n.id for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for e in rnd.edges:
        if isinstance(e, Extension) and e.source in parent and e.target in parent:
            union(e.source, e.target)

    components = defaultdict(list)
    for nid in parent:
        components[find(nid)].append(nid)

    contested = _contested_targets(rnd.edges, by_id)

    # 2-3. One survivor per component: introduction = earliest speech; liveness =
    # every speech present, contested where that duplicate was attacked.
    rep_of = {}                 # any member id -> surviving (representative) id
    survivor = {}               # representative id -> new Node
    for members in components.values():
        member_nodes = [by_id[m] for m in members]
        rep = min(member_nodes, key=lambda n: (speech_index(n.speech), n.id))
        for m in members:
            rep_of[m] = rep.id
        live = {}
        for n in member_nodes:
            status = CONTESTED if contested.get(n.id) else CONCEDED
            # if duplicates share a speech, an active clash wins over concession
            if live.get(n.speech) != CONTESTED:
                live[n.speech] = status
        live = {s: live[s] for s in sorted(live, key=speech_index)}
        survivor[rep.id] = type(rep)(
            id=rep.id, label=rep.label, side=rep.side, speech=rep.speech,
            position=rep.position, liveness=live)

    # 4-5. Rebuild elements in original order: emit each survivor once; rewire
    # non-extension edges to survivors; drop ExtensionEdges, collapse self-loops,
    # and drop edges made redundant by the collapse.
    new_elements = []
    emitted = set()
    seen_edges = set()
    for el in rnd.elements:
        if isinstance(el, Node):
            rep_id = rep_of[el.id]
            if rep_id not in emitted:
                emitted.add(rep_id)
                new_elements.append(survivor[rep_id])
        elif isinstance(el, Edge):
            if isinstance(el, Extension):
                continue                                    # discarded
            src = rep_of.get(el.source, el.source)
            tgt = rep_of.get(el.target, el.target)
            if src == tgt:
                continue                                    # self-loop from collapse
            key = (src, tgt, el.etype)
            if key in seen_edges:
                continue                                    # redundant after collapse
            seen_edges.add(key)
            new_elements.append(type(el)(id=el.id, source=src, target=tgt))

    return Round(elements=new_elements, version=SCHEMA_VERSION)


def _contested_targets(edges, by_id):
    """id -> True for every node that is the TARGET of an opposing-side attack.

    Orientation is by speech recency (§2.2 of judge_spec): the later-speech node
    is the attacker, the earlier is the target -- the one 'attacked that speech'.
    Same-side attacks are incoherent and ignored."""
    flag = {}
    for e in edges:
        if not isinstance(e, _ATTACK_TYPES):
            continue
        a = by_id.get(e.source)
        b = by_id.get(e.target)
        if a is None or b is None or a.side == b.side:
            continue
        ia, ib = speech_index(a.speech), speech_index(b.speech)
        if ia == ib:
            continue                     # same speech -> same side; can't clash
        target = a if ia < ib else b     # earlier speech = the node under attack
        flag[target.id] = True
    return flag
