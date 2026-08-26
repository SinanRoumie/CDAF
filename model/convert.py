"""One-time converter: normalize v1 elements to v2 (§5 of the extension
migration spec). Pure; imports nothing app-side.

v1 encodes extension as duplicate same-claim nodes joined by `ExtensionEdge`.
This re-encodes that structure losslessly as per-node liveness:

  1. collapse each ExtensionEdge-connected component to ONE node,
  2. introduction `speech` = the earliest speech in the component,
  3. `liveness` = every speech present in the component, each tagged CONTESTED
     if an opposing attack targeted that duplicate that speech, else CONCEDED
     (§7 -- inferred from structure, not a flat default),
  4. rewire edges that pointed at a collapsed duplicate to the survivor,
  5. discard the duplicates and every `ExtensionEdge`.

It runs on RAW element dicts, BEFORE typed parsing, so `ExtensionEdge` never has
to exist as a parsed edge object -- the last step of retiring the type. The old
structure holds exactly the information the new one needs, so this is a
re-encoding, not a guess. The file loader (serialize.from_dict) calls it for
every sub-v2 round; already-v2 rounds are never converted.
"""

from __future__ import annotations

from collections import defaultdict

from .nodes import CONTESTED, CONCEDED
from .speeches import speech_index

_EXTENSION_ETYPE = "ExtensionEdge"
_ATTACK_ETYPES = frozenset({"DefensiveAttackEdge", "OffensiveAttackEdge"})


def _is_node(el):
    return "source" not in el["data"]


def convert(elements: list) -> list:
    """Collapse v1 ExtensionEdge duplicates into per-node liveness. Takes a list
    of raw element dicts (v1) and returns a new list of v2 element dicts. Does
    not mutate the input."""
    nodes = [el for el in elements if _is_node(el)]
    edges = [el for el in elements if not _is_node(el)]
    by_id = {el["data"]["id"]: el for el in nodes}

    # 1. Union-find over ExtensionEdge components (same-claim duplicates).
    parent = {el["data"]["id"]: el["data"]["id"] for el in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for e in edges:
        if e["data"].get("etype") == _EXTENSION_ETYPE:
            s, t = e["data"]["source"], e["data"]["target"]
            if s in parent and t in parent:
                union(s, t)

    components = defaultdict(list)
    for nid in parent:
        components[find(nid)].append(nid)

    contested = _contested_targets(edges, by_id)

    # 2-3. One survivor per component: introduction = earliest speech; liveness =
    # every speech present, contested where that duplicate was attacked.
    rep_of = {}
    survivor_liveness = {}
    for members in components.values():
        rep = min(members, key=lambda m: (speech_index(by_id[m]["data"]["speech"]), m))
        for m in members:
            rep_of[m] = rep
        live = {}
        for m in members:
            sp = by_id[m]["data"]["speech"]
            status = CONTESTED if contested.get(m) else CONCEDED
            if live.get(sp) != CONTESTED:   # a clash wins over concession
                live[sp] = status
        survivor_liveness[rep] = {s: live[s] for s in sorted(live, key=speech_index)}

    # 4-5. Rebuild elements in original order: emit each survivor once (with its
    # liveness); rewire non-extension edges to survivors; drop ExtensionEdges,
    # collapse self-loops, and drop edges made redundant by the collapse.
    out = []
    emitted = set()
    seen_edges = set()
    for el in elements:
        if _is_node(el):
            rep = rep_of[el["data"]["id"]]
            if rep in emitted:
                continue
            emitted.add(rep)
            node = by_id[rep]
            new = {"data": dict(node["data"])}
            new["data"]["liveness"] = dict(survivor_liveness[rep])
            if "position" in node:
                new["position"] = dict(node["position"])
            out.append(new)
        else:
            if el["data"].get("etype") == _EXTENSION_ETYPE:
                continue                                    # discarded
            src = rep_of.get(el["data"]["source"], el["data"]["source"])
            tgt = rep_of.get(el["data"]["target"], el["data"]["target"])
            if src == tgt:
                continue                                    # self-loop from collapse
            key = (src, tgt, el["data"].get("etype"))
            if key in seen_edges:
                continue                                    # redundant after collapse
            seen_edges.add(key)
            data = dict(el["data"])
            data["source"], data["target"] = src, tgt
            out.append({"data": data})
    return out


def _contested_targets(edges, by_id):
    """id -> True for every node that is the TARGET of an opposing-side attack.

    Orientation is by speech recency (§2.2 of judge_spec): the later-speech node
    is the attacker, the earlier is the target -- the one 'attacked that speech'.
    Same-side attacks are incoherent and ignored."""
    flag = {}
    for e in edges:
        if e["data"].get("etype") not in _ATTACK_ETYPES:
            continue
        a = by_id.get(e["data"]["source"])
        b = by_id.get(e["data"]["target"])
        if a is None or b is None or a["data"]["side"] == b["data"]["side"]:
            continue
        ia, ib = speech_index(a["data"]["speech"]), speech_index(b["data"]["speech"])
        if ia == ib:
            continue                     # same speech -> same side; can't clash
        target = a if ia < ib else b     # earlier speech = the node under attack
        flag[target["data"]["id"]] = True
    return flag
