"""JSON (de)serialization for CDAF rounds.

Round-trips the app's existing save format exactly, with one added top-level
field, "version". The element shape is:

    node:  {"data": {"id", "label", "ntype", "side", "speech"}, "position": {"x", "y"}}
    edge:  {"data": {"id", "source", "target", "etype"}}

`position` is written only when present and is tolerated as absent on load.
Files without a "version" key load as version 1.
"""

from __future__ import annotations

import json

from .edges import EDGE_CLASSES, Edge
from .nodes import NODE_CLASSES, Node, Position
from .round import SCHEMA_VERSION, Round


# --- single element <-> object -------------------------------------------------

def node_to_element(node: Node) -> dict:
    element = {
        "data": {
            "id": node.id,
            "label": node.label,
            "ntype": node.ntype,
            "side": node.side,
            "speech": node.speech,
        }
    }
    if node.position is not None:
        element["position"] = {"x": node.position.x, "y": node.position.y}
    return element


def edge_to_element(edge: Edge) -> dict:
    return {
        "data": {
            "id": edge.id,
            "source": edge.source,
            "target": edge.target,
            "etype": edge.etype,
        }
    }


def element_to_obj(element: dict):
    """Parse one on-disk element dict into a Node or Edge instance."""
    data = element["data"]
    if "source" in data:
        etype = data["etype"]
        cls = EDGE_CLASSES.get(etype)
        if cls is None:
            raise ValueError(f"Unknown edge type: {etype!r}")
        return cls(id=data["id"], source=data["source"], target=data["target"])

    ntype = data["ntype"]
    cls = NODE_CLASSES.get(ntype)
    if cls is None:
        raise ValueError(f"Unknown node type: {ntype!r}")
    pos = element.get("position")
    position = Position(pos["x"], pos["y"]) if pos else None  # optional on load
    return cls(
        id=data["id"], label=data["label"], side=data["side"],
        speech=data["speech"], position=position,
    )


def _element(obj) -> dict:
    return edge_to_element(obj) if isinstance(obj, Edge) else node_to_element(obj)


# --- round <-> dict / json -----------------------------------------------------

def to_dict(rnd: Round) -> dict:
    return {"version": rnd.version, "elements": [_element(el) for el in rnd.elements]}


def from_dict(d: dict) -> Round:
    version = d.get("version", SCHEMA_VERSION)  # default to v1 for pre-version files
    elements = [element_to_obj(el) for el in d.get("elements", [])]
    return Round(elements=elements, version=version)


def dumps(rnd: Round, *, indent: int = 2) -> str:
    return json.dumps(to_dict(rnd), indent=indent)


def loads(text: str) -> Round:
    return from_dict(json.loads(text))


def save(rnd: Round, path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(dumps(rnd))


def load(path) -> Round:
    with open(path, encoding="utf-8") as fh:
        return loads(fh.read())


def elements_from_round(rnd: Round) -> list:
    """The round as a list of on-disk element dicts (nodes + edges, in order)."""
    return [_element(el) for el in rnd.elements]
