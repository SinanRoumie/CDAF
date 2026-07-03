"""CDAF edge types -- pure typed data.

Each edge type is a thin dataclass mirroring EXACTLY the fields the app stores
for an edge:

    id, source, target, etype (the type, carried by the class)

Edges never carry a position. Note the on-disk `etype` strings keep the "Edge"
suffix (e.g. "SupportEdge"); the class names are the short forms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .nodes import Element


@dataclass
class Edge(Element):
    """Common base for all edge types. `etype` is the on-disk type string,
    supplied by each concrete subclass (not a per-instance field)."""
    source: str
    target: str

    etype: ClassVar[str] = ""


@dataclass
class Support(Edge):
    etype: ClassVar[str] = "SupportEdge"


@dataclass
class Extension(Edge):
    etype: ClassVar[str] = "ExtensionEdge"


@dataclass
class DefensiveAttack(Edge):
    etype: ClassVar[str] = "DefensiveAttackEdge"


@dataclass
class OffensiveAttack(Edge):
    etype: ClassVar[str] = "OffensiveAttackEdge"


@dataclass
class Comparison(Edge):
    etype: ClassVar[str] = "ComparisonEdge"


# Registry: on-disk etype string -> class. Order is the canonical type order.
EDGE_CLASSES = {
    cls.etype: cls
    for cls in (Support, Extension, DefensiveAttack, OffensiveAttack, Comparison)
}
EDGE_TYPE_NAMES = tuple(EDGE_CLASSES.keys())
