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
    kind: ClassVar[str] = "support"


# NOTE: there is no `Extension` edge type. Extension migrated to per-node
# `liveness` (Model C). Legacy v1 files still carry "ExtensionEdge" elements;
# model.convert collapses them into liveness at the dict level, before parsing,
# so the type never needs to exist as a class.


@dataclass
class DefensiveAttack(Edge):
    etype: ClassVar[str] = "DefensiveAttackEdge"
    kind: ClassVar[str] = "defensive_attack"


@dataclass
class OffensiveAttack(Edge):
    etype: ClassVar[str] = "OffensiveAttackEdge"
    kind: ClassVar[str] = "offensive_attack"


@dataclass
class Comparison(Edge):
    etype: ClassVar[str] = "ComparisonEdge"
    kind: ClassVar[str] = "comparison"


# Registry: on-disk etype string -> class. Order is the canonical type order.
EDGE_CLASSES = {
    cls.etype: cls
    for cls in (Support, DefensiveAttack, OffensiveAttack, Comparison)
}
EDGE_TYPE_NAMES = tuple(EDGE_CLASSES.keys())
