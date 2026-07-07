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
    """LEGACY (extension migration). Extension is no longer an authorable edge
    type -- persistence is per-node `liveness` now. This class is retained ONLY
    so (a) the one-time converter can recognise ExtensionEdges in v1 files while
    collapsing them into liveness, and (b) the pre-E2 judge (which still walks
    Extension edges) keeps importing. It is intentionally absent from
    EDGE_CLASSES / EDGE_TYPE_NAMES, so it cannot be authored or round-tripped as
    a live edge. Delete this class in E2 when the judge switches to liveness."""
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
# `Extension` is deliberately excluded -- it is a retired, non-authorable type
# (see its docstring); nothing new authors or parses it as a live edge.
EDGE_CLASSES = {
    cls.etype: cls
    for cls in (Support, DefensiveAttack, OffensiveAttack, Comparison)
}
EDGE_TYPE_NAMES = tuple(EDGE_CLASSES.keys())
