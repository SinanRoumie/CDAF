"""CDAF node types -- pure typed data.

No Dash, no dash-cytoscape, no app imports. Each node type is a thin dataclass
that mirrors EXACTLY the fields the app stores for a node:

    id, label, ntype (the type, carried by the class), side, speech, position

`position` is optional: app-saved rounds always include it, but hand-authored
or generated rounds may omit it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional


@dataclass
class Position:
    """Screen coordinates of a node. Numbers are stored as-is (int or float)."""
    x: float
    y: float


@dataclass
class Element:
    """Thin common base for everything in a round -- just the id."""
    id: str


@dataclass
class Node(Element):
    """Common base for all node types. `ntype` is the on-disk type string,
    supplied by each concrete subclass (not a per-instance field)."""
    label: str
    side: str
    speech: str
    position: Optional[Position] = None

    ntype: ClassVar[str] = ""


@dataclass
class Uniqueness(Node):
    ntype: ClassVar[str] = "Uniqueness"


@dataclass
class Link(Node):
    ntype: ClassVar[str] = "Link"


@dataclass
class Impact(Node):
    ntype: ClassVar[str] = "Impact"


@dataclass
class Advocacy(Node):
    ntype: ClassVar[str] = "Advocacy"


@dataclass
class Framework(Node):
    ntype: ClassVar[str] = "Framework"


@dataclass
class Weighing(Node):
    ntype: ClassVar[str] = "Weighing"


@dataclass
class BallotDirective(Node):
    ntype: ClassVar[str] = "BallotDirective"


# Registry: on-disk ntype string -> class. Order is the canonical type order.
NODE_CLASSES = {
    cls.ntype: cls
    for cls in (Uniqueness, Link, Impact, Advocacy, Framework, Weighing, BallotDirective)
}
NODE_TYPE_NAMES = tuple(NODE_CLASSES.keys())
