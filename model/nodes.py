"""CDAF node types -- pure typed data.

No Dash, no dash-cytoscape, no app imports. Each node type is a thin dataclass
that mirrors EXACTLY the fields the app stores for a node:

    id, label, ntype (the type, carried by the class), side, speech, position,
    liveness

`position` is optional: app-saved rounds always include it, but hand-authored
or generated rounds may omit it.

`liveness` (extension migration, Model C) records the speeches the node was
asserted live in, each tagged with its status that speech (contested/conceded).
`speech` stays the introduction speech; liveness always includes it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Dict, Optional

from .speeches import speech_index

# Per-speech liveness status (§7 of docs/extension_migration_spec.md).
CONTESTED = "contested"   # under active clash that speech
CONCEDED = "conceded"     # standing unanswered that speech
LIVENESS_STATUSES = (CONTESTED, CONCEDED)


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
    supplied by each concrete subclass (not a per-instance field).

    `liveness` maps each speech the node was asserted live in to its status that
    speech (CONTESTED / CONCEDED), ordered by SPEECH_ORDER. It always includes
    `speech`, the introduction speech, which is stamped automatically."""
    label: str
    side: str
    speech: str
    position: Optional[Position] = None
    liveness: Optional[Dict[str, str]] = None

    ntype: ClassVar[str] = ""

    def __post_init__(self):
        # Introduction stamps the first liveness entry automatically; a node's
        # liveness always includes its introduction speech (§2). The status
        # default is a placeholder -- the converter and (E2) the judge set the
        # real per-speech status.
        if self.liveness is None:
            self.liveness = {}
        self.liveness.setdefault(self.speech, CONTESTED)
        self.liveness = {s: self.liveness[s]
                         for s in sorted(self.liveness, key=speech_index)}


@dataclass
class Uniqueness(Node):
    ntype: ClassVar[str] = "Uniqueness"
    kind: ClassVar[str] = "uniqueness"


@dataclass
class Link(Node):
    ntype: ClassVar[str] = "Link"
    kind: ClassVar[str] = "link"


@dataclass
class Impact(Node):
    ntype: ClassVar[str] = "Impact"
    kind: ClassVar[str] = "impact"


@dataclass
class Advocacy(Node):
    ntype: ClassVar[str] = "Advocacy"
    kind: ClassVar[str] = "advocacy"


@dataclass
class Framework(Node):
    ntype: ClassVar[str] = "Framework"
    kind: ClassVar[str] = "framework"


@dataclass
class Weighing(Node):
    """A weigh over a same-type pair. `favors` is the explicit pointer at the pair
    member this weigh prefers (one of the two nodes its Comparison edges connect),
    read directly by the judge (§6.5). Optional: a legacy round (pre-favors channel)
    omits it and the judge derives a structural default."""
    favors: Optional[str] = None

    ntype: ClassVar[str] = "Weighing"
    kind: ClassVar[str] = "weighing"


@dataclass
class BallotDirective(Node):
    ntype: ClassVar[str] = "BallotDirective"
    kind: ClassVar[str] = "ballot_directive"


# Registry: on-disk ntype string -> class. Order is the canonical type order.
NODE_CLASSES = {
    cls.ntype: cls
    for cls in (Uniqueness, Link, Impact, Advocacy, Framework, Weighing, BallotDirective)
}
NODE_TYPE_NAMES = tuple(NODE_CLASSES.keys())
