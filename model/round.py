"""A CDAF round: an ordered collection of nodes and edges, plus a schema version.

Pure typed data -- no legality, no scoring, no evaluation. `elements` preserves
the exact on-disk order (nodes and edges may interleave) so a round round-trips
byte-for-byte. `nodes` / `edges` are convenience views over `elements`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Union

from .edges import Edge
from .nodes import Node

SCHEMA_VERSION = 1


@dataclass
class Round:
    elements: List[Union[Node, Edge]] = field(default_factory=list)
    version: int = SCHEMA_VERSION

    @property
    def nodes(self) -> List[Node]:
        return [el for el in self.elements if isinstance(el, Node)]

    @property
    def edges(self) -> List[Edge]:
        return [el for el in self.elements if isinstance(el, Edge)]
