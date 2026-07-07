"""CDAF domain model -- pure-Python schema for a debate round.

Single source of truth for what a node, edge, and round are. Imports nothing
from Dash / dash-cytoscape / the app, so an evaluator can read these graphs
without pulling in the UI.
"""

from .nodes import (
    Position, Element, Node,
    Uniqueness, Link, Impact, Advocacy, Framework, Weighing, BallotDirective,
    NODE_CLASSES, NODE_TYPE_NAMES,
    CONTESTED, CONCEDED, LIVENESS_STATUSES,
)
from .edges import (
    Edge, Support, Extension, DefensiveAttack, OffensiveAttack, Comparison,
    EDGE_CLASSES, EDGE_TYPE_NAMES,
)
from .round import Round, SCHEMA_VERSION
from .speeches import SPEECH_ORDER, SPEECH_SIDE, speech_index
from .convert import convert
from . import serialize

__all__ = [
    "Position", "Element", "Node",
    "Uniqueness", "Link", "Impact", "Advocacy", "Framework", "Weighing", "BallotDirective",
    "NODE_CLASSES", "NODE_TYPE_NAMES",
    "CONTESTED", "CONCEDED", "LIVENESS_STATUSES",
    "Edge", "Support", "Extension", "DefensiveAttack", "OffensiveAttack", "Comparison",
    "EDGE_CLASSES", "EDGE_TYPE_NAMES",
    "Round", "SCHEMA_VERSION",
    "SPEECH_ORDER", "SPEECH_SIDE", "speech_index",
    "convert", "serialize",
]
