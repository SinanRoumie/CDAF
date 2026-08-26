"""Environment-side machinery for the CDAF RL build.

The `judge` package is a PURE scoring function: given a well-typed Round it emits a
verdict + trace and never rejects a graph. Everything ABOUT running episodes -- the
action space, the state graph, legal-action generation, the observation contract,
and episode validity -- lives here, so the judge stays pure and the RL loop owns
which graphs a training agent may generate episodes over.

Modules:
  actions       the five action types + pinned vocabulary (roles, edge types, budgets)
  state         RoundState: the graph under construction + `to_round()` materializer
  legal_actions structural legality only (incl. the ruled local Fence-A rule)
  observation   monotonic settled facts (no judge pass executed)
  environment   Gym-style reset()/step(); judge called once at termination
  validator     structural admission (Fences A+G) + the Fence-B scope guard
"""

from .validator import (
    RoundValidity, validate_round, is_valid,
    trace_has_convergence_marker, assert_scope_ruled, CONVERGENCE_MARKER,
)
from .actions import (
    Introduce, Extend, Concede, Weigh, Connect, EndSpeech,
    ROLES, RELATIONSHIP_EDGE_TYPES, ATTACH_EDGE_TYPES, NEW,
    ROLE_TO_NODE_CLASS, EDGE_TYPE_TO_CLASS, SPEECH_BUDGET, TOTAL_BUDGET,
)
from .state import RoundState, NodeRecord, EdgeRecord
from .legal_actions import check_legality, is_legal, legal_targets, is_inert
from .observation import observe, potential
from .environment import CDAFEnvironment

__all__ = [
    # validity
    "RoundValidity", "validate_round", "is_valid",
    "trace_has_convergence_marker", "assert_scope_ruled", "CONVERGENCE_MARKER",
    # actions / vocab
    "Introduce", "Extend", "Concede", "Weigh", "EndSpeech",
    "ROLES", "RELATIONSHIP_EDGE_TYPES", "ATTACH_EDGE_TYPES", "NEW",
    "ROLE_TO_NODE_CLASS", "EDGE_TYPE_TO_CLASS", "SPEECH_BUDGET", "TOTAL_BUDGET",
    # state
    "RoundState", "NodeRecord", "EdgeRecord",
    # legality / observation / env
    "check_legality", "is_legal", "legal_targets", "is_inert", "observe", "potential",
    "CDAFEnvironment",
]
