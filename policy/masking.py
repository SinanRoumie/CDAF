"""Legal-action masking for the factored policy heads.

THE ONE RULE OF THIS MODULE: it contains no legality logic of its own. Every mask is
built by constructing a concrete candidate action and asking the environment's
legal-action generator (`env.legal_actions.is_legal`) -- the SAME entry point
`CDAFEnvironment.step()` uses to validate a move. If the generator's rules change,
these masks change with them; there is no second copy to drift (environment_shell_spec
§Governing principle / this milestone's constraint).

What the policy legitimately owns is the action VOCABULARY/SHAPE (that an `introduce`
carries a role/target/edge_type, a `weigh` two nodes + favors, etc.) -- it must know
that to build candidate actions at all. Every legality DECISION is delegated.

Masks are aligned to `list(state.nodes)` (dict-insertion order) -- the same order the
observation and `GraphEncoder` node embeddings use -- so mask index k and node
embedding row k refer to the same node. `ActorCritic` asserts this alignment.

Factored / conditional masking. Several arguments are conditional (a `weigh`'s second
node depends on the first; a `connect`'s edge_type depends on both endpoints via the
Support-cycle rule). Those masks are computed GIVEN the prior choice by enumerating
only the candidates consistent with it and calling `is_legal`, so a masked factored
sample can never assemble an illegal complete action.
"""

from __future__ import annotations

from typing import List

import numpy as np

from env import ROLES, RELATIONSHIP_EDGE_TYPES, NEW, is_legal
from env.actions import Introduce, Extend, Concede, Weigh, Connect, EndSpeech

# Fixed, stable orderings so head output index k always means the same thing (head
# weights and checkpoints depend on this).
ACTION_TYPES = ("introduce", "extend", "concede", "weigh", "connect", "end_speech")
ACTION_ROLE_ORDER = (
    "uniqueness", "link", "impact", "advocacy", "framework", "ballot_directive",
)
EDGE_TYPE_ORDER = ("support", "defensive_attack", "offensive_attack")

assert set(ACTION_ROLE_ORDER) == set(ROLES), "role head order must cover ROLES exactly"
assert set(EDGE_TYPE_ORDER) == set(RELATIONSHIP_EDGE_TYPES), \
    "edge head order must cover RELATIONSHIP_EDGE_TYPES exactly"

_CONTENT = ""       # the policy is CONTENT-BLIND: it never emits content. This empty
                    # placeholder satisfies the action dataclass; the judge never reads it.


class LegalActionMask:
    """Builds factored legality masks for one `RoundState` by delegating every decision
    to `env.legal_actions.is_legal`. Instantiate once per state; call the mask methods
    (conditional ones take the prior choice as an argument)."""

    def __init__(self, state):
        self.state = state
        self.node_ids: List[str] = list(state.nodes)
        self.n = len(self.node_ids)

    # --- delegation primitive -------------------------------------------------
    def _legal(self, action) -> bool:
        return is_legal(self.state, action)

    # --- action-type mask -----------------------------------------------------
    def type_mask(self) -> np.ndarray:
        """(6,) bool over ACTION_TYPES: a type is legal iff it has >=1 legal complete
        action. Derived by OR-ing the argument masks (each built via `is_legal`), so
        even the type-level mask reimplements nothing."""
        return np.array([
            bool(self.introduce_target_mask().any()),
            bool(self.extend_target_mask().any()),
            bool(self.concede_target_mask().any()),
            bool(self.weigh_a_mask().any()),
            bool(self.connect_source_mask().any()),
            self._legal(EndSpeech()),
        ], dtype=bool)

    # --- introduce ------------------------------------------------------------
    def introduce_target_mask(self) -> np.ndarray:
        """(n+1,) bool: existing-node targets then a trailing NEW slot. An existing
        target is legal iff SOME edge_type makes the attaching introduce legal; NEW is
        legal iff a fresh-node introduce is legal."""
        m = np.zeros(self.n + 1, dtype=bool)
        for k, nid in enumerate(self.node_ids):
            m[k] = any(self._legal(Introduce(_CONTENT, ACTION_ROLE_ORDER[0], nid, et))
                       for et in EDGE_TYPE_ORDER)
        m[self.n] = self._legal(Introduce(_CONTENT, ACTION_ROLE_ORDER[0], NEW, None))
        return m

    def introduce_role_mask(self, target) -> np.ndarray:
        """(6,) bool over ACTION_ROLE_ORDER for a chosen target (NEW or a node id).
        `target` is NEW -> edge_type None; else a support edge (any legal attach edge)
        stands in to probe role legality."""
        if target == NEW:
            return np.array([self._legal(Introduce(_CONTENT, r, NEW, None))
                             for r in ACTION_ROLE_ORDER], dtype=bool)
        et = self._first_legal_attach_edge(target)
        return np.array([self._legal(Introduce(_CONTENT, r, target, et))
                         for r in ACTION_ROLE_ORDER], dtype=bool)

    def introduce_edge_mask(self, target) -> np.ndarray:
        """(3,) bool over EDGE_TYPE_ORDER for an ATTACHING introduce onto `target`
        (only meaningful when target != NEW)."""
        return np.array([self._legal(Introduce(_CONTENT, ACTION_ROLE_ORDER[0], target, et))
                         for et in EDGE_TYPE_ORDER], dtype=bool)

    def _first_legal_attach_edge(self, target):
        for et in EDGE_TYPE_ORDER:
            if self._legal(Introduce(_CONTENT, ACTION_ROLE_ORDER[0], target, et)):
                return et
        return EDGE_TYPE_ORDER[0]

    # --- extend / concede -----------------------------------------------------
    def extend_target_mask(self) -> np.ndarray:
        return np.array([self._legal(Extend(nid)) for nid in self.node_ids], dtype=bool)

    def concede_target_mask(self) -> np.ndarray:
        return np.array([self._legal(Concede(nid)) for nid in self.node_ids], dtype=bool)

    # --- weigh ----------------------------------------------------------------
    def weigh_a_mask(self) -> np.ndarray:
        """(n,) bool: node a is a legal first member iff it can be weighed against SOME
        other node."""
        return np.array([
            any(self._legal(Weigh(a, b, a)) for b in self.node_ids if b != a)
            for a in self.node_ids
        ], dtype=bool)

    def weigh_b_mask(self, a_id) -> np.ndarray:
        """(n,) bool: legal second members given the chosen first member `a_id`."""
        return np.array([self._legal(Weigh(a_id, b, a_id)) for b in self.node_ids], dtype=bool)

    # favors is a pointer at one of the two chosen nodes; both are always legal targets
    # of `favors` once the pair is legal, so the favors mask is unconditional [True, True]
    # over (a, b). (Kept explicit so a future favors legality rule flows through here.)
    def favors_mask(self, a_id, b_id) -> np.ndarray:
        return np.array([self._legal(Weigh(a_id, b_id, a_id)),
                         self._legal(Weigh(a_id, b_id, b_id))], dtype=bool)

    # --- connect --------------------------------------------------------------
    def connect_source_mask(self) -> np.ndarray:
        """(n,) bool: node s is a legal source iff SOME (target, edge_type) makes a
        connect legal."""
        return np.array([
            any(self._legal(Connect(s, t, et))
                for t in self.node_ids if t != s for et in EDGE_TYPE_ORDER)
            for s in self.node_ids
        ], dtype=bool)

    def connect_target_mask(self, s_id) -> np.ndarray:
        """(n,) bool: legal connect targets given source `s_id` (some edge_type legal)."""
        return np.array([
            any(self._legal(Connect(s_id, t, et)) for et in EDGE_TYPE_ORDER)
            for t in self.node_ids
        ], dtype=bool)

    def connect_edge_mask(self, s_id, t_id) -> np.ndarray:
        """(3,) bool over EDGE_TYPE_ORDER for the ordered pair (s, t). This is where the
        Support-cycle rule surfaces: `support` may be masked out for a specific pair
        while `defensive_attack` / `offensive_attack` stay legal."""
        return np.array([self._legal(Connect(s_id, t_id, et)) for et in EDGE_TYPE_ORDER],
                        dtype=bool)
