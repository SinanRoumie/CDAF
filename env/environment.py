"""The CDAF RL environment shell -- Gym-style `reset()` / `step()`.

Agent-agnostic (environment_shell_spec §Scope): the caller may be an LLM-in-loop
agent (Phase 2) or a trained policy under self-play (Phase 5); the environment does
not know or care which. It introduces NO evaluation logic -- the judge is called
exactly ONCE, on the complete graph, at termination.

Reward is BINARY and terminal only (no per-move shaping in V1). Non-terminal steps
return reward 0. At termination the completed graph is materialized and handed to
the judge; the verdict maps to the reward. Because the game is two-sided and
zero-sum, `reward` is reported from AFF's perspective (+1 AFF win / 0 NEG win) and
`info["rewards"]` gives the per-side split so a self-play harness can assign each
policy its own return.

Termination sequence (order is load-bearing):
  1. materialize state -> model.Round (`state.to_round`)
  2. STRUCTURAL ADMISSION ASSERTION (Fences A+G): `validate_round` must pass. Under
     Phase 1 the generator makes both corners unreachable, so a failure here is an
     environment BUG -- raise, never return a reward (Ruling 1 / item 3).
  3. run the judge ONCE -> (ballot, trace)
  4. SCOPE GUARD (Fence B): `assert_scope_ruled(trace)` -- the unruled unequal-
     magnitude convergence corner is unreachable in V1; if the marker ever appears
     the episode is unscoreable and we raise (Ruling 1).
  5. map ballot -> binary reward; attach judge diagnostics to `info`.
"""

from __future__ import annotations

from typing import Dict, Tuple

from judge import judge as run_judge
from judge.config import AFF, NEG

from .state import RoundState
from .actions import EndSpeech
from .legal_actions import check_legality
from .observation import observe
from .validator import validate_round, assert_scope_ruled

# Trace record kinds surfaced in `info` on terminal steps (diagnostics only; NOT
# part of the observation, never a training signal for the agent).
_DIAG_KINDS = ("DROP", "EXTENSION_FAIL", "BALLOT", "UNRESOLVED")


class CDAFEnvironment:
    """One debate round as an episode. Construct, `reset()`, then `step(action)`
    until `done`. Not thread-safe; one round per instance."""

    def __init__(self):
        self.state: RoundState = RoundState()

    # --- gym contract ---------------------------------------------------------
    def reset(self) -> Dict:
        """Initialize an empty graph at slot 1AC with that slot's budget. Returns the
        opening observation."""
        self.state = RoundState()
        return observe(self.state)

    def step(self, action) -> Tuple[Dict, float, bool, Dict]:
        """Apply one action. Returns (observation, reward, done, info).

        Illegal actions raise ValueError -- the caller is expected to consult
        `legal_actions.check_legality` and only submit structurally legal actions
        (the environment does not silently no-op an illegal move)."""
        if self.state.terminated:
            raise ValueError("step() called on a terminated episode")

        ok, reason = check_legality(self.state, action)
        if not ok:
            raise ValueError(f"illegal action {type(action).__name__}: {reason}")

        self.state.apply(action)            # mutates; may auto-advance the speech

        if self.state.terminated:
            return self._terminate()

        return observe(self.state), 0.0, False, {}

    # --- termination ----------------------------------------------------------
    def _terminate(self) -> Tuple[Dict, float, bool, Dict]:
        rnd = self.state.to_round()

        # (2) STRUCTURAL ADMISSION ASSERTION -- must hold; a failure is an env bug.
        validity = validate_round(rnd)
        assert validity.ok, (
            "STRUCTURAL ADMISSION failed at termination -- the legal-action generator "
            f"should have made this unreachable (Ruling 1 / item 3): {validity.reasons}")

        # (3) judge ONCE on the complete graph.
        ballot, trace = run_judge(rnd)

        # (4) SCOPE GUARD (Fence B) -- unreachable in V1; raises if ever hit.
        assert_scope_ruled(trace)

        # (5) binary terminal reward, reported from AFF's perspective.
        aff_reward = 1.0 if ballot == AFF else 0.0
        info = {
            "winner": ballot,
            "rewards": {AFF: aff_reward, NEG: 1.0 - aff_reward},
            "diagnostics": _diagnostics(trace),
        }
        return observe(self.state), aff_reward, True, info


def _diagnostics(trace) -> list:
    """Judge trace records useful for logging/debugging on terminal steps (DROP,
    EXTENSION_FAIL, ballot rationale, ...). Diagnostics only -- not observation,
    not training signal."""
    out = []
    for r in trace:
        kind = getattr(r, "kind", None)
        if kind in _DIAG_KINDS:
            out.append(r)
    return out
