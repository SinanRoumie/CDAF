"""The CDAF RL environment shell -- Gym-style `reset()` / `step()`.

Agent-agnostic (environment_shell_spec §Scope): the caller may be an LLM-in-loop
agent (Phase 2) or a trained policy under self-play (Phase 5); the environment does
not know or care which. It introduces NO evaluation logic -- the judge is called
exactly ONCE, on the complete graph, at termination.

The terminal reward is BINARY (no per-move shaping). Non-terminal steps return
reward 0. At termination the completed graph is materialized and handed to the
judge; the verdict maps to the reward. Because the game is two-sided and zero-sum,
the ballot reward is reported from AFF's perspective (+1 AFF win / 0 NEG win) and
`info["rewards"]` gives the per-side split so a self-play harness can assign each
policy its own return.

POTENTIAL-BASED REWARD SHAPING (PBRS). The flat chain-extension bonus is RETIRED. The env
now EXPOSES a scalar potential Φ(s) via `info['phi']` each step (Φ_maxdiff over the mid-round
chain resolver -- observation.potential); the TRAINING LOOP forms the per-step shaping reward
F_t = λ·(γ·Φ_L(s') − Φ_L(s)) (rollout.apply_pbrs). Shaping lives entirely in training, not in
this env's reward: the env's terminal reward is the UNSHAPED ballot (minus the dormant inert
penalty). PBRS is policy-invariant (environment_shell_spec §step()), so nothing is annealed.

INERT-ACTION PENALTY (dormant, coef 0.0). A per-side penalty is subtracted from each side's
return at termination -- `inert_penalty_coef` per NO-OP RE-EXTEND that side took (the only
reward-side inert class; same-side attack / offense-at-non-polarity / redundant connect are
now structurally ILLEGAL, and the no-op re-extend is primarily priced via full-slot COST).
Detection is per-step (classified BEFORE `apply`); application is a terminal aggregate. At
0.0 it is byte-identical to no penalty -- a documented backstop (see `legal_actions.is_inert`,
`state.action_cost`).

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
from .legal_actions import check_legality, is_inert
from .observation import observe, potential
from .validator import validate_round, assert_scope_ruled

# Trace record kinds surfaced in `info` on terminal steps (diagnostics only; NOT
# part of the observation, never a training signal for the agent).
_DIAG_KINDS = ("DROP", "EXTENSION_FAIL", "BALLOT", "UNRESOLVED")


class CDAFEnvironment:
    """One debate round as an episode. Construct, `reset()`, then `step(action)`
    until `done`. Not thread-safe; one round per instance."""

    def __init__(self, inert_penalty_coef: float = 0.0):
        self.state: RoundState = RoundState()
        # Coefficient for the per-side inert-action penalty (see is_inert). One unit is
        # subtracted from a side's return per structurally-doomed-at-creation action it
        # took. Default 0.0 (byte-identical to no penalty); the run config supplies the
        # real value (currently 0.0, dormant backstop).
        self.inert_penalty_coef: float = inert_penalty_coef
        # NOTE: the flat chain-extension bonus (`chain_extension_bonus`) is RETIRED. Shaping
        # is now potential-based (PBRS): the env exposes Φ(s) via `info['phi']` each step and
        # the TRAINING LOOP forms the per-step shaping reward (environment_shell_spec §step()).
        # The env's terminal reward is unshaped: ballot minus the (dormant) inert penalty.

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

        # Inert-action detection: classify BEFORE apply (a no-op re-extend is an
        # idempotent set-add, unrecoverable afterwards) and against the ACTING side
        # (current_side may advance inside apply). REWARD ONLY -- the action is legal and
        # proceeds normally; only the terminal per-side penalty is affected.
        inert, kind = is_inert(self.state, action)
        if inert:
            self.state.record_inert(self.state.current_side, kind)

        self.state.apply(action)            # mutates; may auto-advance the speech

        if self.state.terminated:
            return self._terminate()

        # Env raw reward is 0 on non-terminal steps; it EXPOSES Φ(s) in info so the training
        # loop can form the per-step PBRS shaping reward (environment_shell_spec §step()).
        return observe(self.state), 0.0, False, {"phi": potential(self.state)}

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

        # (5) binary terminal ballot reward, reported from AFF's perspective, MINUS the
        # per-side inert-action penalty (both sides; dormant at coef 0.0). The flat
        # chain-extension bonus is RETIRED -- shaping is PBRS, applied in training, not here.
        aff_ballot = 1.0 if ballot == AFF else 0.0
        neg_ballot = 1.0 - aff_ballot
        # Per-side inert penalty: coef x (count of that side's inert actions). Non-positive.
        aff_penalty = self.inert_penalty_coef * self.state.inert_by_side.get(AFF, 0)
        neg_penalty = self.inert_penalty_coef * self.state.inert_by_side.get(NEG, 0)
        aff_reward = aff_ballot - aff_penalty   # NOTE: with a nonzero penalty the two sides
        neg_reward = neg_ballot - neg_penalty   # no longer sum to 1 (penalty is not zero-sum).
        info = {
            "winner": ballot,
            "rewards": {AFF: aff_reward, NEG: neg_reward},
            "reward_breakdown": {
                AFF: {"ballot": aff_ballot, "inert_penalty": -aff_penalty},
                NEG: {"ballot": neg_ballot, "inert_penalty": -neg_penalty},
            },
            "phi": 0.0,                                       # Φ(terminal) = 0 (invariance boundary)
            "inert_counts": dict(self.state.inert_by_kind),   # per-class diagnostics (both sides)
            "inert_counts_by_side": {s: dict(k)               # side x kind diagnostics
                                     for s, k in self.state.inert_by_side_kind.items()},
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
