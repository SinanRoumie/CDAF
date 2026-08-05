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

CHAIN-EXTENSION REWARD SHAPING (optional, off by default). On top of the ballot
reward, AFF earns a small bonus at termination iff it carried AT LEAST ONE chain
that is extended, in-scope, and sign +1 -- a genuine, spine-carried AFF offense
chain -- regardless of who won the ballot. The bonus is BINARY: one such chain is
worth exactly as much as three; magnitude does not scale it.

  Rationale (recorded): in uniform-random play AFF builds an offense chain ~75% of
  rounds but CARRIES one (extends its spine through every own-side speech) only
  ~1.6%, and passes zero ballot gates in 500 rounds -- so the terminal reward is
  constant-zero and nothing bootstraps. Rewarding chain EXISTENCE teaches "carry a
  spine," which is closer to a RULE of the game than a strategic opinion. Rewarding
  chain COUNT or MAGNITUDE would teach "extend everything," which is bad debate and
  is exactly the genuine strategy we want to stay EMERGENT -- hence the binary gate.

The coefficient (`chain_extension_bonus`) is configurable and ANNEALABLE to zero;
the final policy should train on the terminal reward alone, so the default is 0.0
(terminal-only, byte-identical to an unshaped env). It lives in the REWARD, never
in the observation -- the agent sees no signal that its chain was credited.

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

    def __init__(self, chain_extension_bonus: float = 0.0):
        self.state: RoundState = RoundState()
        # Coefficient for the AFF chain-extension shaping bonus (see module docstring).
        # Public and mutable so a training loop can ANNEAL it between episodes
        # (construct-per-episode or set on a reused instance). 0.0 == terminal reward
        # alone, byte-identical to an unshaped env; this is the default the final
        # policy trains under.
        self.chain_extension_bonus: float = chain_extension_bonus

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

        # (5) binary terminal ballot reward, reported from AFF's perspective, PLUS the
        # optional chain-extension shaping bonus (AFF-only, binary, off by default).
        aff_ballot = 1.0 if ballot == AFF else 0.0
        neg_ballot = 1.0 - aff_ballot
        bonus = (self.chain_extension_bonus
                 if self.chain_extension_bonus and _aff_carried_offense_chain(trace)
                 else 0.0)
        aff_reward = aff_ballot + bonus     # NOTE: with bonus>0 the two sides no longer
                                            # sum to 1 -- the bonus is an AFF auxiliary
                                            # reward, deliberately NOT zero-sum.
        info = {
            "winner": ballot,
            "rewards": {AFF: aff_reward, NEG: neg_ballot},
            "reward_breakdown": {
                AFF: {"ballot": aff_ballot, "chain_extension_bonus": bonus},
                NEG: {"ballot": neg_ballot},
            },
            "diagnostics": _diagnostics(trace),
        }
        return observe(self.state), aff_reward, True, info


def _aff_carried_offense_chain(trace) -> bool:
    """True iff AFF carried at least one chain that is EXTENDED, IN-SCOPE, and sign
    +1 -- a genuine spine-carried AFF offense chain. Reads the judge's CHAIN records
    off the terminal trace (the same records the fuzz diagnostic ranks on); presence,
    not count or magnitude (§ shaping rationale in the module docstring). `sign` is the
    QPN sign: +1 real AFF offense, -1 turned (favors NEG), "?" unresolved -- only +1
    counts. This never touches the ballot tally, so it credits a carried chain even in
    a round AFF lost."""
    for r in trace:
        if (getattr(r, "kind", None) == "CHAIN"
                and r.side == AFF and r.extended and r.in_scope and r.sign == 1):
            return True
    return False


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
