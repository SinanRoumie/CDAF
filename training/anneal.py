"""Schedules: the entropy coefficient decay and the chain-extension shaping anneal
(`rl_training_spec.md` §Entropy, §Reward / §Annealing trigger).

============================================================================
INTEGRITY FLAG -- READ BEFORE TOUCHING ANYTHING IN THIS FILE.
The shaping coefficient, the annealing trigger, and the evaluation protocol are
NEVER to be adjusted to make a run look better (spec §Adjustment protocol,
"Never adjusted to make a run look better"). If a run stalls below the trigger,
the finding is THAT IT STALLED. Loosening the trigger because a run has not reached
it converts a diagnostic into a rationalization, invisibly. These three are not free
parameters during debugging. Any change to them is a new experiment, ruled on
explicitly -- not a tuning tweak. This comment exists so the temptation is named.
============================================================================

Everything here reads SEMANTIC values through `SemanticsConfig.require(...)`, so an
unset schedule fails loudly rather than silently running on a default.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .config import SemanticsConfig


# ---------------------------------------------------------------------------
# entropy schedule
# ---------------------------------------------------------------------------

def entropy_coef(update: int, total_updates: int, semantics: SemanticsConfig) -> float:
    """Linear decay from `entropy_coef_initial` to `entropy_coef_final` over the first
    `entropy_decay_fraction` of training, constant `final` afterwards. All three are
    SEMANTIC (spec: 'initial 0.05, decayed to 0.005 over the first third') and required.

    Started well above the usual 0.01 default (see spec): the factored action space is
    large, warm-start narrows exploration around the demonstrated opening, and the shaping
    bonus creates a specific collapse risk toward `extend`."""
    e0 = float(semantics.require("entropy_coef_initial"))
    e1 = float(semantics.require("entropy_coef_final"))
    frac = float(semantics.require("entropy_decay_fraction"))
    decay_updates = max(1.0, frac * total_updates)
    if update >= decay_updates:
        return e1
    return e0 + (e1 - e0) * (update / decay_updates)


# ---------------------------------------------------------------------------
# chain-extension shaping anneal (conditional trigger)
# ---------------------------------------------------------------------------

@dataclass
class ShapingAnnealController:
    """The chain-extension shaping bonus coefficient over the run. Off by default (the
    env's `chain_extension_bonus` hook defaults 0.0 and `shaping_enabled` is False), so
    with shaping disabled this returns 0.0 forever and never reads a semantic value.

    When shaping IS enabled, the coefficient sits at `shaping_coef` until the CONDITIONAL
    trigger fires -- AFF BALLOT win rate (never total/shaped return) over a rolling window
    exceeds `anneal_trigger_ballot_winrate` -- after which it decays linearly to zero over
    `anneal_decay_updates` updates. Triggering on ballot win rate (not shaped return) stops
    the bonus from inflating its own trigger.

    The controller is STATEFUL: feed it each episode's AFF ballot outcome via
    `record_ballot`, and query `current_bonus(update)` per update. It logs ballot win rate
    and would let the caller log shaped return separately (spec: if they diverge, the policy
    is farming the bonus)."""
    semantics: SemanticsConfig
    triggered_at_update: int = None
    _window: deque = field(default=None)

    def __post_init__(self):
        # Set up the rolling window (and thus consult the trigger fields) ONLY when shaping
        # is active. With shaping off -- the shipped state, incl. the ruled coefficient 0.0
        # -- the controller is fully inert and never reads the (deferred) trigger fields.
        if not self.semantics.shaping_active():
            return
        window = int(self.semantics.require("anneal_trigger_window_episodes"))
        self._window = deque(maxlen=window)

    def record_ballot(self, aff_won: bool) -> None:
        """Record one episode's AFF BALLOT result (True iff AFF won the ballot). No-op when
        shaping is inactive (no window)."""
        if self._window is not None:
            self._window.append(1.0 if aff_won else 0.0)

    def aff_ballot_win_rate(self) -> float:
        if not self._window:
            return 0.0
        return sum(self._window) / len(self._window)

    def maybe_trigger(self, update: int) -> bool:
        """Check the conditional trigger; latch `triggered_at_update` the first time the
        rolling AFF ballot win rate crosses the threshold over a full window."""
        if not self.semantics.shaping_active() or self.triggered_at_update is not None:
            return self.triggered_at_update is not None
        threshold = float(self.semantics.require("anneal_trigger_ballot_winrate"))
        window = int(self.semantics.require("anneal_trigger_window_episodes"))
        if len(self._window) >= window and self.aff_ballot_win_rate() > threshold:
            self.triggered_at_update = update
        return self.triggered_at_update is not None

    def current_bonus(self, update: int) -> float:
        """The shaping coefficient to hand the env THIS update. 0.0 when shaping is inactive
        (disabled or coefficient 0 -- the shipped state); the full `shaping_coef` before the
        trigger; a linear decay to 0 over `anneal_decay_updates` after it."""
        if not self.semantics.shaping_active():
            return 0.0
        coef = float(self.semantics.require("shaping_coef"))
        if self.triggered_at_update is None:
            return coef
        decay_updates = float(self.semantics.require("anneal_decay_updates"))
        elapsed = update - self.triggered_at_update
        if elapsed >= decay_updates:
            return 0.0
        return coef * (1.0 - elapsed / decay_updates)
