"""The entropy-coefficient decay schedule (`rl_training_spec.md` §Entropy).

The former chain-extension shaping anneal (`ShapingAnnealController`, conditional trigger +
decay) is RETIRED: shaping is now potential-based (PBRS), which is policy-invariant and
therefore never withdrawn/annealed (rl_training_spec §Annealing — retired; §Reward). Only
the entropy schedule -- an unrelated exploration knob -- remains here.

Reads SEMANTIC values through `SemanticsConfig.require(...)`, so an unset schedule fails
loudly rather than silently running on a default.
"""

from __future__ import annotations

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
