"""Run configuration for CDAF RL training (`rl_training_spec.md` §Configuration).

The config has TWO top-level sections mirroring the spec's adjustment protocol, and the
split is STRUCTURAL, not cosmetic:

  * `tuning`    -- mechanical throughput/optimization knobs. Delegable: a diverging loss
                   or an underutilized machine is sufficient evidence to change them, and
                   they have no bearing on WHAT the policy learns. These carry sensible
                   defaults (implementation judgment, per the milestone brief).

  * `semantics` -- parameters that change WHAT BEHAVIOR IS REWARDED OR EXPLORED (entropy
                   schedule, shaping coefficient, annealing trigger + window, pool
                   composition + sampling ratio, discount). Each is ruled on by the human;
                   until ruled it is an `_UNSET` sentinel and reading it raises, so a run
                   can never start having silently picked a default for a semantic
                   parameter. As of 2026-08-06 every semantic parameter has been ruled
                   EXCEPT the three shaping-anneal TRIGGER fields, which are DELIBERATELY
                   DEFERRED (see below): they are consumed only when shaping is active, so
                   while shaping is off they do not block a run yet still fail-loud if read.

Any config diff touching `semantics` defines a NEW experiment, not a continuation
(spec §Configuration). `resolve()` writes the fully-resolved config to the run's output
directory so "which config produced this checkpoint" stays answerable.

DISCOUNT γ HAS BEEN RULED -- 0.999, by EXPLICIT USER RULING on 2026-08-06 (see the
`discount` field below for the value and its recorded rationale). γ is the one semantic
parameter needed just to make the code RUN -- the critic's Monte-Carlo value target
during warm-start and the GAE returns during PPO both consume it -- so BEFORE the ruling
neither the critic warm-start nor a PPO update could execute at all. That was the intended
fail-loud, and it is now resolved by a human DECISION, not by a silently-chosen default:
the ruled value is recorded here exactly the way the other semantic parameters will be as
they are ruled. The remaining ELEVEN semantic parameters are still UNSET and still block a
real run. Tests and the smoke script that must exercise the mechanics pass EXPLICIT,
clearly-labelled throwaway values for those eleven -- never defaults baked into this file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, fields
from typing import Any, Optional


class UnsetHyperparameter(RuntimeError):
    """Raised when a run tries to consume a SEMANTIC hyperparameter the human has not yet
    ruled on. The message names the parameter and points at the adjustment protocol."""


class _Unset:
    """Sentinel for a semantic hyperparameter with no value yet. Distinct from None (which
    is a legitimate 'feature off' value elsewhere) and falsy so an accidental `if cfg.x`
    is caught. Any arithmetic/coercion raises with a pointed message."""

    _name = "<semantic hyperparameter>"

    def __init__(self, name: str = None):
        if name:
            self._name = name

    def __bool__(self):
        raise UnsetHyperparameter(
            f"{self._name} is UNSET -- it is a SEMANTIC hyperparameter and requires an "
            "explicit human ruling before any run (rl_training_spec §Adjustment protocol). "
            "Set it in the run config's `semantics` section.")

    def __float__(self):
        self.__bool__()

    def __repr__(self):
        return f"UNSET({self._name})"


UNSET = _Unset()


def _unset(name: str) -> Any:
    """Field default factory producing a NAMED unset sentinel."""
    return _Unset(name)


# The shaping-anneal TRIGGER fields. The anneal controller consumes them ONLY when shaping
# is ACTIVE (enabled + nonzero coefficient); with shaping off -- the shipped state -- the
# bonus mechanism is inert and never reads them. They are therefore DEFERRABLE: not
# required for a run to start while shaping is off, yet still fail-loud if accessed
# directly. Deferral is deliberate -- these three must be set from an OBSERVED win-rate
# plateau, never guessed, and never tuned to make a run look better (rl_training_spec
# §Adjustment protocol; integrity flag in `anneal.py`).
_SHAPING_ANNEAL_TRIGGER_FIELDS = (
    "anneal_trigger_ballot_winrate",
    "anneal_trigger_window_episodes",
    "anneal_decay_updates",
)


# ---------------------------------------------------------------------------
# tuning -- mechanical, delegable, carries defaults (implementation judgment).
# ---------------------------------------------------------------------------

@dataclass
class TuningConfig:
    """Throughput & optimization knobs. Defaults are implementation judgment (the brief
    grants throughput parameters to me); all are freely re-tunable on mechanical evidence
    (diverging loss, idle machine) without redefining the experiment.

    Note: clip_range, gae_lambda, value_loss_coef and the LR/epoch/minibatch knobs are the
    PPO 'standard defaults' the spec leaves mechanical -- only `discount` (γ) is semantic
    and lives in SemanticsConfig, matching the spec's two enumerated lists exactly."""
    # --- optimization ---
    learning_rate: float = 3e-4           # spec PPO table; linear-decayed over the run
    lr_linear_decay: bool = True
    minibatch_size: int = 256
    epochs_per_batch: int = 4
    grad_clip: float = 0.5
    # --- PPO mechanical (standard defaults; NOT semantic) ---
    clip_range: float = 0.2
    gae_lambda: float = 0.95
    value_loss_coef: float = 0.5
    # Target-KL early-stop (mechanical stability safeguard, not semantic): if an epoch's
    # mean approx-KL exceeds this, the remaining epochs of that PPO update are skipped, so a
    # single update can't push the policy arbitrarily far. Chosen from the validation-v1 KL
    # regime -- healthy steady-state was ~0.07-0.16, the early blow-ups were 0.68-1.10; 0.30
    # sits between, so it trips ONLY on genuine blow-ups and is inert during normal updates.
    # None disables it. This is the standard PPO2/spinning-up epoch early-stop.
    target_kl: Optional[float] = 0.30
    # --- rollout throughput ---
    episodes_per_update: int = 2000       # spec batch/budget table
    num_workers: int = 1                  # parallel rollout workers (trivially correct:
                                          # each holds its own env, no shared state)
    # --- warm-start (imitation) mechanical knobs ---
    warmstart_epochs: int = 10
    warmstart_minibatch_size: int = 128
    warmstart_learning_rate: float = 3e-4
    # --- snapshot/eval CADENCE (mechanical: when to snapshot, not what the pool is) ---
    snapshot_interval: int = 50           # updates between pool snapshots (spec table)
    ladder_eval_interval: int = 100       # updates between ladder evaluations (spec table)
    # --- budget ---
    total_updates: int = 5000             # spec 'first experiment, not a target'
    seed: int = 0


# ---------------------------------------------------------------------------
# semantics -- every field UNSET; a run refuses to start until the human rules.
# ---------------------------------------------------------------------------

@dataclass
class SemanticsConfig:
    """Parameters that change WHAT the policy is rewarded for or how it explores. Every
    field is UNSET; consuming one before it is set raises `UnsetHyperparameter`. The human
    must set each explicitly (spec §Adjustment protocol: 'Semantic parameters -- require a
    ruling'). Spec starting VALUES are quoted in comments for reference but deliberately
    NOT applied as defaults -- writing them here would be picking the semantic values the
    brief reserves for the human."""

    # Entropy schedule -- RULED (explicit user ruling 2026-08-06), recorded the same way
    # as γ. Imitation warm-start already provides a reasonable starting policy, so
    # aggressive exploration from step one is unnecessary: a MODEST initial coefficient
    # preserves some exploration to escape gaps in warm-start coverage, decaying LINEARLY
    # to ZERO by the HALFWAY point of training so the back half is pure exploitation /
    # refinement. (Deliberately below the spec's illustrative 0.05 -> 0.005 because
    # warm-start narrows the need for early entropy.)
    entropy_coef_initial: Any = 0.01
    entropy_coef_final: Any = 0.0
    entropy_decay_fraction: Any = 0.5

    # Chain-extension shaping bonus coefficient -- RULED 0.0 (user ruling 2026-08-06).
    # FORMALLY SET, not left unset: this defines the "on" magnitude as ZERO rather than
    # undefined. The bonus stays OFF (`shaping_enabled` False). With coefficient 0 the bonus
    # never fires REGARDLESS of trigger state (env: `chain_extension_bonus and ...`), which
    # is exactly why the anneal TRIGGER fields below can stay deferred without affecting any
    # run -- see `shaping_active()` and the readiness gate.
    shaping_coef: Any = 0.0
    shaping_enabled: bool = False         # HOOK is built; OFF by default (non-negotiable).

    # Annealing trigger + window -- DELIBERATELY LEFT UNSET (user ruling 2026-08-06):
    # deferred, NOT yet ruled. These must NEVER be tuned to make a run look better (the
    # existing non-negotiable, flagged in `anneal.py`), so the honest way to set them is by
    # observing where a REAL run's ballot win-rate actually plateaus -- setting them now
    # would be guessing numbers with no empirical basis. They are DEFERRED: while shaping is
    # inactive (`shaping_active()` False -- the shipped state) the controller never reads
    # them, so they do NOT block a run, yet they STILL fail-loud if accessed directly via
    # `require(...)`. Rule them once a real run produces a win-rate curve to look at.
    anneal_trigger_ballot_winrate: Any = field(default_factory=lambda: _unset("semantics.anneal_trigger_ballot_winrate"))
    anneal_trigger_window_episodes: Any = field(default_factory=lambda: _unset("semantics.anneal_trigger_window_episodes"))
    anneal_decay_updates: Any = field(default_factory=lambda: _unset("semantics.anneal_decay_updates"))

    # Self-play pool composition + sampling ratio -- RULED (user ruling 2026-08-06).
    # Approach-A pattern: permanent early ANCHORS + a rolling RECENT window, sized modestly
    # for a FIRST real run rather than long-running production. self_play_ratio 0.5 balances
    # stable mirror-self opponents against pool draws to prevent overfitting to only the
    # current policy; 3 anchors preserve fixed reference points (e.g. the imitation-only
    # start) across training. (Cap 20 with 5 recent + 3 anchors leaves headroom; the spec's
    # illustrative split was 15 recent + 5 anchors -- this run keeps a smaller working set.)
    pool_cap: Any = 20
    pool_recent: Any = 5
    pool_anchors: Any = 3
    self_play_ratio: Any = 0.5

    # Discount γ -- RULED VALUE, not an implementation default. Set to 0.999 by EXPLICIT
    # USER RULING on 2026-08-06, and recorded here the same way every other semantic
    # parameter will be as it is ruled (a human decision, captured -- never silently
    # defaulted). Rationale for the record: episodes are long (~50-100 actions over the
    # full speech budget) and the reward is fully sparse/terminal with the shaping bonus
    # OFF by default, so a HIGH γ is needed for the terminal ballot to back-propagate
    # credit to early-episode actions (e.g. 1AC framing choices). 0.999 was chosen OVER
    # the 0.95-0.99 range (typical for short-episode settings) specifically because of the
    # horizon length, and OVER γ = 1.0 to retain some time-preference for value-estimation
    # stability. This SUPERSEDES the spec PPO table's starting value of 0.99.
    discount: Any = 0.999

    def require(self, name: str) -> Any:
        """Read a semantic field, raising `UnsetHyperparameter` if it is still the
        sentinel. The single consumption gate -- every semantic read goes through here so
        the failure is uniform and loud. This fires for the deferred anneal-trigger fields
        too: deferral relaxes only the pre-run READINESS gate, never direct consumption."""
        val = getattr(self, name)
        if isinstance(val, _Unset):
            raise UnsetHyperparameter(
                f"semantics.{name} is UNSET -- rule on it before running "
                "(rl_training_spec §Adjustment protocol: semantic parameters require a ruling).")
        return val

    def shaping_active(self) -> bool:
        """True iff the chain-extension bonus will apply a NONZERO coefficient -- the only
        state in which the anneal machinery (and its trigger fields) is consumed. Requires
        shaping ENABLED and a SET, nonzero coefficient; a zero or unset coefficient means
        the bonus never fires (env: `chain_extension_bonus and ...`), so the trigger fields
        are irrelevant. Single source of truth for 'is shaping on', read by both the anneal
        controller and the readiness gate so they can never disagree."""
        if not self.shaping_enabled:
            return False
        coef = self.shaping_coef
        if isinstance(coef, _Unset):
            return False
        return float(coef) != 0.0

    def unset_fields(self) -> list:
        """Every semantic field still holding the sentinel -- the honest list of what is
        unruled, INCLUDING deliberately-deferred fields (for reporting)."""
        return [f.name for f in fields(self) if isinstance(getattr(self, f.name), _Unset)]

    def deferred_unset_fields(self) -> list:
        """Unset fields that do NOT block a run in the current shaping state: the
        shaping-anneal TRIGGER fields while shaping is inactive (the controller never reads
        them, so a run is well-defined without them). Empty once shaping is active -- then
        they are required like any other consumed parameter."""
        if self.shaping_active():
            return []
        return [n for n in self.unset_fields() if n in _SHAPING_ANNEAL_TRIGGER_FIELDS]

    def blocking_unset_fields(self) -> list:
        """Unset fields that MUST be ruled before a run can start: all unset semantics minus
        those currently deferred."""
        deferred = set(self.deferred_unset_fields())
        return [n for n in self.unset_fields() if n not in deferred]


@dataclass
class TrainingConfig:
    """Full run config: the two sections + the output directory. `is_ready_for_training`
    is the pre-flight gate the loop calls before any real update."""
    tuning: TuningConfig = field(default_factory=TuningConfig)
    semantics: SemanticsConfig = field(default_factory=SemanticsConfig)
    output_dir: Optional[str] = None

    # --- readiness gate -------------------------------------------------------
    def unset_semantics(self) -> list:
        """Every literally-unset semantic field (honest report, incl. deferred ones)."""
        return self.semantics.unset_fields()

    def deferred_semantics(self) -> list:
        """Unset fields that are deliberately DEFERRED and do not block a run (the
        shaping-anneal triggers while shaping is off)."""
        return self.semantics.deferred_unset_fields()

    def blocking_semantics(self) -> list:
        """Unset fields that WILL be consumed and therefore block a run until ruled."""
        return self.semantics.blocking_unset_fields()

    def is_ready_for_training(self) -> bool:
        """True iff every semantic hyperparameter that WILL BE CONSUMED has been ruled.
        Deliberately-deferred fields (the shaping-anneal triggers while shaping is off) do
        NOT block -- the controller never reads them, so a run is well-defined without them.
        The loop refuses to start a real run while this is False."""
        return not self.blocking_semantics()

    def assert_ready_for_training(self) -> None:
        missing = self.blocking_semantics()
        if missing:
            raise UnsetHyperparameter(
                "cannot start training -- these SEMANTIC hyperparameters are unset, will be "
                f"consumed, and require a human ruling: {missing}. "
                "(rl_training_spec §Adjustment protocol / §Configuration.)")

    # --- serialization --------------------------------------------------------
    def to_dict(self) -> dict:
        """Resolved config as a plain dict; unset semantics serialize to the literal
        string 'UNSET' so a written-out config is honest about what was never ruled on."""
        def clean(d):
            return {k: ("UNSET" if isinstance(v, _Unset) else v) for k, v in d.items()}
        return {"tuning": asdict(self.tuning),
                "semantics": clean(asdict(self.semantics)),
                "output_dir": self.output_dir}

    def write(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        """Build a config from a dict (e.g. a loaded JSON run config). Any semantics key
        absent or set to the string 'UNSET' stays the unset sentinel."""
        tuning = TuningConfig(**(d.get("tuning") or {}))
        sem_in = dict(d.get("semantics") or {})
        sem = SemanticsConfig()
        for k, v in sem_in.items():
            if v == "UNSET":
                continue                    # leave the sentinel in place
            if not hasattr(sem, k):
                raise KeyError(f"unknown semantics key: {k}")
            setattr(sem, k, v)
        return cls(tuning=tuning, semantics=sem, output_dir=d.get("output_dir"))

    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))
