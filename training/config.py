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
                   parameter. As of 2026-08-08 EVERY semantic parameter is ruled: shaping is
                   ENABLED (coef 0.1) with the anneal triggers ARMED at provisional values,
                   and the inert-action penalty is ruled at 0.01. Nothing is deferred now --
                   the deferral MACHINERY remains (it still lets a run proceed with shaping
                   off and the triggers unset), it is simply not exercised by these defaults.

Any config diff touching `semantics` defines a NEW experiment, not a continuation
(spec §Configuration). `resolve()` writes the fully-resolved config to the run's output
directory so "which config produced this checkpoint" stays answerable.

DISCOUNT γ HAS BEEN RULED -- 0.999, by EXPLICIT USER RULING on 2026-08-06 (see the
`discount` field below for the value and its recorded rationale). γ is the one semantic
parameter needed just to make the code RUN -- the critic's Monte-Carlo value target
during warm-start and the GAE returns during PPO both consume it -- so BEFORE the ruling
neither the critic warm-start nor a PPO update could execute at all. That was the intended
fail-loud, and it is now resolved by a human DECISION, not by a silently-chosen default:
the ruled value is recorded here exactly the way the other semantic parameters are as they
are ruled. As of 2026-08-08 all remaining semantic parameters have since been ruled too
(shaping on, anneal triggers armed-but-provisional, inert-action penalty 0.01), so no
semantic parameter blocks a run and neither the tests nor the smoke script need throwaway
values -- the rulings ARE the defaults.
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

    # Potential-based shaping weight λ -- RULED 0.5 (user ruling; REPLACES the retired flat
    # chain-extension bonus, shaping_coef/shaping_enabled/anneal-triggers, all removed).
    # SEMANTIC: it sets how much the dense Φ-proxy shapes early learning, NOT correctness
    # (PBRS is policy-invariant for any λ; rl_training_spec §Reward). Φ_maxdiff ∈ [-1,1] vs the
    # ballot ∈ [0,1], so 0.5 is a half-scale value prior: dense enough to break the
    # sparse-reward bootstrap barrier, small enough that Φ's imperfections are a modest prior
    # for the critic to shed. FIXED, no schedule (invariance ⇒ nothing to withdraw; there is
    # no annealing for PBRS). Consumed in training/rollout `apply_pbrs`, with γ = `discount`
    # (single source, per the invariance constraint -- never a second literal).
    pbrs_lambda: Any = 0.5

    # Inert-action penalty coefficient -- RULED 0.0, DORMANT (user ruling 2026-08-08,
    # revised). Semantic (it changes what behavior is rewarded), recorded as a ruled DEFAULT
    # the same way as the others. The three structurally-incoherent inert classes (same-side
    # attack, offense-at-non-polarity, redundant connect) are now ILLEGAL (masked, never
    # sampled); the one context-dependent class (no-op re-extend) is priced via COST (a full
    # slot; env.state.action_cost), not reward. This coefficient is kept as a DORMANT
    # BACKSTOP -- a per-side reward penalty on no-op re-extends -- to be re-ruled nonzero
    # ONLY on evidence of residual slack-budget no-op spam the cost model does not reach
    # (rl_training_spec §Reward). At 0.0 it is byte-identical to no penalty.
    inert_penalty_coef: Any = 0.0

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

    def unset_fields(self) -> list:
        """Every semantic field still holding the sentinel -- the honest list of what is
        unruled. With the shaping-anneal machinery retired (PBRS is never annealed), there
        are no longer any DEFERRABLE fields: every unset semantic blocks a run."""
        return [f.name for f in fields(self) if isinstance(getattr(self, f.name), _Unset)]

    def deferred_unset_fields(self) -> list:
        """Retained for API compatibility: nothing is deferred anymore (the shaping-anneal
        triggers that used to be deferred are removed). Always empty."""
        return []

    def blocking_unset_fields(self) -> list:
        """Unset fields that MUST be ruled before a run can start -- now simply every unset
        semantic (no deferral)."""
        return self.unset_fields()


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
        """Build a config from a dict (e.g. a loaded JSON run config). A semantics key set
        to the string 'UNSET' is restored to the fail-loud sentinel EXPLICITLY (an absent
        key keeps the dataclass default). We inject the sentinel rather than skip the field:
        now that the ruled defaults are real values (shaping on, triggers armed), skipping
        would silently reload a serialized 'UNSET' as the ruled default -- a silent
        un-setting. Explicit injection keeps a written-out sentinel honest across a
        round-trip regardless of what the current defaults are."""
        tuning = TuningConfig(**(d.get("tuning") or {}))
        sem_in = dict(d.get("semantics") or {})
        sem = SemanticsConfig()
        for k, v in sem_in.items():
            if not hasattr(sem, k):
                raise KeyError(f"unknown semantics key: {k}")
            if v == "UNSET":
                setattr(sem, k, _unset(f"semantics.{k}"))   # restore fail-loud sentinel
            else:
                setattr(sem, k, v)
        return cls(tuning=tuning, semantics=sem, output_dir=d.get("output_dir"))

    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))
