"""Lightweight rollout metrics (`rl_training_spec.md` §Logging).

Aggregates what is cheaply computable from a batch of `Trajectory` objects and the env's
terminal `info` diagnostics: ballot win rate, verdict distribution, mean shaped return,
episode length, and action-type distribution BROKEN OUT BY SPEECH (spec: also where
construction-and-compression would first become visible).

DELIBERATE PARTIAL COVERAGE (flagged, not hidden): the spec's per-gate pass rates
(advocacy / complete_chain / in_scope_impact / N>eps) and the extension-survival rate
(chains found vs extended -- the primary early indicator) require the judge's internal
CHAIN / gate trace records, which the env surfaces only partially through
`info["diagnostics"]` (DROP / EXTENSION_FAIL / BALLOT / UNRESOLVED). Those series are
therefore left as a follow-up that needs a small env/judge diagnostic surface, and are
NOT silently reported as zero. `ballot_win_rate` and `shaped_return` ARE reported as the
separate series the annealing section requires.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import List

from judge.config import AFF
from .rollout import Trajectory


def batch_metrics(trajectories: List[Trajectory]) -> dict:
    """Aggregate a rollout batch into a metrics dict. `ballot_win_rate` is AFF's ballot win
    rate (the annealing trigger reads this, NOT shaped return); `shaped_return` is the mean
    learner terminal reward including any shaping bonus -- logged as a SEPARATE series so a
    divergence (shaped return climbing while ballot win rate is flat = bonus farming) is
    visible."""
    n = len(trajectories)
    if n == 0:
        return {"n_episodes": 0}
    aff_wins = sum(1 for t in trajectories if t.winner == AFF)
    verdicts = Counter(t.winner for t in trajectories)
    lengths = [t.length for t in trajectories]
    shaped_returns = [t.terminal_reward for t in trajectories]

    # action-type distribution broken out by speech (learner decisions only).
    by_speech = defaultdict(Counter)
    for t in trajectories:
        for s in t.steps:
            by_speech[s.slot][s.action_type] += 1

    # inert-action tally, summed per kind across the batch (both sides; diagnostics only --
    # the primary read that the no-op re-extend spam is falling as the policy learns).
    inert_totals = Counter()
    for t in trajectories:
        inert_totals.update(t.inert_counts)
    total_inert = sum(inert_totals.values())

    return {
        "n_episodes": n,
        "ballot_win_rate_aff": aff_wins / n,
        "verdict_distribution": dict(verdicts),
        "mean_shaped_return_learner": sum(shaped_returns) / n,
        "mean_episode_length": sum(lengths) / n,
        "action_type_by_speech": {k: dict(v) for k, v in by_speech.items()},
        "mean_learner_decisions": sum(len(t.steps) for t in trajectories) / n,
        "inert_by_kind": dict(inert_totals),
        "mean_inert_per_episode": total_inert / n,
    }
