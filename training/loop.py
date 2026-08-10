"""The training-loop orchestrator (`rl_training_spec.md`): imitation warm-start, then
self-play PPO with a fixed-interval checkpoint pool.

This ties the pieces together. It does NOT start a run on import or construction; a run
begins only when `SelfPlayTrainer.train(...)` is called, and that call REFUSES to proceed
while any semantic hyperparameter is unset (`config.assert_ready_for_training()`). The
milestone brief is explicit: build and validate the loop, confirm before kicking off a
real run. `training/smoke.py` exercises this wiring with a trivially small step cap.

Order (locked architecture decision, Option 1 then Option 3):
  1. IMITATION WARM-START first (behavior cloning on the fixture corpus). LLM-in-loop is a
     separate, later validation phase -- not part of this milestone.
  2. SELF-PLAY PPO from those weights, opponents drawn from the checkpoint pool.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Optional

import torch

from judge.config import AFF
from policy import ActorCritic
from .config import TrainingConfig
from .checkpoint import new_actor_critic, save_checkpoint, EncoderSpec
from .imitation import BehaviorCloning, build_warmstart_dataset, WarmStartDataset
from .rollout import collect_batch, flatten_steps
from .ppo import PPOUpdater
from .pool import CheckpointPool
from .anneal import entropy_coef
from .metrics import batch_metrics


def warm_start(ac: ActorCritic, config: TrainingConfig, *,
               dataset: WarmStartDataset = None, train_critic: bool = False,
               epochs: Optional[int] = None, max_minibatches_per_epoch: Optional[int] = None,
               seed: int = 0) -> list:
    """Run behavior-cloning warm-start on `ac`. Actor BC needs no semantic parameter and
    runs on its own; `train_critic=True` additionally fits the critic to the demonstration
    Monte-Carlo return, which requires the SEMANTIC discount (raises if unset). Returns
    per-epoch lists of `BCStats`."""
    dataset = dataset or build_warmstart_dataset()
    bc = BehaviorCloning(ac, config, train_critic=train_critic)
    rng = random.Random(seed)
    epochs = epochs if epochs is not None else config.tuning.warmstart_epochs
    history = []
    for _ in range(epochs):
        history.append(bc.train_epoch(dataset, rng=rng,
                                      max_minibatches=max_minibatches_per_epoch))
    return history


@dataclass
class SelfPlayTrainer:
    """Owns the learner, the checkpoint pool, the PPO updater, and the schedules across a
    self-play run. Constructing it is cheap and side-effect-free; `train()` is the only
    method that touches real compute, and it gates on config readiness first."""
    config: TrainingConfig
    ac: ActorCritic = None
    pool: CheckpointPool = None
    updater: PPOUpdater = None
    encoder_spec: EncoderSpec = field(default_factory=EncoderSpec)

    def __post_init__(self):
        if self.ac is None:
            self.ac = new_actor_critic(self.encoder_spec, seed=self.config.tuning.seed)
        outdir = self.config.output_dir or "runs/default"
        pool_dir = os.path.join(outdir, "pool")
        if self.pool is None:
            self.pool = CheckpointPool(directory=pool_dir, semantics=self.config.semantics)
        if self.updater is None:
            self.updater = PPOUpdater(self.ac, self.config)

    # --- one PPO update -------------------------------------------------------
    def run_update(self, update: int, *, rng, torch_generator: torch.Generator = None,
                   n_episodes: Optional[int] = None) -> dict:
        """Collect one self-play batch and take one PPO update. PBRS shaping (potential-based)
        is applied inside `collect_batch` (reading `pbrs_lambda` + `discount` from config, one
        γ source); the entropy coefficient is annealed here. Returns a metrics dict."""
        # Inert-action penalty coefficient is a constant read from config (dormant at 0.0).
        inert_penalty_coef = float(self.config.semantics.require("inert_penalty_coef"))

        def opponent_sampler(r):
            return self.pool.sample_opponent(self.ac, r)

        trajectories = collect_batch(
            self.ac, opponent_sampler, self.config, rng=rng,
            torch_generator=torch_generator, n_episodes=n_episodes,
            inert_penalty_coef=inert_penalty_coef)

        steps = flatten_steps(trajectories)
        ecoef = entropy_coef(update, self.config.tuning.total_updates, self.config.semantics)
        ppo_stats = self.updater.update(steps, entropy_coef=ecoef, rng=rng)

        metrics = batch_metrics(trajectories)
        metrics.update({
            "update": update, "entropy_coef": ecoef,
            "pbrs_lambda": float(self.config.semantics.require("pbrs_lambda")),
            "inert_penalty_coef": inert_penalty_coef,
            "n_ppo_minibatches": len(ppo_stats),
            "mean_policy_loss": _mean(s.policy_loss for s in ppo_stats),
            "mean_value_loss": _mean(s.value_loss for s in ppo_stats),
            "mean_entropy": _mean(s.entropy for s in ppo_stats),
            "mean_approx_kl": _mean(s.approx_kl for s in ppo_stats),
        })
        return metrics

    # --- full run -------------------------------------------------------------
    def train(self, *, warmstart: bool = True, warmstart_critic: bool = True,
              n_updates: Optional[int] = None, episodes_per_update: Optional[int] = None,
              seed: int = 0):
        """Run the full pipeline. GATED: refuses to start while any semantic hyperparameter
        is unset. This is the method the milestone brief says to confirm before calling on
        real compute -- nothing else in this package auto-runs it."""
        self.config.assert_ready_for_training()      # loud refusal if semantics unset

        outdir = self.config.output_dir or "runs/default"
        os.makedirs(outdir, exist_ok=True)
        self.config.write(os.path.join(outdir, "resolved_config.json"))

        rng = random.Random(seed)
        gen = torch.Generator().manual_seed(seed)

        if warmstart:
            warm_start(self.ac, self.config, train_critic=warmstart_critic, seed=seed)
            save_checkpoint(os.path.join(outdir, "warmstart.pt"), self.ac,
                            encoder_spec=self.encoder_spec, meta={"stage": "warmstart"})

        total = n_updates if n_updates is not None else self.config.tuning.total_updates
        history = []
        for update in range(total):
            metrics = self.run_update(update, rng=rng, torch_generator=gen,
                                      n_episodes=episodes_per_update)
            history.append(metrics)
            if (update + 1) % self.config.tuning.snapshot_interval == 0:
                self.pool.snapshot(self.ac, update, encoder_spec=self.encoder_spec)
        save_checkpoint(os.path.join(outdir, "final.pt"), self.ac,
                        encoder_spec=self.encoder_spec, meta={"stage": "final"})
        return history


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0
