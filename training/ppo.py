"""The PPO surrogate for CDAF self-play (`rl_training_spec.md` §PPO configuration).

Standard clipped PPO over the learner's collected decisions. Each decision is re-evaluated
under the CURRENT parameters via `ActorCritic.evaluate_action`, which returns the FACTORED
joint log-prob and total entropy -- per-stage log-probs/entropy summed across the factored
heads, exactly the accounting the spec names ("the hook `sample_action` already left for
this"). The ratio is then a single scalar per decision even though the action is factored.

Objective (from AFF/side-relative advantages already in the trajectory):

    L = policy_loss  +  value_loss_coef * value_loss  -  entropy_coef * entropy

  * policy_loss = -E[min(r_t A_t, clip(r_t, 1-eps, 1+eps) A_t)]  (clipped surrogate)
  * value_loss  =  E[(V(s_t) - return_t)^2]                       (MSE to the GAE return)
  * entropy     =  E[factored entropy]                            (exploration)

DELIBERATE DEVIATIONS FROM A TEXTBOOK PPO, all flagged:
  * `entropy_coef` is NOT defaulted here -- it is a SEMANTIC hyperparameter (the entropy
    SCHEDULE) and must be supplied per update by the loop from the annealed schedule.
    Calling `update` without it is a type error, not a silent 0.
  * Advantages are normalized (mean/std) per batch -- a mechanical variance-reduction
    standard, not a semantic choice.
  * NO value-function clipping. Plain MSE to the return. Value clipping is an optional PPO
    refinement; omitted for a first build and noted so its absence is a choice, not an
    oversight. (Easy to add if value loss misbehaves -- spec §Value loss says to watch it.)
  * `discount`/`gae_lambda` do not appear here -- advantages/returns arrive precomputed by
    `rollout.compute_gae`, which owns the (semantic) discount.
  * TARGET-KL EARLY-STOP (`tuning.target_kl`, mechanical): after each epoch, if its mean
    approx-KL exceeds the threshold the remaining epochs of the update are skipped. Standard
    PPO2/spinning-up safeguard; added after validation-v1 showed update-0/1 KL blow-ups of
    0.68-1.10 (healthy ~0.1) from 3 epochs over a small high-variance batch. It is a
    stability cap, NOT tuned to flatter any metric: the threshold sits above the healthy KL
    regime, so it is inert on normal updates and only fires on genuine blow-ups.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

from policy import ActorCritic
from .config import TrainingConfig
from .rollout import RolloutStep


@dataclass
class PPOStats:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float                    # E[old_logp - new_logp], a standard PPO health gauge
    clip_fraction: float                # fraction of ratios hitting the clip band
    n_steps: int


class PPOUpdater:
    """Runs `epochs_per_batch` passes of clipped-PPO minibatch SGD over a batch of
    `RolloutStep`s. Holds the optimizer so its state (and LR schedule) persist across
    updates."""

    def __init__(self, ac: ActorCritic, config: TrainingConfig,
                 optimizer: Optional[torch.optim.Optimizer] = None):
        self.ac = ac
        self.config = config
        self.optimizer = optimizer or torch.optim.Adam(
            ac.parameters(), lr=config.tuning.learning_rate)
        # Diagnostics from the most recent update (read by loggers): how many of
        # `epochs_per_batch` epochs actually ran before the target-KL early-stop tripped.
        self.last_epochs_run: int = 0
        self.last_early_stopped: bool = False

    # --- minibatch ------------------------------------------------------------
    def _minibatch_loss(self, batch: List[RolloutStep], entropy_coef: float,
                        adv_mean: float, adv_std: float):
        clip = self.config.tuning.clip_range
        vf = self.config.tuning.value_loss_coef
        pol_terms, val_terms, ent_terms = [], [], []
        kls, clips = [], []
        for s in batch:
            enc = self.ac.evaluate(s.obs)                  # shared encoder forward (grad on)
            logp, entropy = self.ac.evaluate_action(enc, s.state, s.action)
            value = self.ac.value(enc.graph_embedding)

            adv = (s.advantage - adv_mean) / adv_std
            ratio = torch.exp(logp - s.old_log_prob)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv
            pol_terms.append(-torch.min(surr1, surr2))
            val_terms.append((value - s.ret) ** 2)
            ent_terms.append(entropy)

            with torch.no_grad():
                kls.append(s.old_log_prob - float(logp))
                clips.append(1.0 if (ratio < 1.0 - clip or ratio > 1.0 + clip) else 0.0)

        policy_loss = torch.stack(pol_terms).mean()
        value_loss = torch.stack(val_terms).mean()
        entropy = torch.stack(ent_terms).mean()
        loss = policy_loss + vf * value_loss - entropy_coef * entropy
        stats = PPOStats(
            policy_loss=float(policy_loss.detach()), value_loss=float(value_loss.detach()),
            entropy=float(entropy.detach()),
            approx_kl=sum(kls) / len(kls), clip_fraction=sum(clips) / len(clips),
            n_steps=len(batch))
        return loss, stats

    # --- full update ----------------------------------------------------------
    def update(self, steps: List[RolloutStep], *, entropy_coef: float, rng=None) -> List[PPOStats]:
        """One PPO update: `epochs_per_batch` shuffled passes of minibatch SGD over
        `steps`. `entropy_coef` MUST be supplied (semantic schedule); no default. Returns
        per-minibatch stats across all epochs."""
        if not steps:
            return []
        advs = [s.advantage for s in steps]
        adv_mean = sum(advs) / len(advs)
        var = sum((a - adv_mean) ** 2 for a in advs) / len(advs)
        adv_std = max(var ** 0.5, 1e-8)          # guard against a zero-variance batch

        bs = self.config.tuning.minibatch_size
        epochs = self.config.tuning.epochs_per_batch
        grad_clip = self.config.tuning.grad_clip
        target_kl = self.config.tuning.target_kl
        all_stats: List[PPOStats] = []
        self.last_epochs_run = 0
        self.last_early_stopped = False
        for _epoch in range(epochs):
            idx = list(range(len(steps)))
            if rng is not None:
                rng.shuffle(idx)
            epoch_stats: List[PPOStats] = []
            for start in range(0, len(idx), bs):
                batch = [steps[i] for i in idx[start:start + bs]]
                self.optimizer.zero_grad()
                loss, stats = self._minibatch_loss(batch, entropy_coef, adv_mean, adv_std)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ac.parameters(), grad_clip)
                self.optimizer.step()
                all_stats.append(stats)
                epoch_stats.append(stats)
            self.last_epochs_run += 1
            # Target-KL early-stop (mechanical): if this epoch moved the policy too far,
            # skip the remaining epochs of THIS update so a single update can't blow up.
            # Prevents the validation-v1 update-0/1 spikes (KL 0.68-1.10) from compounding.
            if target_kl is not None and epoch_stats:
                epoch_kl = abs(sum(s.approx_kl for s in epoch_stats) / len(epoch_stats))
                if epoch_kl > target_kl:
                    self.last_early_stopped = True
                    break
        return all_stats
