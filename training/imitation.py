"""Imitation warm-start consumption (`rl_training_spec.md` §Imitation warm-start).

The policy is initialized by SUPERVISED training on the demonstration actions produced by
`warmstart/convert.py` from the hand-authored 1AC and the fixture corpus, then trained
with PPO from those weights. This module is the supervised half.

CONSUMPTION CONTRACT (spec §How warm-start consumes the output): per fixture, `convert`
yields an ordered list of legal env actions. The training loop constructs a fresh
`CDAFEnvironment` and, for each action in order, (1) `observe(state)` -> the observation
the policy would have seen, (2) records `(observation, demonstrated_action)`, (3)
`step()` to advance. The imitation loss is the FACTORED cross-entropy of the policy heads
(under the legal-action mask) against each demonstrated action's components -- i.e. the
negative factored joint log-prob from `ActorCritic.evaluate_action`. This is behavior
cloning: matching the policy's action distribution to the demonstrated actions.

Warm-start is NOT a scripted 1AC: the policy plays every speech from step one, and PPO is
free to move anywhere the reward leads afterwards. Warm-start only biases WHERE it starts.

CRITIC VALUE TARGET (the spec asks whether there is a natural one -- there IS):
the demonstrated trajectory has a known terminal ballot, so each state's Monte-Carlo
return is well-defined: the side-to-move's terminal reward, discounted back over the
remaining decisions. So the critic target at step t for the side S then to move is

    V_target(s_t) = discount ** (remaining_decisions_from_t) * terminal_reward[S]

with `terminal_reward[S] = 1.0 if S won the ballot else 0.0` (shaping OFF for warm-start;
the bonus is an exploration crutch for PPO, not a demonstration target). Two notes,
flagged honestly:
  * This REQUIRES the discount γ, a SEMANTIC hyperparameter. γ has been RULED (0.999,
    explicit user ruling 2026-08-06; see `config.SemanticsConfig.discount`), so critic
    warm-start now runs. Actor (behavior-cloning) warm-start needs no semantic parameter
    and runs on its own regardless. Were γ ever unset again, `require("discount")` would
    fail loudly rather than default.
  * Discounting is per-DECISION (one factor per remaining action), the same convention
    the PPO GAE uses over the learner's decision sequence, so warm-start and PPO value
    targets are on one clock.
"""

from __future__ import annotations

import copy
import glob
import os
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from model import serialize, SPEECH_SIDE
from judge.config import AFF, NEG
from env import CDAFEnvironment, observe
from env.actions import EndSpeech
from policy import ActorCritic
from warmstart import convert

from .config import TrainingConfig


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------

@dataclass
class WarmStartExample:
    """One (observation, demonstrated action) supervised example, plus the bookkeeping a
    critic value target needs. `state` is a deep-copied `RoundState` snapshot AT the
    decision point -- needed because the legal-action mask is built from state, not from
    the observation dict. `side` is the side to move; `remaining_decisions` counts actions
    from here to termination (the MC discount exponent)."""
    obs: dict
    state: object                       # RoundState snapshot (deepcopy)
    action: object
    side: str                           # AFF / NEG (side to move at this step)
    remaining_decisions: int
    terminal_reward: float              # side-to-move's terminal ballot reward (0/1)
    fixture: str


@dataclass
class WarmStartDataset:
    examples: List[WarmStartExample] = field(default_factory=list)
    fixtures_used: List[str] = field(default_factory=list)
    fixtures_skipped: List[str] = field(default_factory=list)

    def __len__(self):
        return len(self.examples)


def _oracle_dir() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "tests", "oracle")


def corpus_fixture_paths(oracle_dir: str = None) -> List[str]:
    """All *.json fixtures in the oracle corpus (the warm-start demonstration source)."""
    oracle_dir = oracle_dir or _oracle_dir()
    return sorted(glob.glob(os.path.join(oracle_dir, "*.json")))


def _terminal_rewards(winner: str) -> dict:
    """Binary terminal ballot reward per side (shaping OFF for demonstrations)."""
    return {AFF: 1.0 if winner == AFF else 0.0, NEG: 1.0 if winner == NEG else 0.0}


def build_examples_from_actions(actions, winner: str, fixture: str) -> List[WarmStartExample]:
    """Replay a converted action list through a fresh env, emitting one WarmStartExample
    per action with an observation, a state snapshot, the side to move, the remaining
    decision count, and the side's terminal reward. Independent of `convert`'s internal
    replay (re-derived here) so the dataset is self-contained."""
    rewards = _terminal_rewards(winner)
    env = CDAFEnvironment()
    env.reset()
    raw: List[WarmStartExample] = []
    total = len(actions)
    for k, action in enumerate(actions):
        side = env.state.current_side          # side to move at this decision
        obs = observe(env.state)
        snapshot = copy.deepcopy(env.state)
        raw.append(WarmStartExample(
            obs=obs, state=snapshot, action=action, side=side,
            remaining_decisions=0,             # filled below
            terminal_reward=rewards[side] if side in rewards else 0.0,
            fixture=fixture))
        env.step(action)
    # remaining_decisions = actions left until termination AFTER this one (0 for the last).
    for i, ex in enumerate(raw):
        ex.remaining_decisions = total - 1 - i
    return raw


def build_warmstart_dataset(paths: List[str] = None, *, require_ok: bool = True,
                            include_endspeech: bool = True) -> WarmStartDataset:
    """Convert every corpus fixture and assemble the supervised dataset. Fixtures whose
    conversion errored, or (when `require_ok`) whose replay verdict does not match the
    oracle, are SKIPPED and reported -- never trained on toward a graph the judge scores
    differently (spec §Governing principle).

    `include_endspeech` keeps the demonstrated `end_speech` moves as examples (they are
    genuine policy decisions); set False to train only on graph-building moves."""
    paths = paths or corpus_fixture_paths()
    ds = WarmStartDataset()
    for path in paths:
        name = os.path.basename(path)[:-5]
        rnd = serialize.load(path)
        result = convert(rnd, name)
        if result.error is not None or (require_ok and not result.ok):
            ds.fixtures_skipped.append(name)
            continue
        winner = result.replay_verdict[0]
        exs = build_examples_from_actions(result.actions, winner, name)
        if not include_endspeech:
            exs = [e for e in exs if not isinstance(e.action, EndSpeech)]
        ds.examples.extend(exs)
        ds.fixtures_used.append(name)
    return ds


# ---------------------------------------------------------------------------
# behavior cloning
# ---------------------------------------------------------------------------

@dataclass
class BCStats:
    actor_loss: float
    critic_loss: Optional[float]
    entropy: float
    n_examples: int
    unreachable: int                    # demos with a masked (unreachable) stage -> -inf


class BehaviorCloning:
    """Supervised warm-start of an `ActorCritic` against a `WarmStartDataset`.

    Actor loss = mean negative factored joint log-prob of the demonstrated action (the
    factored cross-entropy under the legal-action mask). Critic loss (optional) = MSE of
    the value head against the demonstrated Monte-Carlo return, which needs the SEMANTIC
    discount γ -- so `train_critic=True` requires `config.semantics.discount` to be set,
    and raises loudly otherwise.

    An `entropy` term is reported for logging but NOT added to the BC objective: warm-start
    is pure imitation; exploration is PPO's job (and is handled by the entropy schedule
    there)."""

    def __init__(self, ac: ActorCritic, config: TrainingConfig, *, train_critic: bool = False):
        self.ac = ac
        self.config = config
        self.train_critic = train_critic
        self.discount = None
        if train_critic:
            # Fail loudly if γ is unset -- do NOT silently default it.
            self.discount = float(config.semantics.require("discount"))
        t = config.tuning
        self.optimizer = torch.optim.Adam(ac.parameters(), lr=t.warmstart_learning_rate)

    # --- per-example loss -----------------------------------------------------
    def example_losses(self, ex: WarmStartExample):
        """(actor_loss, critic_loss_or_None, entropy, reachable) for one example. All are
        differentiable tensors (except the bool)."""
        enc_out = self.ac.evaluate(ex.obs)                 # one shared encoder forward
        log_prob, entropy = self.ac.evaluate_action(enc_out, ex.state, ex.action)
        reachable = bool(torch.isfinite(log_prob))
        actor_loss = -log_prob                              # factored cross-entropy
        critic_loss = None
        if self.train_critic:
            value = self.ac.value(enc_out.graph_embedding)
            target = (self.discount ** ex.remaining_decisions) * ex.terminal_reward
            critic_loss = (value - target) ** 2
        return actor_loss, critic_loss, entropy, reachable

    # --- minibatch step -------------------------------------------------------
    def train_minibatch(self, batch: List[WarmStartExample]) -> BCStats:
        self.optimizer.zero_grad()
        actor_terms, critic_terms, ent_terms = [], [], []
        unreachable = 0
        for ex in batch:
            a_loss, c_loss, ent, reachable = self.example_losses(ex)
            if not reachable:
                # A demonstrated action unreachable under the factored masks would inject
                # an infinite gradient; skip it and count it (surfaces a real data problem
                # rather than silently poisoning the batch).
                unreachable += 1
                continue
            actor_terms.append(a_loss)
            ent_terms.append(ent.detach())
            if c_loss is not None:
                critic_terms.append(c_loss)
        if not actor_terms:
            return BCStats(0.0, None if not self.train_critic else 0.0, 0.0, 0, unreachable)
        actor_loss = torch.stack(actor_terms).mean()
        loss = actor_loss
        critic_loss_val = None
        if critic_terms:
            critic_loss = torch.stack(critic_terms).mean()
            critic_loss_val = float(critic_loss.detach())
            loss = loss + self.config.tuning.value_loss_coef * critic_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ac.parameters(), self.config.tuning.grad_clip)
        self.optimizer.step()
        return BCStats(
            actor_loss=float(actor_loss.detach()),
            critic_loss=critic_loss_val,
            entropy=float(torch.stack(ent_terms).mean()) if ent_terms else 0.0,
            n_examples=len(actor_terms), unreachable=unreachable)

    def train_epoch(self, dataset: WarmStartDataset, *, rng=None,
                    max_minibatches: Optional[int] = None) -> List[BCStats]:
        """One pass over the dataset in minibatches. `rng` (a `random.Random`) shuffles
        example order; `max_minibatches` caps the pass (used by the smoke test to run a
        trivial number of steps without touching the whole corpus)."""
        idx = list(range(len(dataset.examples)))
        if rng is not None:
            rng.shuffle(idx)
        bs = self.config.tuning.warmstart_minibatch_size
        stats = []
        for start in range(0, len(idx), bs):
            if max_minibatches is not None and len(stats) >= max_minibatches:
                break
            batch = [dataset.examples[i] for i in idx[start:start + bs]]
            stats.append(self.train_minibatch(batch))
        return stats
