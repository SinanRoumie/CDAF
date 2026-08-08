"""Self-play rollout collection against the real environment (`rl_training_spec.md`
§Self-play, §Batch structure).

One episode = one debate round played by two policies: the LEARNER (the policy being
trained) on its assigned side and an OPPONENT on the other. The environment is the only
legal builder; every action -- learner or opponent -- is sampled through the SAME
legal-action masking the heads already enforce (`ActorCritic.sample_*` -> `LegalActionMask`
-> `env.legal_actions.is_legal`), so no illegal move is ever submitted and the env never
has to reject one.

MDP framing (documented modeling choice): each side sees the round as its OWN sequential
decision problem with the opponent folded into the environment transition. We collect the
LEARNER's decisions only; opponent moves are part of the env dynamics between them. The
terminal binary ballot reward is realized when the round ends and assigned to the
learner's LAST decision; all earlier learner decisions carry reward 0 (the reward is
terminal-only, `env.environment`). Discounting is therefore PER LEARNER-DECISION -- one γ
factor per learner decision, not per raw env step -- which keeps warm-start and PPO value
targets on one clock. γ is SEMANTIC and has been RULED (0.999, explicit user ruling
2026-08-06; see `config.SemanticsConfig.discount`); `compute_gae` still takes it as an
explicit argument and `collect_batch` reads it via `require("discount")`, so an unset γ
would fail loudly rather than default.

TERMINAL REWARD ON VALIDATION FAILURE (constraint check): there is NO reward branch for an
invalid terminal graph. `env.environment._terminate()` ASSERTS `validate_round` and raises
if it fails -- the legal-action generator makes both invalid corners unreachable, so a
failure is an env bug, never a reward. Rollout inherits this: it only ever submits masked-
legal actions, so it never constructs a graph the validator would reject. Confirmed still
true against the current env.

SIDE ASSIGNMENT: random per episode in training (spec: gradients average over the batch,
so variance reduction buys nothing). Mirrored pairs are an EVALUATION concern (`eval.py`),
not here.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch

from judge.config import AFF, NEG
from env import CDAFEnvironment, observe
from policy import ActorCritic

from .config import TrainingConfig


@dataclass
class RolloutStep:
    """One LEARNER decision. `obs` + `state` (a live-then-discarded snapshot) are what PPO
    needs to rebuild the encoder output and the legal-action mask when it re-evaluates the
    action under updated parameters. `old_log_prob` / `old_value` are detached scalars from
    collection time (the PPO ratio's denominator and the value baseline)."""
    obs: dict
    state: object                       # RoundState snapshot at the decision
    action: object
    old_log_prob: float
    old_value: float
    side: str
    slot: str
    action_type: str
    # filled by compute_gae:
    advantage: float = 0.0
    ret: float = 0.0


@dataclass
class Trajectory:
    """One episode's learner decisions plus the terminal outcome. `terminal_reward` is the
    learner side's binary ballot reward (+ shaping bonus if the env's hook is enabled)."""
    steps: List[RolloutStep] = field(default_factory=list)
    learner_side: str = ""
    winner: str = ""
    terminal_reward: float = 0.0
    reward_breakdown: dict = field(default_factory=dict)
    diagnostics: list = field(default_factory=list)
    length: int = 0                     # total env actions in the episode (both sides)


# ---------------------------------------------------------------------------
# episode collection
# ---------------------------------------------------------------------------

def collect_episode(learner: ActorCritic, opponent: ActorCritic, *,
                    learner_side: str, rng, torch_generator: torch.Generator = None,
                    chain_extension_bonus: float = 0.0, capture_round: bool = False):
    """Play one self-play round. `learner_side` is AFF or NEG; the opponent takes the
    other. Sampling is `torch.no_grad` (collection stores detached scalars; PPO recomputes
    with grad later). `chain_extension_bonus` wires the env's shaping HOOK -- default 0.0
    (OFF); the loop passes a nonzero value ONLY when shaping is deliberately enabled.

    Returns the `Trajectory`. If `capture_round=True`, returns `(Trajectory, model.Round)`
    where the Round is the terminal graph materialized via `state.to_round()` (the same path
    `sample_round.py` uses) -- for periodic ROUND VISUALIZATION. Capture only materializes
    the final graph (one `to_round()`), so it is cheap; do it for a FEW episodes, not all."""
    env = CDAFEnvironment(chain_extension_bonus=chain_extension_bonus)
    env.reset()
    traj = Trajectory(learner_side=learner_side)
    steps_taken = 0

    with torch.no_grad():
        while not env.state.terminated:
            side = env.state.current_side
            obs = observe(env.state)
            is_learner = (side == learner_side)
            policy = learner if is_learner else opponent
            enc_out = policy.evaluate(obs)

            if is_learner:
                snapshot = copy.deepcopy(env.state)
                sa, log_prob, _entropy = policy.sample_with_log_prob(
                    enc_out, env.state, generator=torch_generator)
                value = policy.value(enc_out.graph_embedding)
                traj.steps.append(RolloutStep(
                    obs=obs, state=snapshot, action=sa.action,
                    old_log_prob=float(log_prob), old_value=float(value),
                    side=side, slot=env.state.current_slot, action_type=sa.action_type))
            else:
                sa = policy.sample_action(enc_out, env.state, generator=torch_generator)

            _obs, _reward, done, info = env.step(sa.action)
            steps_taken += 1
            if done:
                traj.winner = info["winner"]
                traj.terminal_reward = info["rewards"][learner_side]
                traj.reward_breakdown = info.get("reward_breakdown", {})
                traj.diagnostics = info.get("diagnostics", [])

    traj.length = steps_taken
    if capture_round:
        return traj, env.state.to_round()
    return traj


# ---------------------------------------------------------------------------
# round visualization capture (opt-in, cheap): one representative episode's
# terminal graph, saved builder-loadable (same path as sample_round.py).
# ---------------------------------------------------------------------------

def save_round_json(rnd, path: str, *, role_labels: bool = True) -> str:
    """Serialize a materialized `model.Round` to `path` in the EXACT format the Dash builder
    loads (`model.serialize` -> {"version": 2, "elements": [...]}). The policy is
    content-blind (empty labels); with `role_labels=True` each node's label is set to its
    policy-declared role/type for legibility (the role the introduce chose, not fabricated
    text). Returns `path`."""
    from model import serialize
    if role_labels:
        for el in rnd.elements:
            if hasattr(el, "ntype") and not getattr(el, "label", None):
                el.label = getattr(el, "kind", None) or el.ntype
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    serialize.save(rnd, path)
    return path


def capture_round_to_file(learner: ActorCritic, opponent: ActorCritic, *, learner_side: str,
                          rng, path: str, torch_generator: torch.Generator = None,
                          chain_extension_bonus: float = 0.0, role_labels: bool = True):
    """Play ONE episode and save its terminal graph to `path`, builder-loadable. Returns
    `(Trajectory, path)`. Thin convenience over `collect_episode(capture_round=True)` +
    `save_round_json` for periodic in-training visualization."""
    traj, rnd = collect_episode(
        learner, opponent, learner_side=learner_side, rng=rng,
        torch_generator=torch_generator, chain_extension_bonus=chain_extension_bonus,
        capture_round=True)
    save_round_json(rnd, path, role_labels=role_labels)
    return traj, path


def random_side(rng) -> str:
    """Uniform side assignment for one training episode (spec: random per episode)."""
    return AFF if rng.random() < 0.5 else NEG


# ---------------------------------------------------------------------------
# GAE advantage / return estimation
# ---------------------------------------------------------------------------

def compute_gae(traj: Trajectory, *, discount: float, gae_lambda: float) -> Trajectory:
    """Fill each learner step's `advantage` and `ret` via GAE over the learner's decision
    sequence. Reward is terminal-only: only the last learner step receives
    `terminal_reward`; the bootstrap value past termination is 0 (the episode truly ends).

    `discount` (γ) is SEMANTIC and must be supplied by the caller (never defaulted);
    `gae_lambda` is a mechanical tuning knob."""
    steps = traj.steps
    T = len(steps)
    if T == 0:
        return traj
    adv = 0.0
    for t in reversed(range(T)):
        reward = traj.terminal_reward if t == T - 1 else 0.0
        next_value = 0.0 if t == T - 1 else steps[t + 1].old_value
        delta = reward + discount * next_value - steps[t].old_value
        adv = delta + discount * gae_lambda * adv
        steps[t].advantage = adv
        steps[t].ret = adv + steps[t].old_value
    return traj


# ---------------------------------------------------------------------------
# batch collection
# ---------------------------------------------------------------------------

def collect_batch(learner: ActorCritic, opponent_sampler: Callable[[object], ActorCritic],
                  config: TrainingConfig, *, rng, torch_generator: torch.Generator = None,
                  n_episodes: Optional[int] = None,
                  chain_extension_bonus: float = 0.0) -> List[Trajectory]:
    """Collect a batch of episodes. `opponent_sampler(rng) -> ActorCritic` resolves the
    opponent per episode (e.g. `CheckpointPool.sample_opponent` bound to the learner) --
    this is where the self-play sampling ratio enters. Side is random per episode. GAE is
    applied per trajectory using the SEMANTIC discount (raises if unset).

    `n_episodes` overrides `config.tuning.episodes_per_update` (the smoke test passes a
    trivial count)."""
    discount = float(config.semantics.require("discount"))     # loud if unset
    gae_lambda = config.tuning.gae_lambda
    n = n_episodes if n_episodes is not None else config.tuning.episodes_per_update
    out: List[Trajectory] = []
    for _ in range(n):
        opponent = opponent_sampler(rng)
        side = random_side(rng)
        traj = collect_episode(
            learner, opponent, learner_side=side, rng=rng,
            torch_generator=torch_generator, chain_extension_bonus=chain_extension_bonus)
        compute_gae(traj, discount=discount, gae_lambda=gae_lambda)
        out.append(traj)
    return out


def flatten_steps(trajectories: List[Trajectory]) -> List[RolloutStep]:
    """All learner steps across a batch, flattened -- the unit the PPO minibatcher draws
    from."""
    return [s for t in trajectories for s in t.steps]
