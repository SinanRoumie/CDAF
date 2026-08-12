"""Mechanical-correctness tests for the CDAF training loop (`training/`).

These check WIRING, not training quality (no real run happens this milestone):
  * the factored log-prob / entropy hook (sample == teacher-forced re-eval; unreachable
    demos -> -inf; every corpus demo reachable);
  * config: semantic hyperparameters are unset sentinels that fail loudly, and the run
    gate refuses to start until they are ruled on;
  * imitation warm-start: dataset builds from the corpus, BC minibatch computes finite
    losses with no NaN, critic warm-start REQUIRES the (semantic) discount;
  * PPO surrogate: an update computes finite losses / no NaN and actually moves params;
    `entropy_coef` has no default (must be supplied);
  * rollout: episodes are valid (every learner action legal, terminal reward binary), GAE
    fills finite advantages/returns and requires the discount;
  * checkpoint pool: save/load round-trips and the composition cap is enforced.
"""

from __future__ import annotations

import copy
import math
import os
import random
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

from env import CDAFEnvironment, is_legal, observe, NEW
from env.actions import Introduce, Connect
from judge.config import AFF, NEG
from policy import GraphEncoder, ActorCritic

from training.config import (
    TrainingConfig, TuningConfig, SemanticsConfig, UnsetHyperparameter, UNSET,
)
from training.checkpoint import new_actor_critic, save_checkpoint, load_checkpoint, EncoderSpec
from training.imitation import (
    build_warmstart_dataset, BehaviorCloning, build_examples_from_actions,
)
from training.rollout import collect_episode, compute_gae, collect_batch, random_side
from training.ppo import PPOUpdater
from training.pool import CheckpointPool
from training.loop import SelfPlayTrainer, warm_start
from training.anneal import entropy_coef

from fuzz_env_shell import sample_legal


# --- small fast model + a fully-set (smoke-only) semantics --------------------

_SPEC = EncoderSpec(d_model=32, n_heads=2, n_layers=1, d_global=16)


def _model(seed=0):
    return new_actor_critic(_SPEC, seed=seed)


def _full_semantics() -> SemanticsConfig:
    """A training-READY semantics: every semantic value flows from SemanticsConfig's RULED
    defaults (entropy 0.01/0.0/0.5, pbrs_lambda 0.5, inert_penalty_coef 0.0 (dormant), pool
    20/5/3, self_play_ratio 0.5, discount 0.999). PBRS replaced the flat bonus + anneal
    machinery; nothing is deferred -- the rulings ARE the defaults, so this config is ready."""
    sem = SemanticsConfig()
    assert sem.discount == 0.999          # the ruled value flows through, un-touched
    return sem


def _config(outdir=None, **tuning) -> TrainingConfig:
    t = TuningConfig(minibatch_size=8, epochs_per_batch=2, warmstart_epochs=1,
                     warmstart_minibatch_size=8, episodes_per_update=3, snapshot_interval=1,
                     total_updates=4, **tuning)
    return TrainingConfig(tuning=t, semantics=_full_semantics(), output_dir=outdir)


def _some_state(seed=3, steps=8):
    """A mid-round (state, obs) with several nodes."""
    env = CDAFEnvironment(); env.reset()
    rng = random.Random(seed)
    for _ in range(steps):
        if env.state.terminated:
            break
        a, _ = sample_legal(env.state, rng)
        env.step(a)
    return env.state, observe(env.state)


# --- factored log-prob / evaluate hook ---------------------------------------

def test_sample_log_prob_matches_teacher_forced_reeval():
    ac = _model()
    state, obs = _some_state()
    out = ac.evaluate(obs)
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        sa, logp, ent = ac.sample_with_log_prob(out, state, generator=gen)
        lp2, ent2 = ac.evaluate_action(out, state, sa.action)
    assert torch.allclose(logp, lp2, atol=1e-5)
    assert torch.allclose(ent, ent2, atol=1e-5)
    assert torch.isfinite(logp) and ent.item() >= 0.0


def test_every_corpus_demo_is_reachable():
    """No demonstrated action in the whole warm-start corpus is masked out under the
    factored teacher-forcing decomposition (else BC would see a -inf gradient)."""
    ac = _model()
    ds = build_warmstart_dataset()
    # 4 fixtures (G2/G3/r22/r25) contain now-illegal structural-incoherence constructs and
    # E is a NEG-only chain with no Advocacy/Framework to root it (floating-root
    # restriction); all 5 are legitimately excluded from warm-start; the other 42 convert.
    assert len(ds.fixtures_used) == 42
    assert set(ds.fixtures_skipped) == {"G2", "G3", "r22", "r25", "E"}
    bad = 0
    with torch.no_grad():
        for ex in ds.examples:
            out = ac.evaluate(ex.obs)
            lp, _ = ac.evaluate_action(out, ex.state, ex.action)
            if not torch.isfinite(lp):
                bad += 1
    assert bad == 0, f"{bad} demonstrated actions unreachable under factored masks"


def test_unreachable_action_gives_non_finite_logprob():
    """A structurally-illegal (mask-excluded) action is decodable but scores a NON-FINITE
    log-prob, so BC detects and skips it (its guard is `torch.isfinite`, line ~201) rather
    than training on it. The exact non-finite value depends on the stage: a masked choice
    within an otherwise-legal stage scores -inf; a choice within a stage where EVERY option
    is masked (here every connect edge_type is illegal — support closes a cycle, and both
    attack edge_types are same-side after the masking ruling) scores NaN. BC treats both
    identically."""
    ac = _model()
    env = CDAFEnvironment(); env.reset()
    env.step(Introduce("", "advocacy", NEW, None))    # n1
    env.step(Introduce("", "link", "n1", "support"))  # n2 -> n1
    env.step(Introduce("", "impact", "n2", "support"))  # n3 -> n2
    state = env.state
    s_id = next(nid for nid, n in state.nodes.items() if n.role == "advocacy")
    t_id = next(nid for nid, n in state.nodes.items() if n.role == "impact")
    illegal = Connect(s_id, t_id, "support")           # closes a Support cycle -> illegal
    assert not is_legal(state, illegal)
    out = ac.evaluate(observe(state))
    lp, _ent = ac.evaluate_action(out, state, illegal)
    assert not torch.isfinite(lp), f"expected a non-finite log-prob, got {lp.item()}"


# --- config: fail-loud semantics + run gate ----------------------------------

def test_unset_semantic_bool_raises():
    with pytest.raises(UnsetHyperparameter):
        bool(UNSET)


def test_require_unset_raises_and_set_returns():
    sem = SemanticsConfig()
    sem.pbrs_lambda = UNSET                             # force back to the sentinel
    with pytest.raises(UnsetHyperparameter):
        sem.require("pbrs_lambda")
    sem.pbrs_lambda = 0.5
    assert sem.require("pbrs_lambda") == 0.5


def test_ruled_semantic_values():
    """Every ruled value flows from SemanticsConfig defaults. PBRS replaced the flat bonus:
    pbrs_lambda is the shaping weight; shaping_coef/shaping_enabled/anneal triggers are gone."""
    s = SemanticsConfig()
    assert s.require("discount") == 0.999
    assert s.require("entropy_coef_initial") == 0.01
    assert s.require("entropy_coef_final") == 0.0
    assert s.require("entropy_decay_fraction") == 0.5
    assert s.require("pbrs_lambda") == 0.5               # PBRS shaping weight (replaces shaping_coef)
    assert s.require("inert_penalty_coef") == 0.0        # dormant backstop (no-op priced via cost)
    assert (s.require("pool_cap"), s.require("pool_recent"), s.require("pool_anchors")) == (20, 5, 3)
    assert s.require("self_play_ratio") == 0.5
    # retired flat-bonus / anneal fields no longer exist on the config
    assert not hasattr(s, "shaping_coef")
    assert not hasattr(s, "shaping_enabled")
    assert not hasattr(s, "anneal_trigger_ballot_winrate")


def test_default_config_is_ready_and_fully_ruled():
    """The default config is READY with NOTHING unset: every semantic (incl. pbrs_lambda) is
    ruled. With the shaping-anneal machinery retired there are no deferrable fields."""
    cfg = TrainingConfig()
    assert cfg.is_ready_for_training()
    assert cfg.unset_semantics() == []
    assert cfg.deferred_semantics() == []               # nothing is deferred anymore
    assert cfg.blocking_semantics() == []


def test_unset_semantic_blocks_and_fails_loud():
    """The fail-loud gate is intact after the anneal retirement: forcing any consumed
    semantic (here pbrs_lambda) back to the sentinel makes it block a run and raise on read."""
    sem = SemanticsConfig()
    sem.pbrs_lambda = UNSET
    cfg = TrainingConfig(semantics=sem)
    assert not cfg.is_ready_for_training()
    assert "pbrs_lambda" in cfg.blocking_semantics()
    with pytest.raises(UnsetHyperparameter):
        sem.require("pbrs_lambda")


def test_full_semantics_is_ready():
    assert TrainingConfig(semantics=_full_semantics()).is_ready_for_training()


def test_config_roundtrip_preserves_unset():
    """Round-trip preserves both ruled values and (forced) sentinels. Defaults are all ruled,
    so force one field (pbrs_lambda) back to UNSET to exercise sentinel preservation."""
    sem = SemanticsConfig()
    sem.pbrs_lambda = UNSET                                   # force a sentinel to survive
    cfg = TrainingConfig(semantics=sem)
    d = cfg.to_dict()
    assert d["semantics"]["pbrs_lambda"] == "UNSET"          # sentinel serializes honestly
    assert d["semantics"]["discount"] == 0.999               # a ruled value, serialized as-is
    assert d["semantics"]["inert_penalty_coef"] == 0.0       # ruled 0.0 (dormant), serialized as-is
    back = TrainingConfig.from_dict(d)
    assert "pbrs_lambda" in back.unset_semantics()           # sentinel round-trips
    assert back.semantics.require("discount") == 0.999


def test_train_refuses_with_blocking_unset_semantics(tmp_path):
    """The run gate still refuses when a CONSUMED semantic is unset (force γ back to the
    sentinel to make one blocking)."""
    sem = SemanticsConfig()
    sem.discount = UNSET
    trainer = SelfPlayTrainer(
        config=TrainingConfig(semantics=sem, output_dir=str(tmp_path)), encoder_spec=_SPEC)
    with pytest.raises(UnsetHyperparameter):
        trainer.train(n_updates=1, episodes_per_update=2)


# --- imitation warm-start ----------------------------------------------------

def test_warmstart_dataset_builds():
    ds = build_warmstart_dataset()
    assert len(ds) > 0
    assert len(ds.fixtures_used) == 42          # 47 - 4 masked-construct - 1 unrootable (E)
    ex = ds.examples[0]
    assert ex.side in (AFF, NEG)
    assert ex.remaining_decisions >= 0
    assert ex.terminal_reward in (0.0, 1.0)


def test_bc_minibatch_finite_and_updates_params():
    ac = _model()
    cfg = _config()
    ds = build_warmstart_dataset()
    bc = BehaviorCloning(ac, cfg, train_critic=True)
    before = [p.detach().clone() for p in ac.parameters()]
    stats = bc.train_minibatch(ds.examples[:8])
    assert math.isfinite(stats.actor_loss) and stats.actor_loss >= 0.0
    assert stats.critic_loss is not None and math.isfinite(stats.critic_loss)
    assert stats.unreachable == 0
    after = [p.detach().clone() for p in ac.parameters()]
    assert any(not torch.equal(a, b) for a, b in zip(before, after))  # params moved


def test_bc_critic_runs_with_ruled_discount():
    """Previously blocked by the fail-loud UNSET γ; now runs end-to-end on the REAL ruled
    value (0.999) inherited from the config -- no throwaway."""
    ac = _model()
    cfg = _config()                                    # _full_semantics -> ruled γ = 0.999
    bc = BehaviorCloning(ac, cfg, train_critic=True)
    assert bc.discount == 0.999                        # the ruled value, not a test stub
    ds = build_warmstart_dataset()
    stats = bc.train_minibatch(ds.examples[:8])
    assert stats.critic_loss is not None and math.isfinite(stats.critic_loss)


def test_bc_critic_fails_loud_if_discount_forced_unset():
    """The fail-loud guarantee is intact: if γ were ever un-ruled, critic warm-start
    refuses rather than defaulting."""
    sem = SemanticsConfig()
    sem.discount = UNSET                               # force γ back to the sentinel
    cfg = TrainingConfig(tuning=TuningConfig(), semantics=sem)
    with pytest.raises(UnsetHyperparameter):
        BehaviorCloning(_model(), cfg, train_critic=True)


def test_bc_actor_only_needs_no_semantics():
    """Actor behavior cloning runs with NO semantic parameter set at all."""
    cfg = TrainingConfig(tuning=TuningConfig(warmstart_minibatch_size=8), semantics=SemanticsConfig())
    ds = build_warmstart_dataset()
    bc = BehaviorCloning(_model(), cfg, train_critic=False)
    stats = bc.train_minibatch(ds.examples[:8])
    assert math.isfinite(stats.actor_loss) and stats.critic_loss is None


# --- rollout -----------------------------------------------------------------

def test_collect_episode_valid_trajectory():
    ac = _model()
    rng = random.Random(0)
    gen = torch.Generator().manual_seed(0)
    traj = collect_episode(ac, ac, learner_side=AFF, rng=rng, torch_generator=gen)
    assert traj.winner in (AFF, NEG)
    assert traj.terminal_reward in (0.0, 1.0)
    assert len(traj.steps) > 0
    # every recorded learner action was legal at its snapshot state
    for s in traj.steps:
        assert s.side == AFF
        assert is_legal(s.state, s.action)
        assert math.isfinite(s.old_log_prob) and math.isfinite(s.old_value)


def test_compute_gae_fills_finite_targets():
    ac = _model()
    rng = random.Random(1)
    traj = collect_episode(ac, ac, learner_side=NEG, rng=rng)
    gamma = SemanticsConfig().require("discount")       # the ruled γ = 0.999
    compute_gae(traj, discount=gamma, gae_lambda=0.95)
    for s in traj.steps:
        assert math.isfinite(s.advantage) and math.isfinite(s.ret)


def test_collect_batch_runs_with_ruled_discount():
    """Previously blocked by the fail-loud UNSET γ; collect_batch now reads the ruled
    0.999 via require() and produces valid trajectories with finite GAE targets."""
    cfg = _config()                                     # ruled γ = 0.999
    ac = _model()
    batch = collect_batch(ac, lambda r: ac, cfg, rng=random.Random(0), n_episodes=2)
    assert len(batch) == 2
    for t in batch:
        for s in t.steps:
            assert math.isfinite(s.advantage) and math.isfinite(s.ret)


def test_collect_batch_fails_loud_if_discount_forced_unset():
    """Fail-loud intact: force γ back to the sentinel and collect_batch refuses."""
    sem = SemanticsConfig()
    sem.discount = UNSET
    cfg = TrainingConfig(tuning=TuningConfig(), semantics=sem)
    ac = _model()
    with pytest.raises(UnsetHyperparameter):
        collect_batch(ac, lambda r: ac, cfg, rng=random.Random(0), n_episodes=1)


def test_random_side_covers_both():
    rng = random.Random(0)
    seen = {random_side(rng) for _ in range(50)}
    assert seen == {AFF, NEG}


# --- PPO ---------------------------------------------------------------------

def test_ppo_update_finite_and_moves_params():
    ac = _model()
    cfg = _config()
    rng = random.Random(2)
    gen = torch.Generator().manual_seed(2)
    batch = collect_batch(ac, lambda r: ac, cfg, rng=rng, torch_generator=gen, n_episodes=4)
    steps = [s for t in batch for s in t.steps]
    assert steps
    updater = PPOUpdater(ac, cfg)
    before = [p.detach().clone() for p in ac.parameters()]
    stats = updater.update(steps, entropy_coef=0.05, rng=rng)
    assert stats
    for st in stats:
        assert math.isfinite(st.policy_loss) and math.isfinite(st.value_loss)
        assert math.isfinite(st.entropy) and math.isfinite(st.approx_kl)
    after = [p.detach().clone() for p in ac.parameters()]
    assert any(not torch.equal(a, b) for a, b in zip(before, after))


def test_ppo_entropy_coef_has_no_default():
    """entropy_coef is semantic; `update` must be given it explicitly (keyword-only)."""
    ac = _model()
    updater = PPOUpdater(ac, _config())
    with pytest.raises(TypeError):
        updater.update([])                     # missing required keyword-only entropy_coef


# --- checkpoint pool ---------------------------------------------------------

def test_checkpoint_roundtrip(tmp_path):
    ac = _model(seed=1)
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, ac, encoder_spec=_SPEC, meta={"stage": "test"})
    ac2, meta = load_checkpoint(path)
    assert meta["stage"] == "test"
    # identical outputs on the same observation
    _state, obs = _some_state()
    with torch.no_grad():
        v1 = ac.value(ac.evaluate(obs).graph_embedding)
        v2 = ac2.value(ac2.evaluate(obs).graph_embedding)
    assert torch.allclose(v1, v2, atol=1e-6)


def test_pool_snapshot_and_cap(tmp_path):
    sem = _full_semantics()
    sem.pool_cap, sem.pool_recent, sem.pool_anchors = 3, 2, 1
    pool = CheckpointPool(directory=str(tmp_path / "pool"), semantics=sem)
    ac = _model()
    for u in range(5):
        pool.snapshot(ac, update=u, encoder_spec=_SPEC)
    assert len(pool.entries) == 3                        # 1 anchor + 2 recent
    updates = sorted(e.update for e in pool.entries)
    assert updates[0] == 0 and updates[-1] == 4          # anchor (first) + latest kept
    assert any(e.is_anchor for e in pool.entries)
    # evicted files are gone; kept files exist
    for e in pool.entries:
        assert os.path.exists(e.path)


def test_pool_caches_deserialized_opponents(tmp_path):
    """load_opponent deserializes each checkpoint ONCE and reuses the in-memory policy
    across calls (the validation-v1 per-episode-reload fix)."""
    sem = _full_semantics()
    pool = CheckpointPool(directory=str(tmp_path / "pool"), semantics=sem)
    ac = _model()
    e = pool.snapshot(ac, update=0, encoder_spec=_SPEC)
    first = pool.load_opponent(e)
    second = pool.load_opponent(e)
    assert first is second                              # same object, not re-loaded
    assert e.path in pool._cache


def test_pool_eviction_drops_cache(tmp_path):
    """An evicted checkpoint is removed from the in-memory cache as well as from disk."""
    sem = _full_semantics()
    sem.pool_cap, sem.pool_recent, sem.pool_anchors = 3, 2, 1
    pool = CheckpointPool(directory=str(tmp_path / "pool"), semantics=sem)
    ac = _model()
    entries = [pool.snapshot(ac, update=u, encoder_spec=_SPEC) for u in range(3)]
    pool.load_opponent(entries[1])                      # cache a non-anchor that will evict
    assert entries[1].path in pool._cache
    for u in range(3, 6):                               # push past the cap -> evict oldest non-anchors
        pool.snapshot(ac, update=u, encoder_spec=_SPEC)
    live = {e.path for e in pool.entries}
    assert all(p in live for p in pool._cache), "cache holds a path no longer in the pool"


def test_capture_round_returns_loadable_round(tmp_path):
    """collect_episode(capture_round=True) yields a materialized Round; save_round_json
    writes it in the builder's exact load format."""
    import json
    from model import serialize as mser
    from training.rollout import save_round_json
    ac = _model()
    traj, rnd = collect_episode(ac, ac, learner_side=AFF, rng=random.Random(0),
                                capture_round=True)
    assert rnd is not None and len(rnd.elements) >= 0
    path = str(tmp_path / "rounds" / "round_update000.json")
    save_round_json(rnd, path)
    raw = json.load(open(path))
    assert raw["version"] == 2 and "elements" in raw
    reloaded = mser.elements_from_round(mser.from_dict(raw))   # builder's exact load path
    assert isinstance(reloaded, list)


def test_target_kl_early_stops_ppo_epochs():
    """A tiny target_kl trips the epoch early-stop; None runs all epochs."""
    ac = _model()
    rng = random.Random(3)
    cfg = _config()
    batch = collect_batch(ac, lambda r: ac, cfg, rng=rng, n_episodes=4)
    steps = [s for t in batch for s in t.steps]
    # tiny threshold -> stop after the first epoch
    cfg_stop = _config()
    cfg_stop.tuning.target_kl = 1e-9
    cfg_stop.tuning.epochs_per_batch = 4
    up = PPOUpdater(ac, cfg_stop)
    up.update(steps, entropy_coef=0.01, rng=rng)
    assert up.last_early_stopped and up.last_epochs_run == 1
    # disabled -> all epochs run
    cfg_all = _config()
    cfg_all.tuning.target_kl = None
    cfg_all.tuning.epochs_per_batch = 3
    up2 = PPOUpdater(ac, cfg_all)
    up2.update(steps, entropy_coef=0.01, rng=rng)
    assert not up2.last_early_stopped and up2.last_epochs_run == 3


def test_pool_sample_opponent_returns_policy(tmp_path):
    sem = _full_semantics()
    pool = CheckpointPool(directory=str(tmp_path / "pool"), semantics=sem)
    ac = _model()
    rng = random.Random(0)
    # empty pool -> always self
    assert pool.sample_opponent(ac, rng) is ac
    pool.snapshot(ac, update=0, encoder_spec=_SPEC)
    opp = pool.sample_opponent(ac, rng)
    assert isinstance(opp, ActorCritic)


# --- schedules ---------------------------------------------------------------

def test_entropy_schedule_decays():
    """Ruled schedule: 0.01 -> 0.0 linearly over the first half of training."""
    sem = _full_semantics()
    e_start = entropy_coef(0, 300, sem)
    e_mid = entropy_coef(50, 300, sem)
    e_end = entropy_coef(299, 300, sem)
    assert e_start == pytest.approx(0.01)
    assert e_end == pytest.approx(0.0)
    assert e_start > e_mid > e_end
    assert entropy_coef(150, 300, sem) == pytest.approx(0.0)   # zero by the halfway point


def test_apply_pbrs_side_relative_and_terminal_boundary():
    """PBRS (replaces the retired flat bonus + anneal controller): apply_pbrs fills per-step
    shaping F_t = λ(γ·Φ_L(s') - Φ_L(s)), side-relative (Φ_NEG = -Φ_AFF), with Φ(terminal)=0
    on the last step -- so Σ F_t telescopes to -λ·Φ_L(s_0) (γ=1 here), the invariance identity."""
    from training.rollout import apply_pbrs, Trajectory, RolloutStep

    def mk(phi, side="AFF"):
        return RolloutStep(obs={}, state=None, action=None, old_log_prob=0.0, old_value=0.0,
                           side=side, slot="1AC", action_type="introduce", phi=phi)

    aff = Trajectory(learner_side="AFF", steps=[mk(0.0), mk(0.4), mk(0.8)])
    apply_pbrs(aff, pbrs_lambda=0.5, discount=1.0)
    # F0=0.5(0.4-0); F1=0.5(0.8-0.4); F2=0.5(0-0.8)  [last -> Φ(terminal)=0]
    assert [s.shaping for s in aff.steps] == pytest.approx([0.2, 0.2, -0.4])
    assert sum(s.shaping for s in aff.steps) == pytest.approx(0.0)   # Φ_L(s0)=0 -> telescopes to 0

    neg = Trajectory(learner_side="NEG", steps=[mk(0.0, "NEG"), mk(0.4, "NEG"), mk(0.8, "NEG")])
    apply_pbrs(neg, pbrs_lambda=0.5, discount=1.0)   # Φ_NEG = -Φ -> signs flip
    assert [s.shaping for s in neg.steps] == pytest.approx([-0.2, -0.2, 0.4])

    # λ=0 leaves shaping at 0 (byte-identical to no PBRS)
    off = Trajectory(learner_side="AFF", steps=[mk(0.4), mk(0.8)])
    apply_pbrs(off, pbrs_lambda=0.0, discount=1.0)
    assert all(s.shaping == 0.0 for s in off.steps)
