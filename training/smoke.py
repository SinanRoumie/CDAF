"""SMOKE TEST -- prove the training loop WIRES UP AND RUNS WITHOUT CRASHING. This is NOT a
real training run and NOT a quality check.

It runs a TRIVIALLY CAPPED pass of every stage -- a couple of warm-start minibatches, two
PPO updates of a handful of episodes each, one pool snapshot -- purely to confirm no shape
error, no NaN, and no dead wiring. Nothing here is tuned; the step counts are as small as
possible while still touching each code path once.

As of the 2026-08-08 rulings, every CONSUMED semantic value is the real ruled default from
`SemanticsConfig` -- this smoke run exercises the actual γ / entropy schedule / pool /
ratio, not throwaways. Shaping is now ON (coef 0.1) with the anneal triggers ARMED at
provisional values, and the inert-action penalty is ON (0.01), so this smoke also exercises
the shaping bonus, the anneal controller, and the inert-penalty path end-to-end. Only the
TUNING/throughput knobs are shrunk here to keep the smoke trivial.

Run:  python -m training.smoke     (or  python training/smoke.py)
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.config import TrainingConfig, TuningConfig, SemanticsConfig
from training.checkpoint import new_actor_critic, EncoderSpec
from training.loop import SelfPlayTrainer, warm_start
from training.imitation import build_warmstart_dataset


def _smoke_config(outdir: str) -> TrainingConfig:
    """A capped SMOKE config. As of the 2026-08-08 rulings, ALL consumed semantic values
    are the REAL ruled defaults from SemanticsConfig (no throwaways) -- shaping is ON with
    the anneal triggers armed and the inert penalty ON, all exercised end-to-end. Only the
    TUNING/throughput knobs are shrunk here (tiny batch, 2 updates) to keep the smoke
    trivial; those are mine to set."""
    tuning = TuningConfig(
        minibatch_size=8, epochs_per_batch=2, warmstart_epochs=1,
        warmstart_minibatch_size=8, episodes_per_update=3, snapshot_interval=1,
        total_updates=2)
    return TrainingConfig(tuning=tuning, semantics=SemanticsConfig(), output_dir=outdir)


def main():
    print("=" * 72)
    print("TRAINING-LOOP SMOKE TEST -- CAPPED, NOT A REAL RUN")
    print("  (2 PPO updates x 3 episodes, 1 warm-start epoch capped to 2 minibatches)")
    print("=" * 72)

    # tiny/fast architecture -- smoke only
    spec = EncoderSpec(d_model=32, n_heads=2, n_layers=1, d_global=16)

    with tempfile.TemporaryDirectory() as outdir:
        config = _smoke_config(outdir)
        assert config.is_ready_for_training(), \
            f"smoke config should have all semantics set: {config.unset_semantics()}"
        print(f"[config] all semantics RULED (γ={config.semantics.require('discount')}, "
              f"entropy {config.semantics.require('entropy_coef_initial')}->"
              f"{config.semantics.require('entropy_coef_final')}, "
              f"pool {config.semantics.require('pool_cap')}/{config.semantics.require('pool_recent')}/"
              f"{config.semantics.require('pool_anchors')}); "
              f"PBRS pbrs_lambda={config.semantics.require('pbrs_lambda')}, inert_penalty="
              f"{config.semantics.require('inert_penalty_coef')} (dormant)")

        # (1) warm-start dataset builds from the corpus.
        ds = build_warmstart_dataset()
        print(f"[warmstart] dataset: {len(ds)} examples from {len(ds.fixtures_used)} "
              f"fixtures ({len(ds.fixtures_skipped)} skipped)")

        # (2) actor+critic warm-start, capped to 2 minibatches/epoch.
        ac = new_actor_critic(spec, seed=0)
        hist = warm_start(ac, config, dataset=ds, train_critic=True,
                          epochs=1, max_minibatches_per_epoch=2, seed=0)
        s = hist[0][0]
        print(f"[warmstart] first minibatch: actor_loss={s.actor_loss:.4f} "
              f"critic_loss={s.critic_loss} entropy={s.entropy:.4f} "
              f"n={s.n_examples} unreachable={s.unreachable}")

        # (3) self-play PPO: 2 capped updates + a pool snapshot each.
        trainer = SelfPlayTrainer(config=config, ac=ac, encoder_spec=spec)
        history = trainer.train(warmstart=False, n_updates=2, episodes_per_update=3, seed=1)
        for m in history:
            print(f"[ppo] update={m['update']} winrate_aff={m['ballot_win_rate_aff']:.2f} "
                  f"pol_loss={m['mean_policy_loss']:.4f} val_loss={m['mean_value_loss']:.4f} "
                  f"ent={m['mean_entropy']:.4f} kl={m['mean_approx_kl']:.4f} "
                  f"ent_coef={m['entropy_coef']:.4f} pbrs_lambda={m['pbrs_lambda']} "
                  f"inert/ep={m['mean_inert_per_episode']:.2f} {m['inert_by_kind']}")
        print(f"[pool] {len(trainer.pool.entries)} checkpoint(s) in pool")

    print("=" * 72)
    print("SMOKE TEST PASSED -- loop wiring runs end to end without crashing.")
    print("This proved NO crash on a trivial cap; it trained NOTHING meaningful.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
