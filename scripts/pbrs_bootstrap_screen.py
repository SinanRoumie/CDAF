"""PBRS bootstrap-robustness screen -- TRACKED driver (transfers via git clone; cloud-ready).

Runs the §Validation (PBRS) bootstrap screen: per seed, a bootstrap-only PPO run at
pbrs_lambda (from config) and a chosen episodes-per-update E, measuring last-15-update mean
win-rate + extension-survival against the v3-like PASS bar (win >= 0.10 AND survival >= 0.20).
No decay phase -- PBRS is never withdrawn.

This is the tracked home of the former runs/pbrs_screen/run.py (which was gitignored). It
writes all OUTPUTS into the gitignored `runs/pbrs_screen/` so nothing large lands in a tracked
path. Parameters are env-overridable so the E re-screen is a one-line invocation and CI/cloud
smoke runs are trivial:

    PBRS_E=100 python scripts/pbrs_bootstrap_screen.py           # the E=100 re-screen
    PBRS_E=4 PBRS_UPDATES=2 PBRS_SEEDS=0 python scripts/...       # fast local sanity

Warm-start: reuses runs/screen_v1/warmstart.pt if present (local, for comparability with the
earlier screens); otherwise builds a fresh seed-0 BC warm-start (e.g. on a clean cloud clone,
where runs/ is empty) -- BC is deterministic, so the init is identical either way.
"""

from __future__ import annotations

import json
import os
import random
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # scripts/ -> repo root
sys.path.insert(0, ROOT)

from judge import judge as run_judge
from judge.config import AFF, NEG
from training.config import TrainingConfig, TuningConfig, SemanticsConfig
from training.checkpoint import new_actor_critic, save_checkpoint, load_checkpoint, EncoderSpec
from training.imitation import build_warmstart_dataset, BehaviorCloning
from training.rollout import collect_episode, apply_pbrs, compute_gae, flatten_steps, random_side
from training.ppo import PPOUpdater
from training.pool import CheckpointPool
from training.anneal import entropy_coef
from training.metrics import batch_metrics

# --- env-overridable knobs (isolate the E variable; pbrs_lambda stays at the ruled 0.5) ---
E = int(os.environ.get("PBRS_E", 20))              # episodes per update (the lever under test)
BOOTSTRAP_UPDATES = int(os.environ.get("PBRS_UPDATES", 40))
SEEDS = [int(x) for x in os.environ.get("PBRS_SEEDS", "0,1,2,3,4").split(",")]
SNAPSHOT_INTERVAL = 15
ENTROPY_TOTAL = 80                                  # same schedule as prior screens' bootstrap
WIN_FLOOR = 0.10
SURV_FLOOR = 0.20
OUTPUT_DIR = os.path.join(ROOT, "runs", "pbrs_screen")   # gitignored
SHARED_WARMSTART = os.path.join(ROOT, "runs", "screen_v1", "warmstart.pt")


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def run_seed(seed, warmstart_path, spec, cfg):
    ac, _ = load_checkpoint(warmstart_path)
    pool = CheckpointPool(directory=os.path.join(OUTPUT_DIR, f"pool_seed{seed}"), semantics=cfg.semantics)
    updater = PPOUpdater(ac, cfg)
    rng = random.Random(seed); gen = torch.Generator().manual_seed(seed)
    discount = float(cfg.semantics.require("discount"))
    pbrs_lambda = float(cfg.semantics.require("pbrs_lambda"))
    gl = cfg.tuning.gae_lambda
    rows = []
    for u in range(BOOTSTRAP_UPDATES):
        ecoef = entropy_coef(u, ENTROPY_TOTAL, cfg.semantics)
        counts = {"self": 0, "pool": 0}

        def sampler(r, _ac=ac, _pool=pool, _c=counts):
            opp = _pool.sample_opponent(_ac, r); _c["self" if opp is _ac else "pool"] += 1
            return opp

        trajs = []; rounds_found = rounds_ext = 0; phis = []
        for _ in range(E):
            opp = sampler(rng); side = random_side(rng)
            traj, rnd = collect_episode(ac, opp, learner_side=side, rng=rng,
                                        torch_generator=gen, capture_round=True)
            apply_pbrs(traj, pbrs_lambda=pbrs_lambda, discount=discount)
            compute_gae(traj, discount=discount, gae_lambda=gl)
            trajs.append(traj)
            _b, trace = run_judge(rnd); f = e = 0
            for r in trace:
                if getattr(r, "kind", None) == "CHAIN" and getattr(r, "side", None) == AFF:
                    f += 1
                    if r.extended and r.in_scope and r.sign == 1:
                        e += 1
            rounds_found += 1 if f > 0 else 0
            rounds_ext += 1 if e > 0 else 0
            phis.extend(s.phi for s in traj.steps)
        steps = flatten_steps(trajs)
        updater.update(steps, entropy_coef=ecoef, rng=rng)
        m = batch_metrics(trajs)
        rows.append({"u": u, "win": m["ballot_win_rate_aff"], "surv": rounds_ext / E,
                     "found": rounds_found / E, "phi_mean": _mean(phis)})
        if u % 10 == 0 or u == BOOTSTRAP_UPDATES - 1:
            print(f"    seed{seed} u{u:02d} win={rows[-1]['win']:.2f} surv={rows[-1]['surv']:.2f} "
                  f"found={rows[-1]['found']:.2f} phi_mean={rows[-1]['phi_mean']:+.3f} ent={ecoef:.4f}",
                  flush=True)

    last15 = rows[-15:]; first15 = rows[:15]
    win = _mean(r["win"] for r in last15); surv = _mean(r["surv"] for r in last15)
    passed = win >= WIN_FLOOR and surv >= SURV_FLOOR
    phi_first = _mean(r["phi_mean"] for r in first15); phi_last = _mean(r["phi_mean"] for r in last15)
    win_first = _mean(r["win"] for r in first15)
    divergence = (phi_last - phi_first > 0.10) and (win - win_first <= 0.02)
    phi_trend = ("climbing" if phi_last - phi_first > 0.05
                 else "declining" if phi_last - phi_first < -0.05 else "stable")
    print(f"  seed{seed}: last-15 win={win:.3f} surv={surv:.3f} -> {'PASS' if passed else 'FAIL'} | "
          f"Φ {phi_first:+.3f}->{phi_last:+.3f} ({phi_trend}) divergence={divergence}", flush=True)
    return {"seed": seed, "last15_win": win, "last15_surv": surv, "passed": passed,
            "phi_first15": phi_first, "phi_last15": phi_last, "phi_trend": phi_trend,
            "divergence": divergence, "rows": rows}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cfg = TrainingConfig(
        tuning=TuningConfig(minibatch_size=256, epochs_per_batch=3, warmstart_epochs=20,
                            warmstart_minibatch_size=128, episodes_per_update=E,
                            snapshot_interval=SNAPSHOT_INTERVAL, total_updates=120, seed=0),
        semantics=SemanticsConfig(), output_dir=OUTPUT_DIR)
    spec = EncoderSpec()
    if os.path.exists(SHARED_WARMSTART):
        print(f"[warmstart] reusing {SHARED_WARMSTART} (shared seed-0 BC init)", flush=True)
        warmstart_path = SHARED_WARMSTART
    else:
        print("[warmstart] building fresh seed-0 BC (no shared warmstart -- e.g. clean clone) ...",
              flush=True)
        ac = new_actor_critic(spec, seed=0); ds = build_warmstart_dataset()
        bc = BehaviorCloning(ac, cfg, train_critic=True); rng = random.Random(0)
        for _ in range(20): bc.train_epoch(ds, rng=rng)
        warmstart_path = os.path.join(OUTPUT_DIR, "warmstart.pt")
        save_checkpoint(warmstart_path, ac, encoder_spec=spec, meta={"stage": "warmstart"})
    print(f"[config] PBRS bootstrap screen: pbrs_lambda={cfg.semantics.require('pbrs_lambda')} "
          f"E={E} seeds={SEEDS} bootstrap={BOOTSTRAP_UPDATES} | bar: win>={WIN_FLOOR} "
          f"surv>={SURV_FLOOR} | bootstrap-only (no decay)", flush=True)

    results = [run_seed(s, warmstart_path, spec, cfg) for s in SEEDS]

    print("\n" + "=" * 66, flush=True)
    print(f"PBRS λ={cfg.semantics.require('pbrs_lambda')} E={E} BOOTSTRAP-ROBUSTNESS SUMMARY", flush=True)
    print("=" * 66, flush=True)
    for r in results:
        print(f"  seed{r['seed']}: win={r['last15_win']:.3f} surv={r['last15_surv']:.3f} "
              f"-> {'PASS' if r['passed'] else 'FAIL'} | Φ {r['phi_first15']:+.3f}->{r['phi_last15']:+.3f} "
              f"({r['phi_trend']}) divergence={r['divergence']}")
    n_pass = sum(r["passed"] for r in results)
    print(f"  ==> {n_pass}/{len(SEEDS)} seeds PASS")
    with open(os.path.join(OUTPUT_DIR, f"pbrs_screen_E{E}_results.json"), "w") as fh:
        json.dump({"pbrs_lambda": cfg.semantics.require("pbrs_lambda"), "E": E, "seeds": SEEDS,
                   "results": [{k: v for k, v in r.items() if k != "rows"} for r in results],
                   "full_rows": {r["seed"]: r["rows"] for r in results}}, fh, indent=2)
    print(f"[done] wrote {OUTPUT_DIR}/pbrs_screen_E{E}_results.json", flush=True)


if __name__ == "__main__":
    main()
