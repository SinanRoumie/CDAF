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
from collections import Counter

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # scripts/ -> repo root
sys.path.insert(0, ROOT)

from judge import judge as run_judge
from judge.config import AFF, NEG
from training.config import TrainingConfig, TuningConfig, SemanticsConfig
from training.checkpoint import new_actor_critic, save_checkpoint, load_checkpoint, EncoderSpec
from training.imitation import build_warmstart_dataset, BehaviorCloning
from training.rollout import (collect_episode, apply_pbrs, compute_gae, flatten_steps,
                              random_side, save_round_json)
from warmstart.convert import _reason_class
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
OUTPUT_DIR = os.environ.get("PBRS_OUTDIR", os.path.join(ROOT, "runs", "pbrs_screen"))  # gitignored
# Screen B toggle: opening unlock-curriculum ON for the LEARNER's rollouts (policy-layer
# mask; rl_training_spec §Opening curriculum). Screen A leaves it OFF (default).
CURRICULUM = os.environ.get("PBRS_CURRICULUM", "0") == "1"
PROBE_EPISODES = int(os.environ.get("PBRS_PROBE_EPISODES", "200"))
# Round export (for human review in the Dash builder). When PBRS_EXPORT_DIR is set, rounds
# from the LAST 15 updates are saved as builder-loadable oracle-format JSON + a sidecar, in
# three disjoint buckets, capped per seed per bucket. SCREEN_TAG labels files (e.g. A / A2).
EXPORT_DIR = os.environ.get("PBRS_EXPORT_DIR")
SCREEN_TAG = os.environ.get("PBRS_SCREEN_TAG", "A")
EXPORT_CAP = int(os.environ.get("PBRS_EXPORT_CAP", "4"))     # per bucket, per seed


_ATTACK_KINDS = {"defensive_attack", "offensive_attack"}


def _analyze_dead_chains(rnd, trace):
    """For every AFF chain the judge FOUND but did NOT credit as surviving (not
    extended-and-in_scope-and-sign==1), return one record with WHY it died:
      - collapse_reason: the judge's own reason (extension_fail / defensive_kill / sign_flip /
        unresolved_sign / no_link_premise / unrooted_disad / ...), or None.
      - cause: collapse_reason if set, else the sub-cause from the flags (not_extended /
        out_of_scope / sign_not_positive / unknown).
      - death_speech: the EXTENSION_FAIL missing_speech for this chain, if any.
      - neg_contested: True iff some NEG-owned ATTACK edge lands on a member of the chain's
        AFF same-side Support component (the crux: did a NEG argument actually hit this chain,
        or did it die with no NEG action against it).
      - responsible / responsible_side: the node the judge blames, and whose side it is.
    """
    nodes = {n.id: n for n in rnd.nodes}
    adj = {}
    for e in rnd.edges:
        if getattr(e, "kind", None) == "support":
            a, b = nodes.get(e.source), nodes.get(e.target)
            if a is not None and b is not None and a.side == b.side == AFF:
                adj.setdefault(e.source, set()).add(e.target)
                adj.setdefault(e.target, set()).add(e.source)

    def component(x):
        seen, stack = {x}, [x]
        while stack:
            c = stack.pop()
            for nb in adj.get(c, ()):
                if nb not in seen:
                    seen.add(nb); stack.append(nb)
        return seen

    extfail = {getattr(r, "chain_id", None): getattr(r, "missing_speech", None)
               for r in trace if getattr(r, "kind", None) == "EXTENSION_FAIL"}
    out = []
    for r in trace:
        if getattr(r, "kind", None) != "CHAIN" or getattr(r, "side", None) != AFF:
            continue
        if r.extended and r.in_scope and r.sign == 1:
            continue                                        # survived -> not dead
        cr = getattr(r, "collapse_reason", None)
        cause = cr or ("not_extended" if not r.extended else
                       "out_of_scope" if not r.in_scope else
                       "sign_not_positive" if r.sign != 1 else "unknown")
        chain_id = getattr(r, "chain_id", "") or ""
        impact_id = chain_id.split(":", 1)[1] if ":" in chain_id else None
        members = component(impact_id) if impact_id in nodes else set()
        neg = False
        for e in rnd.edges:
            if getattr(e, "kind", None) in _ATTACK_KINDS:
                sn, tn = nodes.get(e.source), nodes.get(e.target)
                if sn and tn and ((e.source in members and tn.side == NEG)
                                  or (e.target in members and sn.side == NEG)):
                    neg = True
                    break
        resp = getattr(r, "responsible", None)
        out.append({
            "chain_id": chain_id, "collapse_reason": cr, "cause": cause,
            "death_speech": extfail.get(chain_id), "neg_contested": neg,
            "responsible": resp,
            "responsible_side": (nodes[resp].side if resp in nodes else None)})
    return out


def _export_round(rnd, trace, winner, found, survived, seed, u, ep, counts, dead_analysis):
    """Save one round (oracle-format JSON) + a sidecar, if its bucket is not yet full.
    Buckets are DISJOINT: affwin (AFF won); nearmiss (AFF found offense but did NOT win);
    negwin (NEG won with no AFF offense found)."""
    if not EXPORT_DIR:
        return
    bucket = "affwin" if winner == AFF else ("nearmiss" if found else "negwin")
    if counts[bucket] >= EXPORT_CAP:
        return
    counts[bucket] += 1
    base = f"{SCREEN_TAG}_{bucket}_seed{seed}_u{u:02d}_ep{ep:03d}"
    save_round_json(rnd, os.path.join(EXPORT_DIR, base + ".json"))
    aff_chains = [{
        "chain_id": getattr(r, "chain_id", None), "extended": r.extended,
        "in_scope": r.in_scope, "sign": r.sign, "mag": getattr(r, "mag", None),
        "delta": getattr(r, "delta", None),
        "collapse_reason": getattr(r, "collapse_reason", None)}
        for r in trace if getattr(r, "kind", None) == "CHAIN"
        and getattr(r, "side", None) == AFF]
    with open(os.path.join(EXPORT_DIR, base + ".sidecar.json"), "w") as fh:
        json.dump({
            "screen": SCREEN_TAG, "bucket": bucket, "seed": seed, "update": u, "episode": ep,
            "verdict": {"winner": winner, "reason_class": _reason_class(trace)},
            "offense_found": bool(found), "offense_survived": bool(survived),
            "aff_chains": aff_chains,
            "dead_chain_analysis": dead_analysis}, fh, indent=2)
SHARED_WARMSTART = os.path.join(ROOT, "runs", "screen_v1", "warmstart.pt")


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


# --- 1AC construction probe (Screen B): what does AFF build after the fixed opening? ---

def _support_adj(state):
    adj = {}
    for e in state.edges:
        if e.edge_type == "support":
            adj.setdefault(e.source, set()).add(e.target)
            adj.setdefault(e.target, set()).add(e.source)
    return adj


def _aff_1ac_scoring_chain_exists(state) -> bool:
    """True iff, by end of the 1AC, an AFF advocacy->link->impact chain exists connected
    over Support edges (direction-agnostic, as the judge reads it)."""
    advs = {nid for nid, r in state.nodes.items() if r.role == "advocacy"}
    links = {nid for nid, r in state.nodes.items() if r.role == "link"}
    impacts = [nid for nid, r in state.nodes.items()
               if r.role == "impact" and r.owner == "AFF"]
    if not advs or not links or not impacts:
        return False
    adj = _support_adj(state)
    for im in impacts:                                  # support-reachable link AND advocacy?
        seen, stack, hit_link, hit_adv = {im}, [im], False, False
        while stack:
            cur = stack.pop()
            if cur in links:
                hit_link = True
            if cur in advs:
                hit_adv = True
            for nb in adj.get(cur, ()):
                if nb not in seen:
                    seen.add(nb); stack.append(nb)
        if hit_link and hit_adv:
            return True
    return False


def _probe_1ac(ac, n_episodes, gen):
    """Roll the trained learner as AFF through the 1AC only; record the role at 1AC moves
    2-4 (move 1 is the curriculum-fixed advocacy) and how often a complete advocacy->link->
    impact scoring chain exists by end of the 1AC."""
    from collections import Counter
    from env import CDAFEnvironment
    from env.observation import observe
    move_role = {2: Counter(), 3: Counter(), 4: Counter()}
    end_speech_at = Counter()                           # how many introduces before end_speech
    chain = 0
    for _ in range(n_episodes):
        env = CDAFEnvironment(); env.reset()
        intro_roles = []
        while env.state.current_slot == "1AC" and not env.state.terminated:
            out = ac.evaluate(observe(env.state))
            sa = ac.sample_action(out, env.state, generator=gen)
            a = sa.action
            tname = type(a).__name__
            if tname == "Introduce":
                intro_roles.append(a.role)
            env.step(a)
            if tname == "EndSpeech":
                break
        end_speech_at[len(intro_roles)] += 1
        for k in (2, 3, 4):
            if len(intro_roles) >= k:
                move_role[k][intro_roles[k - 1]] += 1
            else:
                move_role[k]["<none>"] += 1
        chain += 1 if _aff_1ac_scoring_chain_exists(env.state) else 0
    return {
        "move2_roles": dict(move_role[2]), "move3_roles": dict(move_role[3]),
        "move4_roles": dict(move_role[4]),
        "intro_count_hist": dict(end_speech_at),
        "scoring_chain_by_end_1ac_frac": chain / n_episodes,
        "n_episodes": n_episodes,
    }


def run_seed(seed, warmstart_path, spec, cfg):
    ac, _ = load_checkpoint(warmstart_path)
    ac.curriculum = CURRICULUM                          # Screen B: opening unlock ladder ON
    pool = CheckpointPool(directory=os.path.join(OUTPUT_DIR, f"pool_seed{seed}"), semantics=cfg.semantics)
    updater = PPOUpdater(ac, cfg)
    rng = random.Random(seed); gen = torch.Generator().manual_seed(seed)
    discount = float(cfg.semantics.require("discount"))
    pbrs_lambda = float(cfg.semantics.require("pbrs_lambda"))
    gl = cfg.tuning.gae_lambda
    rows = []
    export_counts = {"affwin": 0, "negwin": 0, "nearmiss": 0}
    death_cause = Counter(); death_neg = Counter(); death_speech = Counter(); death_n = 0
    for u in range(BOOTSTRAP_UPDATES):
        ecoef = entropy_coef(u, ENTROPY_TOTAL, cfg.semantics)
        counts = {"self": 0, "pool": 0}

        def sampler(r, _ac=ac, _pool=pool, _c=counts):
            opp = _pool.sample_opponent(_ac, r); _c["self" if opp is _ac else "pool"] += 1
            return opp

        trajs = []; rounds_found = rounds_ext = 0; phis = []
        for ep_idx in range(E):
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
            if u >= BOOTSTRAP_UPDATES - 15:
                dead = _analyze_dead_chains(rnd, trace)          # WHY each found chain died
                for d in dead:
                    death_n += 1
                    death_cause[d["cause"]] += 1
                    death_neg["neg_contested" if d["neg_contested"] else "uncontested"] += 1
                    if d["death_speech"]:
                        death_speech[d["death_speech"]] += 1
                _export_round(rnd, trace, traj.winner, f > 0, e > 0,
                              seed, u, ep_idx, export_counts, dead)
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
    probe = None
    if CURRICULUM:
        probe = _probe_1ac(ac, PROBE_EPISODES, gen)
        print(f"    seed{seed} 1AC-probe: chain-by-end={probe['scoring_chain_by_end_1ac_frac']:.2f} "
              f"move2={probe['move2_roles']} move3={probe['move3_roles']}", flush=True)
    death_stats = {"n_dead_aff_chains": death_n, "by_cause": dict(death_cause),
                   "neg_contested_split": dict(death_neg),
                   "by_death_speech": dict(death_speech)}
    print(f"    seed{seed} dead-AFF-chains(last15)={death_n} by_cause={dict(death_cause)} "
          f"neg_split={dict(death_neg)}", flush=True)
    return {"seed": seed, "last15_win": win, "last15_surv": surv, "passed": passed,
            "phi_first15": phi_first, "phi_last15": phi_last, "phi_trend": phi_trend,
            "divergence": divergence, "probe": probe, "death_stats": death_stats, "rows": rows}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cfg = TrainingConfig(
        tuning=TuningConfig(minibatch_size=256, epochs_per_batch=3, warmstart_epochs=20,
                            warmstart_minibatch_size=128, episodes_per_update=E,
                            snapshot_interval=SNAPSHOT_INTERVAL, total_updates=120, seed=0),
        semantics=SemanticsConfig(), output_dir=OUTPUT_DIR)
    spec = EncoderSpec()
    _override = os.environ.get("PBRS_WARMSTART")             # explicit warm-start path (A4 cell)
    if _override:
        print(f"[warmstart] using PBRS_WARMSTART override {_override}", flush=True)
        warmstart_path = _override
    elif os.path.exists(SHARED_WARMSTART):
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
