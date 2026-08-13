# CDAF — transfer / handoff doc

Living handoff for the CDAF project (competitive policy-debate rounds as typed
argument graphs → deterministic judge → PPO self-play). This file is the fast path
back into context; the specs in `docs/` are ground truth over anything here.

**Last refreshed:** 2026-08-13 (post-iCloud-eviction recovery; see Incident log).

---

## 0. Where the repo lives now (READ FIRST)

- **Repo path: `/Users/sinan/CDAF`.** It was previously on `~/Desktop`, inside
  iCloud "Desktop & Documents" sync, and got partially evicted to dataless stubs
  (see Incident log §7). It has been fully materialized and **moved out of the sync
  scope**. Do not move it back under `~/Desktop` or `~/Documents`.
- Branch: `add-cdaf-judge`. Remote: `git@github.com:SinanRoumie/CDAF.git`.
- venv: `/Users/sinan/CDAF/.venv` (Python 3.11, numpy 2.4.6, torch 2.13.0). The
  system `python`/anaconda is 3.9 and lacks the deps — **always use
  `.venv/bin/python`** (the `source .venv/bin/activate` shim did not take in one
  session; call the interpreter directly).
- Tests need `PYTHONPATH=tests` for a couple of modules (`fuzz_env_shell`).

## 1. What this project is

- `model/` — typed argument graph (nodes with roles: advocacy / framework / link /
  impact / uniqueness / ballot-directive; typed edges: support / defensive_attack /
  offensive_attack / comparison), serializer, v1→v2 converter, `SPEECH_ORDER`.
- `judge/` — deterministic verdict + decision trace + templated RFD. `JUDGE_VERSION
  = 6`. DF-QuAD accrual, framework subsystem, recursive weighing clash-resolution.
- `env/` — RL environment shell: legal-action generation, per-speech move budgets,
  extension as per-node liveness.
- `policy/` — graph encoder (permutation-invariant, content-blind), actor-critic
  heads, legal-action masks.
- `training/` — PPO, GAE, PBRS reward shaping, behavior-cloning warm-start,
  checkpoint pool for self-play, rollout collection.
- `warmstart/` — fixture → env-trajectory converter for BC.
- `cdaf_app.py` — Dash graph builder (Yaz reviews exported rounds here).
- `scripts/pbrs_bootstrap_screen.py` — the tracked, cloud-transferable screen
  driver (see §4).

## 2. How to verify the tree is healthy

```
find . -type f -flags +dataless | wc -l        # must be 0 (no iCloud stubs)
.venv/bin/python -m pytest -q                    # expect: 435 passed
```
Plus the two semantic gates:
- **Oracle unmoved:** `tests/oracle/test_oracle.py` green — no `(winner,
  reason_class)` moved.
- **Warm-start converts 42/42:** `tests/test_warmstart_convert.py` asserts
  `len(_CONVERTIBLE) == 42` and `passed == 42` (47 fixtures − 4 masked-construct −
  1 unrootable(E)).

**Current status (2026-08-13): all green, 435 passed.** Note the count moved from
433 → 435 with the Ruling 1&2 commit + the fixture repairs below; any on-pod
verify gate that still hardcodes `433 passed` must be updated to `435`.

## 3. The three rulings (this milestone — COMMITTED)

All three are landed on `add-cdaf-judge` (`5ff9396`, `b6d8e7a`, `9c23815`).

### Ruling 1 — weighing operands must be SAME-KIND
Cross-kind weighs are rejected at creation by a predicate in the Weigh branch of
`_structural_legal` (`env/legal_actions.py`). Consumers:
- **BallotDirective vs BallotDirective:** DESCRIPTIVE / INERT — legal, no consumer,
  deliberately so.
- **Uniqueness vs Uniqueness:** consumed only when COMPETING — EITHER (A) an explicit
  `DefensiveAttackEdge` between the two weighed uniquenesses (the r15/r16 structure),
  OR (B) the `nonunique_on_link` route (a non-unique Uniqueness attacking the Link the
  other weighed Uniqueness supports). Two uniquenesses with neither relationship are
  inert.
- All 18 fixture weighs are same-kind → no verdict changed (verified).

### Ruling 2 — judge-invisible edges
- **V1a** (cross-side defensive attack on a BallotDirective) → **LEGALITY MASK**.
- **V1b, V2, V3** → documented only. They are **SILENTLY UNREAD** — no trace, no
  collapse_reason, no signal. (Do NOT describe them as emitting a `collapse_reason`;
  they don't. Follow-up filed: "emit a named `judge_invisible_edge` reason" so a
  future session doesn't hunt for a collapse path that does not exist.)
- **TRAP (do not regress):** do NOT mask support-on-BD generally. Same-side Link→BD
  and Framework→BD anchor, and cross-side Impact→BD anchors via capture. A blanket
  mask breaks r6 and the T1/T1b capture fixtures.

### Ruling 3 — extension incentive — CLOSED, no code change
Both halves already hold: new offense first introduced in a rebuttal (1AR/2NR/2AR)
is penalised at ballot and in Φ; a constructive chain must be carried through every
own-side speech to score. PBRS invariance (Φ(terminal)=0) forbids adding NET
incentive via Φ. Elevated `extension_fail` is therefore a **learning problem**
(addressed by longer training / more budget), not a rules change.

## 4. The screen driver (`scripts/pbrs_bootstrap_screen.py`)

Runs the PBRS bootstrap-robustness screen: per seed, a bootstrap-only PPO run at
`pbrs_lambda=0.5` and episodes-per-update E, measuring last-15-update mean win-rate
+ extension-survival against the PASS bar (**win ≥ 0.10 AND survival ≥ 0.20**). No
decay — PBRS is never withdrawn. Transfers to the pod **via git clone**; writes all
outputs into gitignored `runs/`.

**Env knobs:**

| Var | Default | Meaning |
|---|---|---|
| `PBRS_E` | 20 | episodes per PPO update (the lever) |
| `PBRS_UPDATES` | 40 | bootstrap updates |
| `PBRS_SEEDS` | `0,1,2,3,4` | comma-sep seed list |
| `PBRS_OUTDIR` | `runs/pbrs_screen` | output dir (gitignored) |
| `PBRS_CURRICULUM` | `0` | `1` = opening unlock-ladder ON (Screen B) |
| `PBRS_PROBE_EPISODES` | 200 | Screen-B 1AC construction probe count |
| `PBRS_EXPORT_DIR` | unset | if set, export last-15-update rounds (builder JSON + sidecar) |
| `PBRS_SCREEN_TAG` | `A` | filename label |
| `PBRS_EXPORT_CAP` | 4 | rounds per bucket **per seed** |
| `PBRS_WARMSTART` | unset | explicit warm-start checkpoint override |

**Budget config hook (NEW, in `env/actions.py`, read ONCE at import):**

| Var | Default | Meaning |
|---|---|---|
| `CDAF_SPEECH_BUDGET` | (built-in, sum 52) | JSON object, slot→positive int; PARTIAL merges over defaults; keys ⊆ SPEECH_ORDER; loud on bad input |
| `CDAF_EXTEND_COST_K` | 4 | extend/concede batch size K (positive int) |

Because it reads at import, a worker must set the env var in its own process
*before* importing `env` — i.e. set it on the pod's job invocation (as we do for the
sweep), not mid-run.

**Round export buckets** (`_export_round`) — three disjoint buckets, `EXPORT_CAP`
each per seed:
- `affwin` — AFF won.
- `nearmiss` — AFF **found** offense but did NOT win.
- `negwin` — NEG won, no AFF offense found.

Each round writes `<tag>_<bucket>_seed<s>_u<uu>_ep<eee>.json` (oracle format,
builder-loadable) + a `.sidecar.json` with: seed, update, episode, verdict
{winner, reason_class}, `offense_found`, `offense_survived`, per-AFF-chain
{extended, in_scope, sign, mag, delta, collapse_reason}, and `dead_chain_analysis`
(collapse_reason, cause, death_speech, `neg_contested`, responsible/side). **This
sidecar already carries everything Phase 1 asks for.**

Note a wording nuance for Yaz: the `nearmiss` bucket is "found but didn't **win**";
the Phase-1 brief says "found but didn't **survive**." They usually coincide; the
sidecar's `offense_found`/`offense_survived` flags let you filter precisely.

## 5. Current RL state / the open problem

- **PBRS** = potential-based reward shaping: per-step `F_t = λ(γΦ(s') − Φ(s))`,
  λ=0.5, γ=0.999. Φ = mid-round maxdiff of best extended AFF vs NEG chain ∈ [−1,1].
  Policy-invariant (Φ(terminal)=0); never withdrawn.
- **Headline problem — extension collapse.** Random play builds offense constantly
  (~75% of rounds "offense found") but **extension survival collapses to ~1.6%**.
  Every AFF loss traces to a correct judge gate. The behavior that must emerge first
  is *carry one spine across all own-side speeches*. **The number to watch is
  extension-survival rate; if it hasn't moved off ~1.6% within a few hundred
  updates, more steps won't fix it.** Dominant chain-death cause is `extension_fail`.
- **Latest measurements (κ=0.0):** E-sweep `runs/pbrs_screen_rescreen_cloud`
  gave 3/5 seeds PASS at both E=100 and E=150; `runs/screenA_20260812` gave 2/5.
- **κ (PHI_NASCENT_KAPPA) is a dead end, now 0.0.** It was an extra mid-round Φ
  channel for near-complete (passed-every-gate-but-extension) chains,
  `Φ = (aff_ext − neg_ext) + κ(aff_nascent − neg_nascent)`. Trials at κ=0.3 measured
  3/5 → 2/5 → 1/5 as the gate was refined — structurally incapable of adding net
  signal under PBRS invariance. **Reverted to 0.0** (`judge/config.py:
  PHI_NASCENT_KAPPA = 0.0`, commit `f83c645`). The dormant `kappa` branch remains in
  `phi_maxdiff` (optional cleanup). **See the κ incident in §7 — this is the doc bug
  that cost an experiment.**

## 6. Phase 1 — budget sweep (in flight; see `docs/`/launch kit)

Question: **is per-speech move budget the binding constraint on extension
survival?** 5 configs × 5 seeds, one pod per config, all parallel. Standard
protocol: E=100, 40 updates, PBRS λ=0.5, 1 torch thread/worker, clean git-clone
tree verified on-pod. Configs (exact `CDAF_SPEECH_BUDGET` dicts in the launch kit):
- **B0** current defaults (sum 52) — no override.
- **B1** 1.25× all (sum 64).
- **B2** 1.5× all (sum 80).
- **B3** 2× all (sum 104).
- **B4** rebuttal-weighted: constructives + 2NC/1NR unchanged, rebuttals (1AR/2NR/
  2AR) → 10 each (sum 67) — isolates whether rebuttal budget specifically binds.

Report per config: per-seed win/survival; chain-death cause distribution (does
`extension_fail` fall as budget rises?); per-speech budget spend broken down by
speech (introduces/connects/extends/weighs) + exhaustion. Round exports (3 buckets,
last 15 updates) + sidecars per config, one folder per config.

**Launch is blocked on two things Yaz owns** (see §8): committing the working tree
(pods clone from git) and RunPod auth.

## 7. Incident log

- **iCloud eviction (2026-08-13).** Repo on `~/Desktop` (iCloud sync) with "Optimize
  Mac Storage" on + near-full disk → macOS evicted 11,591 files to dataless stubs,
  including all of `.git`. Recovered: fully materialized, moved to `/Users/sinan/
  CDAF` (out of sync scope). Post-recovery verify: **0 dataless, 435 passed, oracle
  unmoved, warm-start 42/42, budget hook works.** Stale `.pyc`/`.pytest_cache`
  compiled at the old Desktop path were cleared (they embedded `~/Desktop/CDAF` in
  tracebacks — cosmetic, but cleared).
  - **Fixture fallout (fixed):** the Ruling 1&2 legality masks lowered the
    random-legal fuzzer's reachable node ceiling from 33 → ~31, tripping the
    breadth-guard asserts in `tests/test_encoder.py` and `tests/test_action_heads.py`
    (`max>=30` with a `max>=33` break condition, `seeds=range(200)`). Fix (Yaz-ruled
    "widen seeds, break at 30"): `seeds=range(2000)`, break at `max>=30`. This also
    unmasked a hidden failure — `test_weigh_second_target_excludes_self` hardcoded
    `node_ids[1]` as a legal weigh partner, now cross-kind-illegal under Ruling 1;
    repaired to pick a partner the mask actually marks legal. These test edits are
    **uncommitted** as of this writing.
- **κ=0.3 phantom-revert (prior, the doc bug this file exists to prevent).** A
  handoff recorded the nascent-Φ channel as reverted ("byte-identical to
  pre-attempt") while the working tree **still carried κ=0.3 uncommitted**. That
  κ=0.3 shipped to the pod in a Screen A re-run, so that run's 0/5 is **not a valid
  κ=0 measurement**. Lesson baked into §5: κ is genuinely 0.0 now (committed
  `f83c645`); verify `PHI_NASCENT_KAPPA` before trusting any "κ reverted" claim.

## 8. Open dependencies / next actions

- **Yaz — git:** commit + push the current working tree so pods clone it. Uncommitted:
  `env/actions.py` (budget hook), `training/rollout.py` + `scripts/pbrs_bootstrap_
  screen.py` (spend instrumentation), `tests/test_encoder.py` +
  `tests/test_action_heads.py` (fixture repairs). After this, the on-pod verify gate
  must expect **435 passed** (not 433).
- **Yaz — RunPod:** this machine's `runpod` CLI has no API key
  (`AuthenticationError: No API key provided`). Provide auth (`runpod config` or
  `RUNPOD_API_KEY`) or launch the pods. Teardown is Yaz's.
- **Follow-up (filed):** emit a named `judge_invisible_edge` collapse reason for
  V1b/V2/V3 (currently silently unread).
- **Deferred (pre-existing):** turn-offense milestone; V2 restorative-support `{s_k}`
  slot; builder Comparison-direction enforcement. See `PROGRESS.md`.
</content>
</invoke>
