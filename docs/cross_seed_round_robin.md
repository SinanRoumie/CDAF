# Cross-Seed Round Robin — W3_B4 (u0499), 5 seeds

**Evaluation only, no training.** Goal: quantify how far the trained self-play policies are
from convergence. This is **not** a test for Nash equilibrium — failing to find an exploit
would prove nothing. The claim rests only on *positive* findings: systematic cross-seed
dominance and qualitative regime divergence.

## Method

- **Policies:** each seed's final checkpoint `runs_phase2/W3_B4/seed{0..4}/ckpt_u0499.pt`
  (500 PPO updates, B4 rebuttal-weighted budget, shared BC warm-start). Five seeds.
- **Games:** every seed as AFF vs every seed as NEG, both orientations (25 ordered cells,
  diagonal = self-play). Played through the exact masked-legal self-play path training uses
  (`training.rollout.collect_episode`) and scored by the deterministic judge. Sampling is the
  policy's own categorical (stochastic, not argmax), so repeated rounds give a genuine
  win-rate distribution.
- **n = 1000 rounds/cell** (25,000 rounds total). Worst-case (p=0.5) 95% CI half-width
  = 1.96·0.5/√1000 = ±0.031; tighter near the low rates observed. Cell point-values therefore
  carry ~±0.015–0.03 Bernoulli noise (see re-verification) — the conclusions rest on
  multi-σ gaps far larger than that, never on precise cell values.
- **Infra:** one RunPod CPU pod per seed computed that seed's AFF row.
- Artifacts: `runs_rr/rr_aff{0..4}.json` (raw per-cell counts + reason_class),
  `runs_rr/rr_analysis.json`, `runs_rr/degeneracy.json`, `runs_rr/verify.json`,
  `runs_rr/confirm_s4.json`. Scripts: `scripts/cross_seed_rr.py` (one AFF row),
  `scripts/rr_report.py` (matrix/baselines/dominance/cycles), `scripts/rr_degeneracy.py`,
  `scripts/rr_verify.py`.

## [1] Directional win-rate matrix  M[AFF][NEG] = P(AFF-seed wins)

|  AFF \ NEG | s0 | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|
| **s0** | 0.147 | 0.389 | 0.253 | 0.141 | 0.079 |
| **s1** | 0.000 | 0.000 | 0.002 | 0.018 | 0.000 |
| **s2** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| **s3** | 0.003 | 0.002 | 0.054 | 0.036 | 0.002 |
| **s4** | 0.033 | 0.176 | 0.256 | 0.361 | 0.045 |

The matrix is NEG-heavy throughout (these policies win almost entirely as NEG), but AFF win
rate varies systematically by opponent — that variation is the signal.

## [2] Presumption-adjusted baseline — 50% is not the neutral point

NEG wins every tie and every near-zero-net-offense round (`|N| ≤ ε → presumption = NEG`, judge
§1/§7). The equal-skill AFF rate is what a policy scores against **itself** (the diagonal):

| seed | self-play AFF win | 95% CI |
|---|---|---|
| s0 | 0.147 | 0.126–0.170 |
| s1 | 0.000 | 0.000–0.004 |
| s2 | 0.000 | 0.000–0.004 |
| s3 | 0.036 | 0.026–0.049 |
| s4 | 0.045 | 0.034–0.060 |

**p_pres = mean diagonal = 0.046** — the directional neutral point, not 0.5.

**Showing the work** — all 5,000 self-play rounds by judge `reason_class`:

| reason_class | count | frac |
|---|---|---|
| AFF offense (AFF wins) | 228 | 0.046 |
| NEG offense (NEG earns it) | 339 | 0.068 |
| AFF structural failure | 320 | 0.064 |
| framework lock-out | 1081 | 0.216 |
| presumption (tie → NEG) | 3032 | 0.606 |

Of 4,772 NEG self-play wins, **4,433 (92.9%) are AFF-failed-to-establish-offense**
(presumption / framework lock-out / structural failure); only 339 are NEG carrying its own
offense. NEG's raw dominance is presumption on undecided rounds, not superior play — hence the
neutral point is 0.046, not 0.5.

### ⚠️ Caveat: p_pres is a poor single summary; use per-seed baselines

The diagonal spans **0.000 → 0.147** — the seeds do not even agree on their own self-play
baseline. So the single mean 0.046 must **not** be used to read the directional matrix
globally. A directional cell M[i][j] is interpretable only against **seed i's own diagonal**
M[i][i] (its per-seed baseline), because "how often does this seed establish AFF offense at
all" differs by an order of magnitude across seeds.

**The dominance matrix [3] does not depend on p_pres at all** — side-averaging over both role
assignments cancels the presumption term exactly, so its baseline is 0.5 for every pair
regardless of any seed's p_pres. **The dominance matrix is therefore the object that carries
the convergence claim; the directional matrix is descriptive and per-seed-relative.**

## [3] Dominance matrix — the presumption-free comparison (carries the claim)

> **S(A,B) = ½·M[A][B] + ½·(1 − M[B][A])**

Under equal skill S(A,B) = ½p + ½(1−p) = **0.500 for any p_pres**: presumption helps whoever is
NEG equally often across the two orientations, so it cancels. S is a proper antisymmetric
tournament (S(A,B)+S(B,A)=1); baseline **exactly 0.5**.

| A \ B | s0 | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|
| **s0** | .500 | .695\* | .627\* | .569\* | .523\* |
| **s1** | .305\* | .500 | .501 | .508\* | .412\* |
| **s2** | .373\* | .499 | .500 | .473\* | .372\* |
| **s3** | .431\* | .492\* | .527\* | .500 | .321\* |
| **s4** | .477\* | .588\* | .628\* | .679\* | .500 |

`*` = |z| ≥ 2 vs 0.5. **18 of 20 ordered pairs are significant** (only s1↔s2 ties); mean
|S−0.5| = 0.084 (equal-equilibrium expectation ≈ 0). These are nowhere near baseline.

**Copeland ranking** (# seeds significantly beaten):
**s0 (4) > s4 (3) > s1 (1) > s3 (1) > s2 (0)** — a strict order (s1/s3 tie on Copeland; s1
beats s3 head-to-head, S=0.508). seed0 beats all four; seed2 loses to all four. Starkest:
s0-AFF beats s1-NEG 0.389 while s1-AFF beats s0-NEG 0.000 → S=0.695, **z=+25**.

## [4] Transitivity

**No 3-cycle** among the significant dominances — the tournament is transitive at |z| ≥ 2, so
the order in [3] holds with no A>B>C>A. There is therefore **no intransitivity disproof** of a
single equilibrium here (reported as asked; a cycle would have been a hard disproof, but its
absence proves nothing on its own).

## [5] Degeneracy — two seeds collapsed, in different regimes

The raw win/loss numbers could read as "s1/s2 play AFF badly." They do not: s1 wins 0.000 as
AFF in all five cells (self-play included) and s2 wins 0.000 in all five. Per-seed self-play
`reason_class` and a structural probe (`scripts/rr_degeneracy.py`, 200 self-play rounds/seed,
capturing the terminal graph) show this is **collapse, not weak AFF play**:

| seed | AFFoff | NEGoff | mean nodes | link | impact | framework | weighing | BD | regime |
|---|---|---|---|---|---|---|---|---|---|
| s0 | 0.135 | 0.110 | 32.7 | 7.8 | 7.3 | 1.6 | 7.4 | 2.8 | **functional (balanced)** |
| s3 | 0.030 | 0.100 | 30.0 | 8.7 | 8.6 | 5.5 | 1.8 | 2.4 | functional (weak) |
| s4 | 0.035 | 0.165 | 27.6 | 4.4 | 5.5 | 4.6 | 1.1 | 9.3 | functional (NEG-heavy) |
| s1 | **0.000** | **0.000** | 43.3 | **2.3** | 11.0 | 0.8 | 7.2 | **14.4** | **collapsed** |
| s2 | **0.000** | 0.005 | 35.3 | 4.5 | 5.5 | **4.9** | **9.8** | 3.4 | **collapsed** |

(AFFoff/NEGoff = fraction of self-play rounds resolving to `AFF offense` / `NEG offense`;
columns = mean nodes of that kind per round.)

- **Every seed builds an AFF advocacy in 100% of rounds** — so the collapse is **not**
  advocacy-absence (the specific structural hypothesis is rejected). s1 and s2 build large
  graphs; the offense simply never resolves.
- **s1 is collapsed by starving links.** It spams ballot-directives (14.4/round) and impacts
  (11.0) but builds almost **no link nodes (2.3/round)**, so its chains fail the link-premise
  requirement (judge §2/§3.3.1, `no_link_premise`): a path of impacts with no literal `Link` is
  not a live offense carrier. Result: 0 offense on **either** side; 97% pure presumption.
- **s2 is collapsed differently** — framework (4.9) and weighing (9.8) spam → 27% of self-play
  ends in framework lock-out, but resolved offense is ~0 (0.5% NEG, 0% AFF).

So s1 and s2 are collapsed into **two distinct non-functional regimes**, neither of which
mounts a resolvable case.

**Consequence for the ranking.** Because these NEG-dominated policies win almost entirely by
denial, the Copeland order is largely an ordering on *NEG stonewall quality near the degenerate
floor*, not on strategic sophistication. This is why a **collapsed** seed (s1) outranks a
**functional-but-weak** one (s3): s1-as-NEG concedes 0.2% while s3-as-NEG concedes more, and
s3's tiny AFF offense doesn't recover the side-average. The order is real and statistically
robust, but it should be read structurally, not as "five strategies ranked by skill."

## Bottom line — distance from convergence

If the five seeds sat at a common equilibrium they would play near S = 0.5 against each other.
Instead: 18/20 matchups show significant systematic dominance (up to 25σ off neutral), in a
strict order s0 > s4 > s1 > s3 > s2, and — structurally — the seeds occupy **qualitatively
different regimes**: one balanced-functional (s0), two other functional variants (s3 weak, s4
NEG-heavy), and **two collapsed policies (s1, s2) that never establish resolved offense, each
degenerate in a different way**. No cycle was found, so the evidence here is the systematic
dominance gaps and the regime divergence, not intransitivity.

**Framing correction (vs an earlier draft):** this is *not* "five strictly-rankable
strategies." It is seeds diverging into qualitatively distinct regimes, including degenerate
ones — which is still strong (arguably stronger) evidence against convergence, but a different
sentence.

## [6] Re-verification of the churned rows (process-incident control)

During collection, a process-monitoring mistake (a `pgrep -f` pattern matched the ssh
check-shell itself, and a follow-up `pkill` self-killed and churned some runs) forced seeds
1/2/4's rows to be re-run. Each launched process runs a **complete, independent** n=1000 row
and writes its JSON only at the end, so the banked values are the intended single clean runs;
all five JSONs were validated (parse, 5 cells, n=1000, counts in range). As a positive control,
one representative cell from each churned row was re-run with a **disjoint RNG stream**:

| cell | banked | banked 95% CI | re-run (fresh RNG) | verdict |
|---|---|---|---|---|
| aff1 vs neg3 | 0.018 | 0.011–0.028 | 0.019 | inside |
| aff2 vs neg0 | 0.000 | 0.000–0.004 | 0.000 | inside |
| aff4 vs neg3 | 0.361 | 0.332–0.391 | 0.315 | 2.2σ low |

The aff4/neg3 re-run landed just outside the banked CI. Resolved with an **n=4000** confirmatory
draw (3rd RNG): **0.331** (CI 0.317–0.346); pooled over all three draws (6000 rounds) = 0.334.
So the true value is ≈0.33: the banked 0.361 was a high n=1000 draw and the re-run 0.315 a low
one, both within Bernoulli noise of the truth. **This is sampling variance, not churn
corruption** (corrupted data would not yield a coherent ~0.33 rate reproduced by an independent
4000-round sample). Impact on conclusions: **none** — S(4,3) shifts 0.679 → 0.665, still ≈21σ.
The banked matrix is retained as the uniform-n=1000 artifact; individual cell values carry the
~±0.02–0.03 noise noted in Method, which the dominance margins dwarf.

## Reproduce

```
# one AFF row (per pod / locally):
python -m scripts.cross_seed_rr --aff-seed <i> --neg-seeds 0,1,2,3,4 --n 1000 --out runs_rr/rr_aff<i>.json
# matrix + baselines + dominance + cycles:
python -m scripts.rr_report --dir runs_rr --out runs_rr/rr_analysis.json
# degeneracy probe:
python -m scripts.rr_degeneracy --k 200 --out runs_rr/degeneracy.json
# churn control:
python -m scripts.rr_verify
```
