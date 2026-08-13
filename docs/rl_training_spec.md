# CDAF RL Training Spec (Phase 5)

## Scope

This spec defines the self-play training loop: opponent selection, side
assignment, reward shaping and its annealing, batch structure, PPO
configuration, evaluation protocol, and logging. It does not define the
encoder or action heads (`encoder_spec.md`), the environment contract
(`environment_shell_spec.md`), or the action vocabulary
(`action_schema_spec.md`).

The judge is the reward function and is never modified by anything here.

## Governing constraint: the reward is the judge

The policy is trained against verdicts, not against opinions about good
debate. The one exception is the chain-extension shaping bonus below, which
exists solely to make the bootstrap problem tractable and is annealed to zero
before the run ends. Every other strategic distinction — how many spines to
carry, when to abandon one, how to spend rebuttal budget, whether to turn or
take out — must emerge from self-play.

## The bootstrap problem

Measured on 500 uniform-random rounds (seed 0):

| Quantity | Rate |
|---|---|
| AFF advocacy node present | 83.6% |
| AFF offense chain found by judge | 75.0% |
| ...that survives extension | 1.6% |
| AFF gates passed (any of four) | 0.0% |
| Verdicts | 100% NEG |

Random play builds offense constantly and carries it almost never. Every AFF
loss traces to a correct gate — extension failure dominates, with
final-speech spikes (unresolved sign) and defensive kills accounting for the
rest. This is the judge behaving as specified, not a bug.

The consequence for training: the terminal reward is constant-zero under
random play, so there is no gradient to climb. Disabling presumption does not
fix this — `aff_sum` is 0 in essentially every round, so removing NEG's
default win just relocates the tie.

The behavior that must appear first is narrow and identifiable: **carry one
spine across all of your side's speeches.** That is the 75% → 1.6% collapse,
and it is the single number to watch early in training.

## Reward

**Terminal ballot reward**, binary, from AFF's perspective, with the per-side
split in `info["rewards"]`.

**Potential-based shaping (PBRS) — the shaping term** (replaces the retired flat
chain-extension bonus). Shaping is a per-step potential difference, not a terminal bonus:

    F_t = λ · ( γ · Φ_L(s') − Φ_L(s) ),   λ = 0.5 (ruled; semantic hyperparameter)

on each learner decision step, side-relative potential Φ_AFF = Φ, Φ_NEG = −Φ, with γ equal
to the training discount (0.999). The environment exposes Φ(s); the training loop forms F_t
(environment_shell_spec §step()).

- **Φ = Φ_maxdiff.** Φ(s) = (best AFF-favoring chain strength) − (best NEG-favoring chain
  strength) over the mid-round chain resolver. Per chain, signed strength is +mag if it
  favors AFF, −mag if it favors NEG, 0 if the sign is unresolved; a chain counts only if it
  is extended-so-far and in-scope, and **chains outweighed by a determinate impact-weigh made
  so far are excluded** (weighing-awareness). Each side's max ∈ [0,1] (mag is a σ-product), so
  Φ_maxdiff ∈ [−1,1]; Φ(empty) = Φ(terminal) = 0. Φ_maxdiff rewards carrying/strengthening the
  single best offense AND successfully out-weighing the opponent's best impact, while being
  count-resistant (extra weak chains don't raise a max) — unlike a sum, which would re-create
  the "extend everything" incentive.
- **Nascent channel (mid-round Φ only) — NEUTRALIZED to κ = 0 (2026-08-12); DEAD END.**
  *Mechanism (retained for the record):* the extended-so-far gate makes Φ = 0 for a graph with
  *attempted-but-incomplete* structure (an in-scope, resolved-sign, positive-mag chain whose
  own-side extension is not complete) — indistinguishable from the empty graph. The nascent
  channel added a discounted second channel over chains passing every gate *except* extension
  and still within their own-side extension window (not lapsed):
  Φ = (best_aff_ext − best_neg_ext) + κ·(best_aff_nascent − best_neg_nascent).
  *Outcome:* it was tried twice (plain, then with the lapsed-exclusion "still within window"
  gate) and measured **degrading the bootstrap screen every time — κ = 0 → 3/5 seeds, κ = 0.3
  → 2/5, refined gate → 1/5** — and shown structurally incapable of injecting net signal
  because PBRS invariance means shaping cannot change the optimum. **Concluded a dead end;
  ruled back to κ = 0.**
  **Incident note (2026-08-12):** the revert to κ = 0 was recorded in a handoff as complete
  ("byte-identical to pre-attempt") but was **never applied to the working tree** — κ = 0.3
  persisted uncommitted (present in no commit on any branch in history) and **shipped to the
  pod in the Screen A re-run**, so **that run's 0/5 result is not a valid measurement** (it was
  taken in the known-degrading κ = 0.3 regime, not the κ = 0 regime of the 3/5 baseline).
  `PHI_NASCENT_KAPPA` is now 0.0; the `phi_maxdiff` `kappa` branch defaults to 0.0 and is left
  dormant (full removal is optional cleanup). At κ = 0, Φ is byte-identical to the extended-only
  potential and all four PBRS correctness constraints hold trivially.
- **Extension incentive (Ruling 3) — CLOSED, no code change (2026-08-13).** The incentive to
  extend the best offense rather than drop and rebuild it fresh is ALREADY implemented on both
  halves: (1) new offense first introduced in a **rebuttal** (1AR/2NR/2AR) is penalised at the
  ballot AND in Φ, above the horizon check, with Advocacy roots exempt; (2) a constructive chain
  must be carried through **every own-side speech** (from its introduction) to score — at the
  ballot and in the extended-so-far Φ gate. PBRS invariance (Φ(terminal) = 0) **forbids** any Φ
  change from adding NET incentive — it could only reshape signal timing — and the maturity-
  weighting that would attempt it is exactly the neutralised **nascent channel** (dead end,
  above). The elevated `extension_fail` rate is therefore a **learning problem** (the policy has
  not yet learned an incentive that already exists), addressed by longer training, **not** a rules
  change. Do not reopen as a Φ or legality change.
- **Policy-invariant.** Under the four correctness constraints (environment_shell_spec
  §step()), PBRS provably does not change the optimal policy — it only speeds learning (dense,
  low-variance critic target + shaped advantage structure). This is the key difference from
  the retired flat bonus, which was non-potential, distorted the optimum, and produced
  bonus-dependence (win-rate tracked the coefficient and collapsed on withdrawal, Real Run 1).
- **λ = 0.5, fixed, no schedule.** Φ as a half-scale value prior: dense enough to break the
  sparse-reward bootstrap barrier, small enough that Φ's imperfections are a modest prior for
  the critic to shed. No warm-up (the dense signal is most needed at init), no decay
  (invariance ⇒ nothing to withdraw). A late taper is only ever an empirical add-on if a run
  shows late-stage stalling attributable to Φ over-tracking — not a default.

**Inert-action penalty — dormant (coefficient `inert_penalty_coef`, ruled 0.0).**
There is currently **no reward-side inert penalty**. The one context-dependent
inert move — a **no-op re-extend** (extend/concede on a node already carried this
speech) — is priced **structurally, via cost**: it consumes a full budget slot
(environment_shell_spec §Liveness stamping), so a wasted carriage costs a real move
rather than being near-free under the extend-batch discount. The other three inert
classes (same-side attack, offense at a non-polarity node, redundant connect) are
**structurally illegal** — never sampled — so they need no reward term at all
(environment_shell_spec §Governing principle).

The `inert_penalty_coef` hook is **kept, ruled to 0.0** (byte-identical to off), as
a documented backstop: if a future run shows residual no-op spam the cost model does
not reach — specifically **late-speech carriages in a speech with slack budget**,
where a wasted slot has little opportunity cost — a small nonzero coefficient can be
enabled as a second lever on top of the full-slot cost. Enabling it is a semantic
ruling; it is dormant until such evidence appears.

### Annealing — retired for PBRS

The conditional-trigger annealing machinery (trigger on AFF ballot win rate; decay the
coefficient to zero over a window) applied to the **flat** chain-extension bonus, which had
to be withdrawn because it distorted the optimum. **PBRS is never withdrawn** — it is
policy-invariant, so the final policy trains with it on. The trigger/window/decay
hyperparameters and the `ShapingAnnealController` are retired for this term.

### Validation (PBRS)

There is nothing to withdraw, so the old "does win rate hold when shaping is decayed?"
diagnostic (Real Run 1) does not apply. Validation rests on:

- **Bootstrap-robustness across seeds — the fragile axis.** Does PBRS at λ=0.5 lift the
  policy off the constant-zero floor (extension-survival + nonzero *ballot* win rate) from
  warm-start, *reliably* across several seeds? Analog of the earlier magnitude/seed screens;
  the fragile step is still bootstrap, and λ (invariance-preserving) is the knob if it
  under-bootstraps.
- **Eval / ballot win rate is the always-uncontaminated verdict.** PBRS adds nothing to the
  ballot term, and evaluation is inherently shaping-free (the policy acts with no reward at
  eval). So the training AFF **ballot** win rate and eval win rate are the direct measures —
  no decay phase needed to expose the truth (the gap Real Run 1's decay was built to reveal
  is closed by construction).
- **Divergence signal — reinterpreted.** Φ (or extension-survival) climbing while ballot
  win rate stays flat is still logged, but now reads as **"train longer / adjust λ," not
  "abort, dependent."** Under invariance the optimum is unchanged, so such a divergence is a
  finite-time transient (over-tracking Φ's imperfections), expected to self-correct — the
  opposite interpretation from the flat-bonus regime, where the same signal meant permanent
  bonus-farming.

## Self-play

### Opponent selection: fixed-interval checkpoint pool

Snapshot the policy every N updates into a pool; sample opponents from it.

Pure self-play is rejected because debate has rock-paper-scissors structure —
turn-spam beats defense-heavy beats turn-spam — so a policy training only
against its current self can cycle indefinitely while local win rates sit at
50%, which is indistinguishable from equilibrium. The pool forces the policy
to stay competent against everything it previously beat.

Prioritized sampling (weighting opponents the policy loses to) is a
refinement, not a starting point: it adds per-opponent win-rate tracking and
hyperparameters, and can overfit to a single hard opponent.

- **Pool size:** bounded — most recent N checkpoints plus a few early
  anchors. Unbounded pools dilute toward weak early checkpoints as they grow.
- **Sampling ratio:** roughly half current-self, half pool. Enough self-play
  for the frontier to advance, enough pool for stability.

### Side assignment

**Training: random per episode.** Gradients average over the batch, so
variance reduction buys nothing here.

**Evaluation: mirrored pairs.** Each matchup is played both ways so side
advantage cancels exactly. Presumption is a large enough structural thumb on
the scale that unmirrored evaluation can report a checkpoint as stronger when
it merely drew NEG more often — and evaluation numbers drive decisions.

## Batch structure

Rollouts are cheap: ~4ms per round, no model in the loop, pure graph
manipulation. Batch size is therefore nearly free, and large batches are
preferred because a sparse binary reward produces high-variance gradients.

- Start with thousands of episodes per PPO update; reduce only if the update
  step becomes the bottleneck rather than rollout generation.
- Parallel rollout workers each hold their own environment instance. There is
  no shared state, so this is trivially correct.

## PPO configuration

Standard defaults for clip range, GAE lambda, learning rate, and epochs per
batch. Two parameters need deliberate attention:

**Entropy coefficient.** The action space is factored and large. Too little
entropy early and the policy collapses onto a single action type before
exploring — most likely `extend`, since the shaping bonus rewards carrying
chains. Start above default and decay.

**Value loss.** With a sparse binary reward the critic's target is nearly
constant early in training, so the critic may learn nothing before the actor
does — and if the critic is uninformative, PPO's advantage estimates are
noise. Watch value loss explicitly; an actor improving while the critic is
flat is a warning sign, not progress.

## Evaluation

Separate from training. Self-play win rate is uninformative — it sits at 50%
by construction.

**Ladder against all pool checkpoints**, run every N updates rather than
continuously. This is the only protocol that detects cycling directly: if
checkpoint 10 loses to checkpoint 3, the policy is cycling regardless of what
recent win rates suggest. Affordable given rollout speed.

**Random agent as a sanity floor.** Absolute and never moves, but saturates
quickly — useful only to confirm nothing has catastrophically regressed.

The hand-authored fixed 1AC is the only reference with real debate structure
in it, but it is an opening rather than a full opponent, so it cannot serve as
a complete evaluation baseline.

### Validation metric: construction and compression

Held as a **falsifiable prediction about emergent behavior**, deliberately
not encoded anywhere in the reward or environment.

Real rounds show broad offense in constructives narrowing to a specific
ballot path in the back half. If a trained policy develops this spontaneously,
that is evidence the action space and judge generate genuine strategic
pressure. If it never emerges, the environment is failing to reproduce an
incentive gradient real debate has — a diagnostic worth acting on, not a
reward term to add.

## Logging

Per update, all of:

- **Gate pass rates, individually** — advocacy, complete_chain,
  in_scope_impact, N>ε. Aggregated pass rate hides which gate is binding.
- **Extension survival rate** — chains found vs. chains extended. Baseline is
  75% → 1.6%. This is the primary early indicator that the policy is learning
  the thing it must learn first.
- **Action type distribution, broken out by speech.** Also where
  construction-and-compression would first become visible.
- **Chain counts per side.**
- **Verdict and reason_class distribution.** Random-play baseline is 79.4%
  presumption, 19.6% framework lock-out, 0.8% NEG offense, 0.2% AFF
  structural failure.
- **Ballot win rate and shaped return, as separate series** (see annealing).
- **Value loss and entropy.**

## Format: policy debate

The environment trains on the policy speech ordering
(1AC, 1NC, 2AC, 2NC/1NR, 1AR, 2NR, 2AR), not Lincoln-Douglas.

LD was considered specifically as a bootstrap aid — three AFF stamps for
extension instead of four, making a carried spine likelier under random play.
That problem is now addressed twice over, by the chain-extension shaping bonus
and by imitation warm-start, so a format switch would buy nothing already
bought and would cost re-authoring all 47 oracle fixtures under different
extension requirements. That is hand-adjudication work, not relabeling.

There is also a semantics argument. LD's AR raises the reactive-offense
question — 1AR theory, the 1AR restart — which is genuinely contested among
practitioners. Policy's rebuttal rules are cleaner, and the judge already
carries machinery for the analogous final-speech case. A formal system is
better built on the format with less contested semantics.

This is not a lock-in. `CONSTRUCTIVE_SPEECHES` is a model-owned label, the
`REBUTTAL_SPEECHES` derivation transfers to any ordering, and Fence G rejects
stale speech labels loudly rather than mis-scoring silently. Switching later
means authoring three tables (`SPEECH_ORDER`, `CONSTRUCTIVE_SPEECHES`,
`SPEECH_BUDGET`) and re-adjudicating the corpus — expensive, but never silent.

## Imitation warm-start

The policy is initialized by supervised training on demonstration actions
from the hand-authored 1AC and the fixture corpus, then trained with PPO from
those weights.

**Warm-start is not a fixed 1AC and does not foreclose learned case
construction.** It biases where the policy starts; PPO is then free to move
anywhere the reward leads, including away from the demonstrated opening. A
*scripted* 1AC — where the environment plays AFF's first speech and the
policy's first decision is the 2AC — would prevent case construction from
being learnable. That is explicitly not what this is. The policy plays every
speech from step one.

The one real cost is reduced exploration near the initialization, which is
part of why the entropy coefficient starts well above default.

**Corpus — `E` is excluded from the warm-start corpus (43 → 42 convertible).** `E`
is the sole fixture whose graph is a single component with neither an Advocacy nor a
Framework (a NEG-only disad chain), so under the legality-layer floating-root
restriction (action_schema §introduce) its root — a non-Advocacy, non-Framework node
— is unreachable, and it has no legal action sequence to imitate. It is dropped from
the demonstration set. **This does not touch the judge.** `E` remains a valid
**oracle** fixture with its expected verdict unchanged: masks live in the
environment and the policy layer and never enter `judge()`, which scores `E` exactly
as before. The distinction is load-bearing — the *warm-start corpus* (what the
policy imitates) and the *oracle corpus* (what pins judge behavior) are different
sets, and only the former changes.

**Converter root-seeding must seed from Advocacy/Framework (verdict-equivalence
re-verified).** The warm-start converter formerly rooted each component at its
lowest-id node (`min(id)`), which floats non-root nodes for 44 of the corpus's
introduction forests — illegal under the floating-root restriction. It instead
**seeds each component from its Advocacy** (or Framework, per the Advocacy/Framework
root rule) and honors the unlock ladder's type precedence (advocacy → link →
{uniqueness, impact} → {framework, ballot_directive}) when ordering introductions,
tie-broken by ascending `id`, so those artifact floating roots become ordinary
attaching introduces (the sink nodes among them build as flipped edge-sources, which
the judge scores identically — action_schema §Floating-root known-dependency).
**Verdict-equivalence is re-verified** via the existing round-trip test
(`ConversionResult.ok`, comparing each replay verdict to the oracle verdict): all 42
retained fixtures reconstruct to `ok = True`. This changes the converter's
determinism rules, canonically specified in `warm_start_data_spec.md` (§Determinism,
rulings B/E) — that document carries the authoritative text; this paragraph records
the ruling and its dependency on the floating-root restriction.

## Opening curriculum (policy-layer mask, AFF 1AC only)

A progressive **unlock ladder** scaffolds the opening speech. It is a
**policy-layer mask** applied on top of the environment's structural legal-action
mask — it lives in the policy's `type_mask`/role gating (encoder_spec), **not** in
`check_legality`. The environment's legal-action generator is unchanged and still
enforces structural legality only (environment_shell_spec §Governing principle);
the curriculum only *further* restricts what the policy may sample during the 1AC,
the way warm-start biases the opening without removing any move permanently. Like
warm-start it shapes where learning starts, not what is ultimately learnable: it
binds on the **1AC only** and **never on NEG**.

The ladder:

- **Move 1 of the 1AC must be `introduce(role = advocacy, target = NEW)`.**
  `end_speech` is masked at that decision point — AFF may not pass the 1AC. This is
  the one place the curriculum forces a specific move; it pairs with the
  legality-layer floating-root restriction (which already makes Advocacy/Framework
  the only legal NEW roots), so move 1 is the Advocacy every AFF component roots on.
- **From move 2 onward: Advocacy stays available; Link unlocks.** A Link must
  attach (legality §Floating-root restriction); with only the Advocacy present its
  sole legal attachment is a Support edge to that Advocacy (an Advocacy
  support-attaches only to a Link — judge_spec §2 / action_schema §Principles),
  which is exactly the intended advantage stem.
- **Reading a Link unlocks Uniqueness and Impact.**
- **Reading an Impact unlocks Framework and BallotDirective.**
- **BallotDirective gates on Impact ALONE — explicitly not on Uniqueness.**
  Gating BD on an upstream Uniqueness would reintroduce the **mandatory-uniqueness**
  rule retracted last session (judge_spec §2, "the former two-shape rule is
  retracted") — a tabula-rasa violation — and would make **r36** (a valid AFF graph
  with a BallotDirective and no Uniqueness) unreachable.
- **From the 1NC onward the ladder is fully unlocked.** The curriculum never binds
  after the 1AC and never binds on NEG.

`weigh`, `connect`, attaching `introduce`, and attack edges need **no** curriculum
gate: they are already structurally unavailable on an empty graph and become
available naturally as their endpoints appear (environment_shell_spec §Governing
principle). The ladder gates only the *role* of a fresh `introduce` and the
availability of `end_speech` at move 1.

This is a **semantic** training scaffold (it changes what the policy explores in
the opening), so like λ and the entropy schedule it is recorded per run
(§Adjustment protocol) and any change to it defines a new experiment.

## Hyperparameters

Starting values. All are subject to empirical revision; see Adjustment
protocol below for which may move and on what evidence.

### PPO

| Parameter | Value |
|---|---|
| Learning rate | 3e-4, linear decay |
| Clip range | 0.2 |
| GAE lambda | 0.95 |
| Discount (γ) | **0.999** (ruled; see note) |
| Epochs per batch | 4 |
| Minibatch size | 256 |
| Gradient clip | 0.5 |
| Value loss coefficient | 0.5 |

> **Discount (γ) — RULED 0.999 (explicit user ruling, 2026-08-06),** superseding the
> 0.99 starting value above. γ is a semantic parameter (it changes how far the policy
> looks ahead), so it required a ruling. Rationale: episodes are long (~50–100 actions
> over the full speech budget) and the reward is fully sparse/terminal with the shaping
> bonus off by default, so a high γ is needed for the terminal ballot to back-propagate
> credit to early-episode actions (e.g. 1AC framing). 0.999 was chosen over the 0.95–0.99
> range (typical for short-episode settings) because of the horizon length, and over
> γ = 1.0 to retain some time-preference for value-estimation stability. Recorded in code
> at `training/config.py::SemanticsConfig.discount`.

### Entropy

Initial 0.05, decayed to 0.005 over the first third of training.

Deliberately well above the usual 0.01 default. Three compounding reasons:
the factored action space is large, warm-start narrows exploration around the
demonstrated opening, and the shaping bonus creates a specific collapse risk
toward `extend`.

### Self-play

| Parameter | Value |
|---|---|
| Snapshot interval | every 50 updates |
| Pool cap | 20 checkpoints (most recent 15 + 5 early anchors) |
| Sampling ratio | 50% current-self, 50% pool |
| Ladder evaluation | every 100 updates |

### Shaping and annealing

### Shaping (PBRS)

| Parameter | Value |
|---|---|
| Shaping mechanism | potential-based (PBRS), Φ = Φ_maxdiff |
| λ (PBRS shaping weight) | 0.5 (ruled; semantic) |
| κ (mid-round Φ nascent-channel discount) | **0.0 (NEUTRALIZED 2026-08-12; dead end — see §Reward)** |
| λ schedule | none (fixed; invariance ⇒ no withdrawal) |
| Inert-action penalty coefficient | 0.0 (dormant; no-op re-extend priced via cost) |

λ = 0.5 is a **semantic** parameter — it sets how much the dense Φ-proxy shapes early
learning, not correctness (PBRS is policy-invariant for any λ). Φ ∈ [−1,1] vs the ballot
∈ [0,1], so λ=0.5 makes Φ a half-scale value prior: dense enough to break the sparse-reward
bootstrap barrier, small enough that Φ's imperfections are a modest prior for the critic to
shed. UNSET-until-ruled, ruled to 0.5.

κ (mid-round Φ **nascent channel**) is **0.0 — the channel is neutralized** (§Reward). It was
ruled to 0.3 on 2026-08-10, then found to *degrade* the bootstrap screen (3/5 → 2/5 → 1/5)
and to be structurally incapable of injecting net signal under PBRS invariance, so it was
ruled a dead end and reverted to κ = 0. A residual κ = 0.3 that was never cleaned from the
working tree shipped to the pod and invalidated the 2026-08-12 Screen A run; κ is now pinned
to 0.0. Any future re-attempt is a fresh semantic ruling, not a default.

The retired flat-bonus parameters (`shaping_coef`, `shaping_enabled`) and the anneal-trigger
fields (`anneal_trigger_ballot_winrate`, `anneal_trigger_window_episodes`,
`anneal_decay_updates`) are removed — PBRS is never annealed (§Annealing — retired). The
inert-action penalty coefficient is unchanged (dormant backstop, §Reward).

### Batch and budget

| Parameter | Value |
|---|---|
| Episodes per update | 2,000 |
| First-run budget | 5,000 updates (~10M episodes, ~500M steps) |

At ~4ms per round, 2,000 episodes is roughly 8 seconds of rollout per update
single-threaded, so parallelism is optional early.

The 5,000-update budget is a **first experiment, not a target**. Sparse-reward
problems can need far more samples than dense ones, and nobody knows what a
binary reward over 52-action episodes requires here; the figure is a planned
starting point, and whether it suffices is empirical. Cheap rollouts are what
make an unfavorable answer survivable — a 10× underestimate is an overnight
difference, not an infeasible one.

The number that matters is extension survival rate, not step count: if it has
not moved off the 1.6% random-play baseline within a few hundred updates,
something is wrong and more steps will not fix it.

## Adjustment protocol

Hyperparameter changes fall into two categories, and they are handled
differently.

**Mechanical tuning — delegable.** Learning rate, minibatch size, epochs per
batch, gradient clip, batch size, worker count. These trade throughput and
gradient noise against each other and have no bearing on what the policy
learns. A diverging loss or an underutilized machine is sufficient evidence to
change them.

**Semantic parameters — require a ruling.** Entropy schedule, the PBRS shaping weight λ,
the inert-action penalty coefficient, pool composition and sampling ratio, discount. Each
of these changes *what behavior is rewarded or explored*, not merely how fast it is learned.
(The former flat-bonus coefficient and annealing trigger/window are retired — PBRS is
policy-invariant and never annealed, §Annealing — retired.)

**Never adjusted to make a run look better:** the evaluation protocol, and — for the flat
bonus that used them, now retired — the shaping coefficient and annealing trigger. If a run
stalls, the finding is that it stalled. Relaxing the criterion that revealed the stall
converts a diagnostic into a rationalization, and does so invisibly. (λ is invariance-
preserving, so it is not in this "never adjust" category — but it must still be recorded per
run, below.)

Any parameter change invalidates cross-run comparison. Record the full
configuration with every run and treat a changed config as a new experiment,
not a continuation.

## Configuration

Hyperparameters are read from a config file, never embedded in code. The file
has two top-level sections mirroring the adjustment protocol above:

- **`tuning`** — learning rate, minibatch size, epochs per batch, gradient
  clip, batch size, worker count.
- **`semantics`** — entropy schedule, the PBRS shaping weight λ, the inert-action
  penalty coefficient, pool composition and sampling ratio, discount.

The split is structural, not cosmetic. It exists so that a delegated tuning
change is visibly confined to one section. A flat config would let a
throughput edit touch the shaping coefficient in the same diff, and neither
party would notice until the run finished.

Each run writes its resolved config into its output directory alongside the
checkpoints, so "which config produced this checkpoint" remains answerable
later. **Any config diff touching `semantics` defines a new experiment, not a
continuation of the previous one.**
