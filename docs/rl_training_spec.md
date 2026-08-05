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

**Chain-extension shaping bonus** (implemented, coefficient default 0.0):
awarded at termination if AFF has at least one chain that is extended,
in-scope, and sign +1 — regardless of who won.

- **Binary.** One qualifying chain is worth the same as three. Magnitude does
  not scale it.
- **Why binary:** rewarding chain existence teaches "carry a spine," which
  restates a gate the judge already applies. Rewarding count or magnitude
  would teach "extend everything," which is bad debate and is exactly the
  strategic judgment that must stay emergent.
- **Annealed to zero** before the run ends, so the final policy is trained on
  the terminal reward alone.

### Annealing trigger

Conditional, not scheduled.

A fixed schedule risks removing the only signal the policy has before it has
bootstrapped, and the bootstrap duration is unknown. Decay therefore begins
only once AFF is demonstrably winning without the crutch.

**Trigger on AFF ballot win rate, never on total reward.** Triggering on
total reward would let the bonus inflate its own trigger. Once ballot win
rate crosses a threshold, decay the coefficient to zero over a fixed window.

Log ballot win rate and shaped return as separate series. If they diverge —
shaped return climbing while ballot win rate stays flat — the policy is
farming bonuses rather than learning to win, and the run needs attention
rather than more steps.

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
bought and would cost re-authoring all 46 oracle fixtures under different
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

## Hyperparameters

Starting values. All are subject to empirical revision; see Adjustment
protocol below for which may move and on what evidence.

### PPO

| Parameter | Value |
|---|---|
| Learning rate | 3e-4, linear decay |
| Clip range | 0.2 |
| GAE lambda | 0.95 |
| Discount (γ) | 0.99 |
| Epochs per batch | 4 |
| Minibatch size | 256 |
| Gradient clip | 0.5 |
| Value loss coefficient | 0.5 |

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

| Parameter | Value |
|---|---|
| Shaping coefficient (while active) | 0.1 |
| Decay trigger | AFF **ballot** win rate > 15% over a 1000-episode window |
| Decay schedule | linear to zero over 200 updates after trigger |

Coefficient 0.1 is large enough to matter against a binary terminal reward
and small enough that winning always dominates carrying a chain.

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

**Semantic parameters — require a ruling.** Entropy schedule, shaping
coefficient, annealing trigger and window, pool composition and sampling
ratio, discount. Each of these changes *what behavior is rewarded or
explored*, not merely how fast it is learned. Loosening the annealing trigger
because the run has not reached 15% is not tuning — it is deciding the policy
may keep its crutch, which is exactly the decision the trigger exists to
prevent from being made casually.

**Never adjusted to make a run look better:** the shaping coefficient, the
annealing trigger, and the evaluation protocol. If a run stalls, the finding
is that it stalled. Relaxing the criterion that revealed the stall converts a
diagnostic into a rationalization, and does so invisibly.

Any parameter change invalidates cross-run comparison. Record the full
configuration with every run and treat a changed config as a new experiment,
not a continuation.

## Configuration

Hyperparameters are read from a config file, never embedded in code. The file
has two top-level sections mirroring the adjustment protocol above:

- **`tuning`** — learning rate, minibatch size, epochs per batch, gradient
  clip, batch size, worker count.
- **`semantics`** — entropy schedule, shaping coefficient, annealing trigger
  and window, pool composition and sampling ratio, discount.

The split is structural, not cosmetic. It exists so that a delegated tuning
change is visibly confined to one section. A flat config would let a
throughput edit touch the shaping coefficient in the same diff, and neither
party would notice until the run finished.

Each run writes its resolved config into its output directory alongside the
checkpoints, so "which config produced this checkpoint" remains answerable
later. **Any config diff touching `semantics` defines a new experiment, not a
continuation of the previous one.**
