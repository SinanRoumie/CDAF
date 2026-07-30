# CDAF Environment Shell Spec (Phase 1)

## Scope

This spec defines the environment interface that debate agents step through:
state representation, observation contract, legal-action generation, and the
`reset()`/`step()` contract. It is deliberately agent-agnostic. The caller may
be an LLM-in-loop agent (Phase 2) or a trained policy under self-play
(Phase 5); the environment does not know or care which.

Depends on: `action_schema_spec.md` (Phase 0) for the action space,
`judge_spec.md` for all evaluation semantics.

This spec introduces **no new evaluation logic**. The judge is called once,
on a complete graph, at episode termination. Nothing in this document
re-implements, approximates, or partially executes any judge pass.

## Governing principle: structural legality only

The legal-action generator enforces **structural** legality and nothing else:

- the target node exists (or `target_id = NEW`)
- it is the acting side's turn
- the speech's move budget is not exhausted
- action parameters are well-formed
- (the one ruled structural invariant) no action leaves a **reachable**
  same-side Support component with more than one terminal impact — Fence A

**Fence A is reachability-scoped.** It applies only to BD-**reachable**
subgraphs, matching the judge and termination-time admission, which both score
only reachable components. An unreachable same-side multi-terminal blob is
permitted to exist; the generator rejects only the action that would make such a
blob reachable. The maintained invariant is therefore "no **reachable** same-side
Support component has >1 terminal impact after any action." Forbidding deferred
repair makes this a local, monotonic check computable from current state plus the
candidate action (confirmed free against the full fixture corpus).

It does **not** enforce strategic legality. Response-window compliance,
whether an extension will ultimately count, whether a new chain in a rebuttal
can establish offense, whether a spike into a conceded-but-uncontested node is
inert — all of these remain outcomes computed by the judge, not prohibitions
enforced at action time.

Two reasons this boundary is drawn here:

1. **No duplicated semantics.** Drop and extension rules live in exactly one
   place (the judge). If the environment also encoded them to filter actions,
   the two could silently diverge.
2. **Preserved learning signal.** An agent that attempts a rebuttal-introduced
   chain and receives no credit learns the strategic cost. An agent
   structurally prevented from attempting it learns nothing. This matches the
   judge's existing posture throughout `judge_spec.md`, where illegitimate
   moves are scored as inert rather than blocked.

## State schema

Round state is the argument graph plus speech-sequence position.

**Nodes** carry:
- `id` — stable identifier, assigned on creation
- `content` — free text; never read by the judge
- `owner` — `AFF` or `NEG` (single agent per side; no sub-side speaker
  tracking)
- `introduction_speech` — slot in `SPEECH_ORDER` at which the node entered
- `role` — agent-declared at introduction (uniqueness, link, impact,
  advocacy, framework, non-spine); never derived from graph position
- `liveness` — dict of `{speech: "contested" | "conceded"}`; see stamping
  rule below

**Edges** carry:
- `source`, `target` — node ids
- `edge_type` — support / defensive_attack / offensive_attack, declared by
  the agent. Maps directly to the model's `DefensiveAttack` /
  `OffensiveAttack` classes at materialization; nothing is inferred.

**Weigh nodes** additionally carry the two compared node ids and the `favors`
pointer, per Phase 0.

**Shared nodes** are formed by ordinary targeting, not a merge mode. An
`introduce` with a `support` edge into an existing node — from either side —
gives that node genuine cross-side in-degree; downstream attack propagation to
all dependent chains follows automatically from the judge's per-node σ (the
shared-node mechanism, e.g. r32). There is no identity-merge action: the
retroactive-fusion case it would have served is unreachable and unneeded in V1
(see `action_schema_spec.md` §introduce), and it has been removed. (An earlier
draft of this section described a merge that "raises in-degree"; no such behavior
was ever implemented — the corrected statement is that sharing is expressed at
creation by targeting.)

**Sequence state**: current speech slot, moves consumed in the current
speech, remaining budget.

Speech budgets (first-iteration defaults, tunable):

| Slot               | Budget |
|--------------------|--------|
| 1AC                | 8      |
| 1NC                | 8      |
| 2AC                | 8      |
| 2NC + 1NR (merged) | 13     |
| 1AR                | 5      |
| 2NR                | 5      |
| 2AR                | 5      |

Total ceiling: 52 actions per round. There is no separate node-count cap;
round size is bounded by the budget sum alone.

## Observation contract

The observation is raw graph state plus **monotonic settled facts** — facts
that, once determined, can never be reversed by a later speech. No judge pass
is executed to produce these; they are derived from graph structure and
speech order only.

Included:

- **Full graph structure** — nodes, edges, ownership, introduction speech,
  in-degree on shared nodes.
- **Closed-window drop outcomes** — for any node whose response window has
  already passed with no opposing clash, the drop is settled permanently.
- **Permanent extension failure** — extension *success* is not knowable
  mid-round, but failure is monotonic: once a spine node misses coverage in
  one of its own side's speeches, that chain is dead and no later speech
  revives it. Chains in this state are flagged.
- **Reachability** — which nodes currently route to an impact, which are
  orphaned. "Routes to an impact" is undirected connectivity to any
  impact-role node; an impact trivially satisfies it (it reaches itself), so
  *orphaned* means a **non-impact** node with no undirected path to any impact.
- **Sequence state** — current slot, remaining budget.

Excluded (provisional, must not appear in the observation):

- extension *eligibility* for chains still alive as candidates
- DF-QuAD magnitudes or any accrual output
- any running or projected verdict
- any lookahead or predicted continuation — the environment does not predict.
  Agent-side planning is permitted and lives entirely on the agent.

Rationale for including settled facts at all: under PPO the critic learns
state value from terminal rewards, so a Phase 5 policy could in principle
infer much of this. A Phase 2 LLM agent has no critic and knows only what the
observation shows it. The settled-facts layer is therefore load-bearing for
Phase 2 and merely convenient for Phase 5.

## Liveness stamping

When an agent extends or concedes a node, the environment records a liveness
entry for the current speech. The `contested` / `conceded` status is
**derived structurally**, mirroring `model/convert.py`'s existing predicate —
never taken from which action verb the agent used.

Rationale: the same graph must produce the same verdict regardless of how it
was built. If status were a function of action history, an env-built round and
a fixture-built round with identical structure could return different
verdicts, and oracle fixtures would stop constraining env behavior. Deriving
it preserves fixture/env equivalence. It also prevents an agent from
*asserting* that a node is contested by verb choice where nothing actually
opposed it.

Two implementation constraints:

- The env must match `convert.py`'s **actual** predicate exactly, not a
  paraphrase of it. Confirm what it keys on before implementing — since only
  one side speaks per slot, "an opposing attack targeted that node that
  speech" cannot mean a same-slot opposing action, so the real rule is
  presumably about live opposing attacks as of that speech.
- Status is computed at end-of-speech or at `to_round()` materialization,
  not at the moment the extend action fires, since later actions within the
  same speech can change what is true.

## Reuse of judge structural primitives

The env **imports** pure, side-effect-free structural helpers (speech
ordering, response-window computation) from `passes.*` rather than inlining
copies. Reimplementing window logic in the env would create exactly the
divergence risk the "do not partially execute any judge pass" constraint
exists to prevent.

The constraint's purpose is to stop the env from (1) producing
provisional verdict-like signal and (2) carrying a second implementation of
judge semantics. Accordingly: call nothing that produces accrual, magnitude,
liveness-scoring, or verdict state. Structural primitives only.

## `reset()` / `step()` contract

Gym-style, for compatibility with off-the-shelf PPO implementations.

`reset() -> observation`
Initializes an empty graph at slot 1AC with that slot's budget.

`step(action) -> (observation, reward, done, info)`

- **Illegal actions raise.** An action that fails the legal-action generator's
  structural check raises `ValueError`; the environment does not silently no-op
  an illegal move. Callers consult the generator before stepping.
- **Non-terminal steps**: `reward = 0`, `done = False`. Binary terminal
  reward only; no per-move shaping in V1. (Phase 1 reward shaping that
  disables presumption for early iterations applies to the *judge
  configuration* at termination, not to intermediate steps.)
- **Turn advance**: on `end_speech()` or budget exhaustion, the acting side
  and slot advance per `SPEECH_ORDER`.
- **Termination**: after the final slot (2AR) completes, the environment
  hands the completed graph to the judge, which runs its six passes exactly
  as it does for oracle fixtures. The verdict maps to binary terminal reward.
  `done = True`. `reward` is reported from **AFF's perspective** (+1 on an AFF
  ballot, 0 on NEG); `info['rewards']` carries the per-side split `{AFF, NEG}`
  so a self-play harness assigns each policy its own return.
- **`info`**: judge diagnostics on terminal steps (`DROP`,
  `EXTENSION_FAIL`, verdict rationale) for logging and debugging. Not part of
  the observation; not visible to the agent as training signal.

## Validation metric (not a reward, not a mechanism)

Construction-and-compression — broad offense in constructives, narrowing to a
specific ballot path in the back half — is held as a **falsifiable prediction
about emergent behavior**, deliberately not encoded anywhere in the
environment. If a trained policy develops it spontaneously, that is evidence
the action space and judge generate genuine strategic pressure. If it never
emerges, the environment is failing to reproduce an incentive gradient real
debate has, and that is a diagnostic worth acting on.

## Open items requiring a ruling before implementation

1. **Node role assignment — RULED, no longer blocking.** Roles are
   agent-declared at introduction as an `introduce` parameter, never derived
   from graph position. `action_schema_spec.md` amended accordingly.

2. **Judge invocation surface — RESOLVED.** The judge exposes
   `judge(round) -> (ballot, trace)`, accepting an in-memory `model.Round`.
   No fixture-file serialization needed. `env/validator.py` already sets the
   precedent of constructing `Round` objects in memory and calling the judge
   directly. The termination step materializes accumulated state via
   `to_round()` and calls `judge()` once.

3. **Terminal reward on validation failure — RULED: unreachability via the
   legal-action generator.** Invalid rounds are prevented at action time
   rather than scored at termination. Consequently `validate_round()` at the
   terminal step is an **assertion, not a branch**: if it ever fails, that is
   an environment bug and must raise loudly rather than return any reward
   value. No number is safer than raising, because any number is something a
   policy can learn to chase.

   Implementation dependency — enumerate what `validate_round()` /
   `is_valid()` check and classify each:

   - **Local structural checks** (well-formedness of a single action or its
     immediate target) — enforceable directly in the generator, no tension
     with the structural-legality-only principle.
   - **Global round-level checks** (conditions on the completed round as a
     whole, e.g. a required role existing somewhere) — not enforceable by
     per-action filtering, since no single action violates them. Requires an
     explicit mechanism (gating `end_speech()` or reserving a final budget
     slot), which is coercion on the action space and needs its own ruling.
     Note the emergence cost: forcing an agent to satisfy a requirement means
     it never learns that failing to satisfy it loses.
   - **Strategic checks**, if any exist — must NOT move into the generator.
     Their presence would mean this ruling needs revisiting.

## Status

All three items are ruled. Item 3 carries an implementation dependency: the
`validate_round()` check classification must be produced before the
legal-action generator's validity fences and the terminal assertion are
written. A follow-on ruling is required only if any check falls into the
global round-level bucket. The rest of the shell is unblocked.