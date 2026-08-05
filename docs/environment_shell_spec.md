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
re-implements, approximates, or partially executes any whole-round judge pass.
(Per-node accrual is a deliberate, bounded exception — see Observation
contract.)

## Governing principle: structural legality only

The legal-action generator enforces **structural** legality and nothing else:

- the target node exists (or `target_id = NEW`); for `connect`, both endpoints
  exist
- it is the acting side's turn
- the speech's move budget is not exhausted
- action parameters are well-formed
- a `connect` may not create a self-loop or close a **Support cycle** (the
  directed-cycle rule is detailed under State schema → `connect`)

It does **not** enforce strategic legality. Response-window compliance,
whether an extension will ultimately count, whether a new chain in a rebuttal
can establish offense, whether a spike into a conceded-but-uncontested node is
inert — all of these remain outcomes computed by the judge, not prohibitions
enforced at action time.

**There is no multi-terminal refusal.** Divergent chains are first-class as of
judge v11 — a same-side Support component may have several terminal impacts,
each scored as its own chain (State schema → Divergence) — so the former
"Fence A" is retired from both the generator and `validate_round`. That also
removed the generator's per-action deepcopy probe, so legality checks are now
O(1) structural predicates; the only non-local check is the `connect`
Support-cycle test, a cheap reachability query. `validate_round` at termination
is now the off-vocab-speech check (Fence G) only — still an assertion, satisfied
by construction because the env stamps every node's speech from the current slot.

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

**Shared nodes, convergence, and `connect`.** Targeting an existing node when
you introduce a new one gives that node in-degree — a **shared leaf** (several
nodes pointing at it). It does **not** build **convergence**: two root→impact
paths meeting at a shared node (a diamond — r32's shared uniqueness, the
cross-side shared impact of Phase-0 Option B). `introduce`, creating exactly one
node and its one edge, can only ever grow a **forest** (edges = nodes − roots,
acyclic). Convergence — and any edge between two nodes that already exist — is
built by the **`connect`** action (one move; creates no node). There is no
identity-merge action: the retroactive-fusion case it would have served is
subsumed by `connect`. (An earlier draft of this section described a merge that
"raises in-degree"; no such behavior was ever implemented — the corrected
statement is that plain targeting expresses shared *leaves*, and `connect`
expresses everything non-forest.)

`connect` legality is structural: both endpoints exist, `source ≠ target` (no
self-loops), a valid `edge_type`, and — for a `support` edge — it must not close
a **Support cycle**. The cycle test is on the **directed** Support graph
(authored `source→target`), *not* the undirected one: an undirected-cycle ban
would forbid diamonds, which `connect` exists to enable — a diamond is an
undirected cycle but a directed DAG (all edges orient acyclically toward the
shared impact). So convergence/divergence structures stay buildable; only genuine
circular support (`a → … → a`) is refused. Forbidding cycles also keeps the
degenerate no-terminal-impact component unreachable.

**Divergence.** A same-side Support component may have **more than one terminal
impact** — a shared trunk fanning out to several impacts. Each terminal impact
scores as its **own chain**: its magnitude is the per-path σ product from root to
that impact, and the branch deltas **sum** at the ballot. The shared trunk's σ
multiplies into every branch, so divergence is efficient (one trunk buys N
impacts) and fragile (a good attack on the trunk degrades all N at once) at once
(judge_spec, v11). This was formerly refused by Fence A; it is now first-class.

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
round size is bounded by the budget sum alone. Note `connect` (like `weigh`)
costs a move but adds no node, so round size is not the same as node count.

## Observation contract

The observation is raw graph state, **monotonic settled facts**, and
**node-level accrual output**.

### Monotonic settled facts

Facts that, once determined, can never be reversed by a later speech. No
whole-round judge pass is executed to produce these; they are derived from
graph structure and speech order only.

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

### Node-level accrual (σ and propagated sign)

Per-node DF-QuAD strength and propagated QPN sign **are** exposed, computed
over the partial graph as it stands.

This is a deliberate exception to the "no judge passes" posture, and the
distinction it rests on is load-bearing enough to state explicitly, because
it will otherwise read as an inconsistency:

> **Node-level accrual is well-defined on a partial graph. Whole-round
> evaluation is not.** DF-QuAD accrual asks "given this node's incoming
> attacks, what is its strength" — a question with a sensible answer at any
> point in the round, over whatever attacks currently exist. Extension and
> the ballot tally ask about coverage across the round's *remaining*
> speeches, which have not happened. Those are different questions, and only
> the second is ill-posed mid-round.

Practically, node strength is also what an agent needs in order to decide
whether a position requires frontlining and how — a decision real debaters
make mid-round from exactly this information.

**Inputs differ by caller, deliberately.** The shared function computes over
*exactly the nodes and edges it is handed* — it applies no BD-reachability gate
of its own. The two callers hand it different scopes on purpose:

- the **judge** passes its **BD-reachable subgraph**, so its σ/eff_pol are
  byte-identical to the prior in-place passes (which only ever accrued over
  reachable nodes);
- the **observation** passes the **whole graph**, because until a
  `BallotDirective` exists — usually not until late in the round — nothing is
  BD-reachable, and a reachability gate would report σ for *nothing* through
  most of the round, handing the policy no signal exactly when it needs it.

A reader who notices the observation reporting σ on nodes the judge would ignore
should read it here: it is the caller-scope choice, not a divergence in the
function. The differential test (`tests/test_accrual_shared.py`) compares judge
vs. shared function on *identical* inputs, so it verifies the function; it does
**not** — and cannot — check the caller-scope split. That legibility is this
spec's job.

**The liveness horizon (`as_of`) — same category as the caller-scope split.**
Node accrual is well-defined mid-round only if its *inputs* are, and one input —
which attacks/weighs are LIVE — runs through the extension gate (`node_extension_ok`
/ `node_live_by_any_side`), which asks whether a node is carried through its maker's
speech schedule. Checked against the *full* schedule that gate reaches into speeches
that have not happened: a 1NC attack would need coverage through 2NC/1NR and 2NR to
count, so mid-round it is deemed "not yet extended" and σ ignores it — the ill-posed
"remaining speeches" question arriving indirectly through the *attacker's* liveness.
That would make mid-round σ a constant (measured: σ stayed 1.0 for the whole round,
dropping only at 2NR) and the feature dead weight.

So the gate takes a horizon `as_of` and requires coverage only through speeches at or
before it. The two callers set it exactly like the scope:

- the **judge** passes `as_of = None` (full schedule) — correct at termination and
  byte-identical to prior behavior;
- the **observation** passes `as_of = the current slot`, so an attack registers the
  moment it exists and lapses only if its maker later fails to extend it.

Measured with a 1NC defensive attack on an AFF link: σ = 0.0 from 1NC onward when NEG
sustains it; σ = 0.0 at 1NC/2AC then 1.0 from 2NC/1NR when NEG drops it. At
termination the two horizons coincide (every speech has occurred), so the judge's
accrual and the differential test are unchanged. `as_of` is a parameter of the one
shared function, not a second implementation.

**Single implementation requirement.** Node-level accrual must be factored
into one pure function over nodes and edges, called by both the judge's
accrual pass and the observation layer. The encoder must not carry its own
copy. A drifted second implementation would train a policy against a
slightly different world model than the judge scores it in — a silent
failure mode considerably worse than ordinary spec drift.

**Status: satisfied.** Accrual is factored into `judge.passes.node_accrual` —
the single implementation. The judge routes through it (`pass_accrual`, over its
reachable subgraph) and the observation calls it (over the whole graph); there is
no second copy. Byte-identical oracle verdicts are preserved and pinned by the
per-fixture, node-for-node differential test.

The function operates on the environment's own state representation
**without materializing a `model.Round`** (the observation builds lightweight
`NodeView`/`EdgeView`s that duck-type `model.Node`/`Edge`; the shared function
keys off `.kind`/`.side`/`.speech`/`.liveness` strings, not `isinstance`). It runs
once per `step()` — tens of millions of times across a training run — so it must
not regress to the round-construction pattern; the Phase-4 encoder must reuse this
same function, never its own copy.

### Excluded (provisional, must not appear in the observation)

- extension *eligibility* for chains still alive as candidates
- **chain** magnitudes, the running tally, or any projected verdict (only
  *node-level* accrual is exposed, never chain-level or whole-round output)
- any lookahead or predicted continuation — the environment does not
  predict. Agent-side planning is permitted and lives entirely on the agent.

Rationale for the settled-facts layer: under PPO the critic learns state
value from terminal rewards, so a policy could in principle infer much of it.
Any non-learning agent knows only what the observation shows it.

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
provisional *whole-round* verdict-like signal and (2) carrying a second
implementation of judge semantics. Accordingly: call nothing that produces
chain magnitude, extension/liveness-scoring, or verdict state. **Per-node
accrual is the single sanctioned exception** — and precisely because it is
shared judge semantics, it must be the *same* pure function the judge's accrual
pass calls, not a second copy (Observation contract → Node-level accrual).

## `reset()` / `step()` contract

Gym-style, for compatibility with off-the-shelf PPO implementations.

`reset() -> observation`
Initializes an empty graph at slot 1AC with that slot's budget.

`step(action) -> (observation, reward, done, info)`

- **Illegal actions raise.** An action that fails the legal-action generator's
  structural check raises `ValueError`; the environment does not silently no-op
  an illegal move. Callers consult the generator before stepping.
- **Non-terminal steps**: `reward = 0`, `done = False`. No per-move shaping;
  every shaping term is applied at termination, in the reward, never in the
  observation. (Phase 1 reward shaping that disables presumption for early
  iterations applies to the *judge configuration* at termination, not to
  intermediate steps.)
- **Chain-extension shaping bonus** (`chain_extension_bonus`, default `0.0`):
  on top of the ballot reward, AFF earns a small bonus at termination iff it
  carried **at least one** chain that is *extended, in-scope, and sign +1* —
  regardless of who won. **Binary**: one such chain is worth exactly as much as
  three; magnitude does not scale it. The coefficient is configurable and
  **annealable to zero** — the final policy trains on the terminal reward alone,
  so the default is off (byte-identical to an unshaped env). Rationale: random
  play builds an AFF offense chain ~75% of rounds but *carries* one only ~1.6%
  and passes zero ballot gates, so the terminal reward is constant-zero and
  nothing bootstraps; rewarding chain **existence** teaches "carry a spine" (a
  rule of the game), whereas rewarding chain count/magnitude would teach "extend
  everything" (bad debate — kept emergent). The bonus is an AFF-only auxiliary
  reward: with it enabled the two sides no longer sum to 1. `info['rewards']`
  carries the shaped per-side returns; `info['reward_breakdown']` splits each
  side into `ballot` and `chain_extension_bonus`.
- **Turn advance**: on `end_speech()` or budget exhaustion, the acting side
  and slot advance per `SPEECH_ORDER`.
- **Termination**: after the final slot (2AR) completes, the environment
  materializes the completed graph, asserts structural admission
  (`validate_round`, Fence G), hands it to the judge — which runs its passes
  exactly as for oracle fixtures — and maps the verdict to the binary terminal
  reward. `done = True`. `reward` is reported from **AFF's perspective** (+1 on
  an AFF ballot, 0 on NEG); `info['rewards']` carries the per-side split
  `{AFF, NEG}` so a self-play harness assigns each policy its own return.
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

   With Fence A retired (v11, multi-terminal components now legal),
   `validate_round()` is the off-vocab-speech check (**Fence G**) only — a
   local structural check satisfied by construction, so the assertion never
   fires in practice. The earlier "classify every `validate_round` check as
   local / global / strategic" exercise is closed: the only non-local candidate
   was the multi-terminal refusal, and it was removed rather than enforced.

## Status

Phase 1 shell built against this spec: agent-agnostic state graph, structural
legal-action generator, an observation carrying **monotonic settled facts +
node-level accrual** (σ + propagated sign), and the Gym `reset()`/`step()`
contract with a single judge call at termination. Divergence (v11) and the
`connect` action are landed; Fence A is retired. Node-level accrual is factored
into the single shared `judge.passes.node_accrual`, called by both the judge and
the observation (no second copy), with a per-fixture differential test pinning
byte-identity.

The **Phase-4 encoder** is not yet built; when it is, it must consume the same
`node_accrual` — never its own copy.
