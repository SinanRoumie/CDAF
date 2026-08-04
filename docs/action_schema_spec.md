# CDAF Action Schema Spec (Phase 0)

## Scope

This spec defines the graph-native action space that debate agents (LLM-in-loop
in the near term, a trained policy under self-play later) operate on. It does
not define the RL algorithm, the policy architecture, or the LLM harness
implementation. Those are separate documents. This spec exists so that both
the LLM-in-loop phase and the self-play phase can be built against the same
environment interface without a rewrite in between.

The judge's evaluation logic is unchanged by this spec. Every mechanism
described here routes through existing judge passes (extension checking,
max-aggregation, QPN/DF-QuAD propagation, reachability). No new judge
machinery is introduced.

## Principles this schema follows

- **Tabula rasa**: no action or parameter encodes a content-quality judgment.
  Where an action carries free text (e.g. a weigh justification), that text is
  never read or scored by the judge. Only the structural claim an action makes
  is evaluated.
- **Emergence over hardcoding**: where a behavior is always strategically
  correct (e.g. an agent never favoring its opponent's node in a weigh),
  the schema does not encode that as a structural restriction. The action
  space stays uniform; the dominance is left for self-play to discover and
  exploit on its own.

## Action types

### `introduce(content, role, target_id | NEW, edge_type)`

Introduces a claim. This single action covers two cases that would
otherwise be separate move types:

- **New node**: `target_id = NEW`. Creates a fresh node with no existing
  identity.
- **Attach to existing node (structural)**: `target_id` names an existing
  node and `edge_type` declares the relationship.

**`introduce` builds shared leaves, not convergence — use `connect` for that.**
A correction to an earlier claim: targeting an existing node when you introduce a
new one gives that node in-degree (a **shared leaf** — several nodes pointing at
it), but it can never build **convergence** — two root→impact paths meeting at a
shared node (a diamond: r32's shared uniqueness, the cross-side shared impact of
Phase-0 Option B). The closing edge of a diamond runs between two nodes that
*already exist*, and `introduce` — creating exactly one node and its one edge —
can only ever grow a **forest** (edges = nodes − roots, acyclic). Convergence,
and any other non-forest structure, requires an edge between two pre-existing
nodes, which is exactly what the `connect` action provides (below). An earlier
draft also carried an "identity merge" case for *retroactively fusing two
already-introduced nodes*; that is subsumed by `connect` (fusing is a special case
of connecting) and was removed.

**Edge type vocabulary.** `edge_type ∈ {support, defensive_attack,
offensive_attack}`. Turn status is *declared*, not derived: an agent electing
`offensive_attack` is making a turn; electing `defensive_attack` is making a
takeout. These are different arguments a debater chooses between, not one
argument the judge classifies after the fact. This maps directly onto the
model's existing `DefensiveAttack` / `OffensiveAttack` edge classes at
materialization, with nothing left to infer and no placeholder subtype
needed.

**Role declaration.** `role` is declared by the agent at introduction —
uniqueness, link, impact, advocacy, framework, or a non-spine role. It is not
derived from graph position. A node's role is therefore a *strategic
assertion* made by the introducing agent, not a structural fact computed by
the environment. This keeps the environment out of the business of
interpreting what a claim is, and it means a misdeclared role carries
whatever structural consequence the judge's existing rules impose rather than
being corrected at action time. The judge's extension rule (§6, spine-node
coverage) reads declared roles directly.

The concrete role vocabulary is {uniqueness, link, impact, advocacy,
framework, ballot_directive}. `ballot_directive` is the non-spine role — the
discovery root the ballot needs. Weighing is *not* a role: a Weighing node is
produced only by the `weigh` action, never by `introduce`.

Any node on the graph is a legal `target_id`, regardless of which side
introduced it or which side is introducing now. Own-side targeting is legal
(a debater may attach to their own prior nodes). All targeting is still subject
to existing response-window and extension rules — legality of the target does
not waive those constraints.

### `extend(node_id)`

Marks an existing node as extended for liveness purposes. Standard mechanism,
unchanged by this schema.

### `concede(node_id)`

Marks an existing node as conceded. Standard mechanism, unchanged.

**Structural note on `extend` vs `concede`.** Both actions have the same
structural effect — a liveness stamp for the current speech. Because
contested/conceded *status* is derived structurally from the graph (never from
the verb the agent used), the two coincide in the materialized graph and differ
only as agent-facing intent and logging.

### `weigh(node_a, node_b, favors, justification)`

Introduces a comparison between two existing nodes.

- `node_a`, `node_b`: the two nodes being compared. May be owned by the same
  side (a debater prioritizing between two of their own impacts) or opposing
  sides.
- `favors`: a pointer at one of the two compared nodes (`node_a` or
  `node_b`) — not a side/ownership flag. This is required in both the
  cross-side and own-side case, since ownership alone cannot disambiguate a
  same-side comparison. `favors` is now read by the judge as the weigh's
  preference (judge_spec §6.5), closing the earlier gap where it had no
  structural channel: a **cross-side** weigh consumes it (a legacy round
  without it falls to the own-side default); a **same-side** weigh is
  descriptive-only in V1 (recorded via `favors_source`, zero tally effect —
  see judge_spec §6.5 and the parked question there).
- `justification`: free text. The judge never reads or scores this text. It
  exists only as the agent's (or a human reviewer's) rationale.

A `weigh` action produces a directional, signed node that feeds into the
existing QPN/DF-QuAD propagation pass like any other signed node. Competing
weigh claims on the same pair of nodes are resolved by the same mechanism
already used for competing signs elsewhere in the graph — no special
tie-breaking logic is introduced by this schema.

No mechanism-selection parameter (magnitude/probability/timeframe/etc.) is
included. Weighing is a preference, expressed as a directional edge with
unstructured justification.

### `connect(source_id, target_id, edge_type)`

Adds a relationship edge between two nodes that **already exist**. Creates no
node. Costs one move, like every other action — a cross-application costs speech
time, so free edges would break the budget economy that keeps the action space
honest.

This is the only action that can add an edge between two pre-existing nodes, so
it is what makes **convergence** (two paths onto a shared node) and any other
non-forest structure buildable at all — `introduce`, creating a node and its one
edge, only ever grows a forest (see the `introduce` note above). It closes the
expressiveness gap that left r32's shared-uniqueness diamond and Phase-0 Option
B's cross-side shared impact unbuildable.

- `edge_type` ∈ {support, defensive_attack, offensive_attack}.
- Structural legality: both endpoints exist, `source_id ≠ target_id` (no
  self-loops), and — for a `support` edge — the connect must not **close a
  Support cycle**. Cycles don't crash the judge but their semantics are
  unaudited, and unaudited territory a trained policy can reach is where
  undiagnosable exploits live; forbidding them also makes the degenerate
  no-terminal-impact component unreachable. The cycle test is on the *directed*
  Support graph (authored source→target), so convergence/divergence DAGs — which
  orient acyclically toward the shared impact — stay buildable; only genuine
  circular support (`a → … → a`) is refused. If cycles turn out to be
  strategically meaningful later, they return as a designed feature with fixtures.

### `end_speech()`

Terminal action for a turn. An agent may call this before exhausting its
move budget for that speech; it is not required to use every available move.

## Turn structure and speech budgets

A speech is one turn, composed of a sequence of the above actions, terminated
by either `end_speech()` or exhaustion of the speech's move budget.

Move budgets are set per speech position, in the ratio of policy debate
speech times, with 2NC and 1NR merged into a single budget-bearing turn:

| Speech slot         | Budget |
|----------------------|--------|
| 1AC                  | 8      |
| 1NC                  | 8      |
| 2AC                  | 8      |
| 2NC + 1NR (merged)   | 13     |
| 1AR                  | 5      |
| 2NR                  | 5      |
| 2AR                  | 5      |

These ratios are a first-iteration default, not a fixed rule — they are
expected to be tuned empirically once the environment is running.

No action type costs more or less than another against this budget.
`introduce`, `extend`, `concede`, and `weigh` all consume one move. The
economy that discourages disconnected or low-value new-node spam is not an
explicit cost — it is the interaction between a finite per-speech budget and
the existing reachability/extension rules: moves spent on a node that never
becomes reachable, or is never extended when required, do not pay off. This
is a deliberate choice to keep the cost structure emergent rather than
hardcoded, consistent with the schema's overall design principle.

## Explicitly out of scope for this spec

- Whether a weigh action counts as engagement for drop-detection purposes
  (parked judge question, unresolved, interacts with this schema but is not
  settled here).
- Whether extended-but-unanswered defensive arguments must be re-mentioned in
  the maker's final speech (parked judge question, same status).
- Any policy architecture, LLM harness implementation detail, or RL
  algorithm detail. Those belong in the environment shell and RL plan
  documents (Phase 1 onward).

## Status

All four open points from the Phase 0 discussion are ruled and reflected
above: unified `introduce` action for new/attach/merge, per-speech budgets at
the stated ratio, uniform per-action cost, and `favors` as a node pointer
rather than a side flag. Amended subsequently: `introduce` carries an
agent-declared `role` parameter, and `edge_type` carries the three-way
support / defensive_attack / offensive_attack vocabulary with turn status
declared rather than derived. Both gaps were surfaced during Phase 1 design.
This spec is ready to be treated as the committed
Phase 0 artifact; Phase 1 (environment shell) should be built against it.
