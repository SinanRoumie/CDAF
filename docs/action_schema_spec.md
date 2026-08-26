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
- **Structural incoherence is illegal; only context-dependent inertness is
  learned.** Several structurally-incoherent moves that can *never* be meaningful in
  any round state are disallowed by the legal-action generator — masking them removes no strategic
  distinction, because they were never a real option: a **same-side attack** (an
  attack edge between two nodes of the same side), an **offense at a non-polarity
  node** (an `offensive_attack` where an endpoint is not a Link or Impact), a
  **redundant connect** (a `connect` duplicating an existing edge), an **Advocacy at
  either end of an attack edge** (an Advocacy may be neither the source nor the target
  of a `defensive_attack` or `offensive_attack`, regardless of edge_type — you support
  or outweigh a proposal, you never attack it or attack *from* it; judge_spec §2), and
  an **Advocacy support-attached to a non-Link node** (an Advocacy carries Support edges
  only to `Link` nodes — its premises route through a Link, never a bare Uniqueness,
  Impact, or BallotDirective), a **cross-kind weigh** (a `weigh` whose two operands are
  different kinds — no matching factor to rank; the judge scores a mixed-type
  `Comparison` inert in every state, judge_spec §3.4/§6.5), and a **cross-side
  defensive attack on a BallotDirective** (a `defensive_attack` across sides whose
  target is a `BallotDirective` — a BD bears no consumed magnitude, so the edge is
  judge-invisible in every state; Ruling 2 V1a). A **no-op
  re-extend** (an `extend`/`concede` on a node already carried this speech) is
  *context-dependent* — extending an uncarried node is a real, often-correct move,
  and only this specific state makes it inert — so it stays **legal** and is priced
  by **cost**: it consumes a full budget slot, bypassing the extend-batch discount,
  so a wasted carriage costs the same as any other wasted move. The policy still
  may do it; it just pays the real opportunity cost, rather than the judgment being
  hardcoded as a prohibition (see environment_shell_spec §Governing principle and
  §Liveness stamping, rl_training_spec §Reward).

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

**Advantage vs. disadvantage is the introducing side, descriptively.** A Link attached to an
Advocacy reads as an **advantage** (AFF) / **disadvantage** (NEG) — this falls out of side-routing
(`resolve_chains` unions only **same-side** Support edges, so a chain's `side` is fixed by its
members' introducing side). A **cross-side** NEG-Link→AFF-Advocacy Support edge does **not** merge
the Advocacy into the disad's same-side component, and by judge_spec §5.3 (absorbing advocacy) it
does not anchor the disad to the AFF framework — **but it is NOT inert:** it is one way a NEG disad
**roots in the shared plan** (judge_spec §2, rule 4) — the direct `link → advocacy` root. A scored
NEG chain that roots in **neither** an AFF Advocacy nor a Framework collapses `unrooted_disad`.

**NEG-offense rooting (rule 4, judge-time enforced, not a legality mask).** A scored NEG chain must
reach a **shared premise** over Support edges: a **disad** reaches the **AFF Advocacy** — directly
(`link → advocacy`) or via the fusion idiom `advocacy → uniqueness → link` (r27); a **framework
argument** reaches a **Framework** (its offense is evaluated under a framework it reads, not a link into the
plan). There is **no mandatory per-Link uniqueness** (the former two-shape rule is retracted): a
uniqueness on a disad Link is optional, drawn only to pre-empt a non-unique. Unlike the
same-side-attack / advocacy-attachment rules above (masked at edge creation), this is a **whole-graph
reachability check at scoring**. A disad joined to the AFF side by an attack or a non-Support path
— its component holds the AFF Advocacy but its spine does not reach it over Support — is **not**
rejected at creation; it is scored 0 with `collapse_reason = "unrooted_disad"` (the disad analog of
`no_link_premise`). A *strictly disconnected* floating disad — its own component with neither an
Advocacy nor a Framework — is additionally **unreachable at action time** under the floating-root
restriction below; the judge-time collapse remains the enforcement for every *connected* case.

**Disad-through-an-AFF-Link (Support Link→Link) vs. the turn (Attack Link→Link) — distinct in the
data model, similar in the builder.** A NEG Link may support-attach to an AFF Link ("I *build on*
your mechanism, and it *also* yields a negative terminal"); because the AFF Link roots in the
Advocacy, the disad's spine reaches the Advocacy one hop further along and roots per rule 4 — no
uniqueness of its own required. The pre-existing turn is a `DefensiveAttack`/`OffensiveAttack` edge
between the same node types ("I *refute/delink* your mechanism", §3.4/§3.5). They are unambiguous to
the judge and to a content-generator (the `etype` field), but **a support line and an attack line
between the same two Links look alike in the graph-builder UI** — inspect edge type, not just
endpoints, when reading these by eye.

Any node on the graph is a legal `target_id`, regardless of which side
introduced it or which side is introducing now. Own-side targeting is legal
(a debater may attach to their own prior nodes). All targeting is still subject
to existing response-window and extension rules — legality of the target does
not waive those constraints.

**Floating-root restriction — only Advocacy or Framework may root a component
(legality mask, both sides, all speeches).** An `introduce` with `target_id = NEW`
is legal **only for `role ∈ {advocacy, framework}`**. Every other role must attach
to an existing node at creation. Equivalently: **every connected component of the
finished graph contains an Advocacy or a Framework.** Two reasons, one present and
one anticipated: (1) **construction redundancy** — for any component that already
has a legal root (an Advocacy or a Framework), building a further node by
float-then-`connect` (two actions) reaches a graph the judge scores identically to
an attaching `introduce` (one action; the judge is direction-agnostic, so
orientation is immaterial), so the restriction removes only a strictly-costlier
redundant construction route — no reachable terminal graph and no strategy, the
same category as an intra-speech ordering restriction; (2) a planned LLM
content-generation layer conditions each node's generated prose on the node it
attaches to, so a parentless non-root node has nothing to condition on. **Why
Framework roots alongside Advocacy:** the judge already treats both as chain roots
— a NEG offense chain roots at an AFF Advocacy *or* its own Framework (judge_spec
§2, rule 4) — so permitting Framework as a component root makes the legality layer
agree with what the judge already treats as a root; the content-generation
rationale points the same way, since a framework argument is a standalone stateable
premise with no upstream parent to condition on, exactly like an Advocacy, unlike a
floating Link which has nothing to condition on and nothing to say. The rule is
**not** justified by inertness: a floating non-root node can be bridged into a chain
by a later `connect` (which has no same-component precondition) and is inert only
*until* connected. The only reachable graphs it removes are components with neither
an Advocacy nor a Framework — where nothing legal can root them — a deliberate
foreclosure (environment_shell_spec §Governing principle: deliberate action-space
foreclosure); within the oracle corpus that is fixture `E` alone. Enforced in
`check_legality` — **not** a curriculum mask and **not** a judge rule. `connect`'s
component-bridging capability is unchanged.

**Known future dependency — the flip is verdict-invisible but not content-neutral.**
Because the judge is direction-agnostic for Support edges (chain membership is
undirected union-find; attack orientation is by speech recency, not stored edge
direction), the agent builds many edges in the *reverse* of their authored
orientation: a sink node such as a BallotDirective is created by an attaching
`introduce` in which the **new node is the edge source**, storing e.g. `bd →
impact` where a fixture authored `impact → bd`. This is invisible to the judge
(confirmed by warm-start replay equivalence) and therefore fully legal, but it is
**not** semantically neutral for a content layer, which will read edge direction as
"A is a premise for B." When that layer lands it will need either a canonicalization
pass over stored orientations or a directed `introduce`. Recorded as a dependency;
not designed now.

### `extend(node_id)`

Marks **exactly one node** — the named node — as carried for the current speech.
`extend` is **atomic**: it stamps that node's liveness and nothing else. There is
**no path-walking and no propagation**; extending a node does not touch its chain,
its impact, or anything it supports. An agent may extend a link while deliberately
**not** extending its impact, letting the impact die — partial-chain carriage is
fully expressible.

**Cost — a speech-wide batch, not per-node-flat and not path-length-based.** The
environment keeps a running counter `extends_this_speech` of `extend`/`concede`
actions taken so far in the current speech (reset to 0 at each speech boundary). The
marginal cost of the next carriage is

    ceil((count + 1) / K) − ceil(count / K)

where `count = extends_this_speech` and `K` is a named tunable constant
(§Turn structure and speech budgets), default **4**. This is **1** on the 1st,
(K+1)-th, (2K+1)-th … carriage of the speech and **0** otherwise, so **N carriages
over a speech cost `ceil(N / K)` slots in total** — one slot buys K carriages. The
discount is scoped to the **whole speech**, not to any chain: an agent gets the same
`1-slot-per-K` rate whether the K nodes it carries are all on one chain or scattered
across unrelated arguments. There is nothing chain-scoped to price, because there is
no path-walk.

  Worked example (K = 4): the 1st–4th carriages of a speech cost 1/0/0/0 (cumulative
  1); the 5th–8th cost 1/0/0/0 again (cumulative 2); the 9th costs 1 (cumulative 3).
  N carriages cost `ceil(N / 4)` — e.g. 3 → 1, 4 → 1, 5 → 2, 8 → 2, 9 → 3.

**No-op re-extend exception.** Re-extending a node **already carried this speech**
changes nothing (the liveness stamp is idempotent). It remains legal but does
**not** get the batch discount: it costs a **full slot** (1), regardless of
`extends_this_speech`, like any wasted move. Only *distinct* carriages — extends of
a node not yet carried this speech — earn the `1-slot-per-K` rate. This prices inert
re-extension at its true opportunity cost instead of letting the batch discount make
it near-free.

  (The 1 lands on the *first* carriage of each K-group — the marginal of the given
  formula. Cumulative cost after N carriages is `ceil(N / K)`.)

**Target may be own-side or opponent-side.** Like every other targeting action
(§introduce), `extend` accepts *any* existing node regardless of which side
introduced it. Cross-side targeting is legal and carries no special cost or
restriction; it costs against the same `extends_this_speech` batch as any carriage.

### `concede(node_id)`

**`concede` is not a distinct mechanic — it is `extend`.** Same atomic single-node
stamp, same speech-wide batched cost, same per-speech liveness — identical in every
respect. "Concede" is simply the descriptive label used when the targeted node
happens to be **opponent-owned**; `extend` is the label when it is your own.
Functionally there is one action with one implementation, and both increment the same
`extends_this_speech` counter.

There is **no** permanence, no free/zero-cost case, no one-shot restriction, and no
new node state beyond the per-speech counter: a node is kept live speech-by-speech
exactly as with any extend (re-stamped each speech it must remain live), and
conceding never restricts future actions (attacks included) on that node — ordinary
legality rules apply throughout.

**Structural note on `extend` vs `concede`.** The two are the same structural act — a
per-speech liveness stamp on the single named node, at the same batched cost. They
differ only as agent-facing intent / logging: `concede` names the cross-side case,
`extend` the same-side case. Because contested/conceded *status* is derived
structurally from the graph (never from the verb), the labels coincide in the
materialized graph and carry no distinct semantics.

### `weigh(node_a, node_b, favors, justification)`

Introduces a comparison between two existing nodes.

- `node_a`, `node_b`: the two nodes being compared. May be owned by the same
  side (a debater prioritizing between two of their own impacts) or opposing
  sides.
- **Same-kind operands (structural legality, enforced at creation).** `node_a`
  and `node_b` must be the SAME kind (both Links, both Impacts, both Frameworks,
  both Uniquenesses, both Weighings, or both BallotDirectives). A CROSS-KIND weigh
  has no matching factor to rank — the judge already scores a `Comparison` over a
  mixed-type pair as inert (judge_spec §3.4 table, §6.5) — so it is now rejected by
  the legal-action generator at creation rather than sampled and scored inert
  (environment_shell_spec §Governing principle: structural legality only). All 18
  oracle-corpus weighs are same-kind, so no verdict changes. Two same-kind sub-cases
  have no live consumer and are documented so they are not later mistaken for gaps:
    - **BallotDirective vs BallotDirective — DESCRIPTIVE / INERT.** Legal, but a BD
      bears no strength any pass consumes (judge_spec §3, node table: a BD is never
      consumed by its own strength), so the weigh has no effect. Deliberate, not a gap.
    - **Uniqueness vs Uniqueness — consumed ONLY when the two are COMPETING.**
      Competing means EITHER (A) an explicit `DefensiveAttack` edge runs between the
      two weighed uniquenesses (the r15/r16 structure — a non-unique `Uniqueness`
      defensive-attacks the `Uniqueness` the other supports), OR (B) the
      `nonunique_on_link` route — a non-unique `Uniqueness` defensive-attacks the
      `Link` that the other weighed `Uniqueness` supports. Both are the same clash,
      resolved through the all-or-nothing poison gate at the 0.5 threshold (judge_spec
      §3.4 Ruling A, §12.4.3). Two uniquenesses with NEITHER relationship are inert —
      nothing for `resolve` to break — legal but effectless.
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

Actions do not all cost one move. `introduce`, `weigh`, and `connect` cost one
slot; `end_speech` costs none (it ends the turn). `extend`/`concede` cost the
**marginal of a speech-wide `ceil(count / K)` batch** — 1 on the 1st, (K+1)-th,
(2K+1)-th … carriage of the speech and 0 otherwise, so N carriages cost
`ceil(N / K)` slots total (§extend). **`K` is a named constant defined beside the
speech-budget constants** (in code: `env/actions.py`, `EXTEND_COST_K`, alongside
`SPEECH_BUDGET`), first-iteration default **4**, tunable on the same footing as the
speech budgets. The economy that discourages disconnected or low-value spam is
still emergent — moves spent on a node that never becomes reachable, or is never
extended when required, do not pay off — now with a batched cost on extension so
keeping many nodes alive in the back half carries real budget pressure rather than
being free, while the `1-slot-per-K` rate keeps ordinary spine carriage cheap.

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
