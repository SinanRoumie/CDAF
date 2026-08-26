# CDAF Warm-Start Data Spec (Phase 5, draft)

> **STATUS: FINAL DRAFT (ready to commit).** This document defines how to turn the
> static fixture corpus into imitation-learning `(observation, action)` pairs. **All
> eight original open questions (A–H) are resolved** (see §Resolved rulings) and folded
> into the strategy below; **no open questions remain**. The one outstanding item is a
> task, not a question — importing `8node1ac.json` as the warm-start 1AC fixture (H) —
> which does not block writing the core inversion. Conversion code can be written
> against this document.

## Scope

This spec defines how a finished, static argument graph (a builder-loadable
fixture: `tests/oracle/*.json`, and any future round saved in the same schema) is
**inverted** into a legal action sequence, so that replaying that sequence through
the environment yields the `(observation, action)` pairs the imitation warm-start
consumes.

It does **not** define the supervised-training objective, the optimizer, or how
warm-start weights hand off to PPO — that is `rl_training_spec.md` (§Imitation
warm-start). It introduces no judge or environment semantics; it only *orders*
actions the environment and judge already define
(`action_schema_spec.md`, `environment_shell_spec.md`).

Depends on: `action_schema_spec.md` (action vocabulary), `environment_shell_spec.md`
(legal-action generator, `step()`/observation contract, liveness stamping),
`rl_training_spec.md` (how warm-start is consumed), `model/serialize.py` +
`model/convert.py` (fixture load / v1→v2), `judge_spec.md` (direction-agnostic
scoring).

## How warm-start consumes the output (from `rl_training_spec.md`)

> The policy is initialized by supervised training on demonstration actions from
> the hand-authored 1AC and the fixture corpus, then trained with PPO from those
> weights. … The policy plays every speech from step one.

So the deliverable is, per fixture, an **ordered list of legal env actions**. The
training loop constructs a fresh `CDAFEnvironment`, and for each action in order:

1. calls `observe(state)` → the observation the policy would have seen,
2. records the pair `(observation, demonstrated_action)`,
3. `step(demonstrated_action)` to advance.

The imitation loss is then the factored cross-entropy of the policy heads (under the
legal-action mask, `policy/heads.py`) against each demonstrated action's components.
Two consequences constrain inversion:

- **Every action in the sequence must be structurally legal at the step it is
  taken** (else `step()` raises). Inversion that emits an illegal intermediate move
  is a failed inversion, not a warm-start example.
- **The replayed terminal graph must reproduce the fixture** (see §Reproduction
  fidelity for exactly *what* "reproduce" means). Otherwise the policy is imitating
  actions toward a graph the fixture never scored.

## Governing principle: reproduce-by-replay, verified by the judge

The fixture is the *target*; the environment is the *only legal builder*. Inversion
never fabricates state directly — it emits actions the environment applies. A
candidate action sequence is **accepted** iff (a) every action is legal in sequence
and (b) the materialized terminal `Round` is *equivalent to the fixture* under the
judge. This keeps inversion honest: the same `validate_round` / `judge` the RL loop
uses at termination is the acceptance oracle, so a warm-start example can never train
the policy toward a graph the judge would score differently than the fixture.

### Reproduction fidelity — RULED (A: judge-equivalent)

**Ruling (A):** "reproduce the fixture" means **judge-equivalent**, not
byte-identical:

- identical node set — same ids, roles, sides, introduction speeches, and per-speech
  liveness *keys* (which speeches each node is live in),
- identical edge *types* and *undirected* incidence (same pair, same
  support/attack/comparison type),
- identical judge verdict + trace class.

It does **not** require identical edge `source`/`target` **orientation**. The judge is
direction-agnostic (`judge_spec.md` §2.2 — edges are read as undirected; chains orient
by node *type* and clashes by *speech recency*, never by drawn direction), and the
warm-start observations come from **env-built** orientation, not the fixture's, so
byte-fidelity to authored orientation buys nothing for the verdict.

**Why this is the right target (and not a byte-fidelity one).** The encoder's
structural attention bias *is* directional (`encoder_spec.md` §Structural bias:
`supports` vs `supported_by` are distinct relations), so the orientation the
environment builds is exactly what the warm-started policy imitates. R2 (below) showed
authored orientation in the corpus is **not a reliable signal**: 43 of 48 cross-speech
support edges are unbuildable in authored orientation, and BD↔Impact splits 29/12 with
no consistent convention. Training the encoder's directional bias on authored
orientation would therefore fit **data-entry artifacts**, not real structure. Under A
the env-built orientation (§Edge inversion, ruling E) is a single, consistent,
mechanically-derived convention — the directional structure the policy learns is
principled rather than an artifact of how a fixture happened to be typed.

## The inversion problem

A fixture records only final structure: nodes (`id, role, side, speech, liveness`),
edges (`source, target, type`), and weigh pointers. It records **no action history** —
not the order nodes were introduced, not which incident edge was a node's
"introduction edge" vs. a later `connect`, not whether a liveness stamp came from an
`extend` or a `concede`, not the intra-speech move order. Inversion must recover
**any** legal history that the environment would accept and that reproduces a
judge-equivalent graph.

**Any-legal-path — RULED (B).** Inversion targets *any* legal action sequence that
reaches a judge-equivalent graph; it does **not** model or approximate realistic human
construction order (advocacy-first, construct-broad-then-narrow, etc.). A fixture's
static graph records no real construction order, so approximating a "plausible" order
would mean **inventing and hardcoding** an assumed strategic pattern — which cuts
against the schema's emergence-over-hardcoding principle: sequencing is exactly the
kind of thing self-play should discover, not a heuristic baked into the training data.
The tiebreaks in §Underdetermination therefore exist **only for determinism** (so the
same fixture yields the same data run-to-run); none of them is a plausibility model,
and none should be read as one.

What is **already determined by the fixture** (not inferred):

- **Turn/side and speech of every node.** Each node carries `side` and `speech`, and
  across the whole corpus `side == SPEECH_SIDE[speech]` holds for every node (0
  exceptions measured). So *which speech and which side introduced each node is
  given*, and AFF/NEG alternation follows `SPEECH_ORDER` mechanically. Turn order is
  **not** an inference problem for oracle fixtures. (See §Speech ownership for the one
  genuine subtlety — `connect` edges and the relative-owner observation.)
- **Which speeches a node is live in.** The `liveness` keys give this directly; each
  key after the introduction speech is one carriage action in that speech.

What must be **inferred**:

1. the **introduction order** of nodes (both across and within a speech),
2. the partition of each node's incident edges into its one **introduction edge**
   (built by `introduce`) vs. **`connect` edges** (built later),
3. the **speech in which each `connect` edge** is added, and by which side,
4. the **verb** (`extend` vs `concede`) for each post-introduction liveness stamp,
5. the **intra-speech ordering** of all of the above against the move budget.

## Inversion strategy

### 1. Node introduction: speech-bucketed topological order over the introduction forest

Bucket nodes by `speech` (order buckets by `SPEECH_ORDER`). Within a speech, all
nodes share one side and are introduced in that speech's turn.

Introduction obeys one hard constraint from the environment: **`introduce(role,
target, edge_type)` creates a new node whose single edge runs `new → target`, so the
`target` must already exist.** Therefore a node introduced via an attaching
`introduce` must have its introduction-edge target already present, and the built edge
is oriented `new → target` — which, under ruling E (§2), is simply *whatever
orientation natural forward construction produces*, never the fixture's authored one.

**Order within the round:** any topological order over the chosen *introduction-edge*
forest such that every introduction edge's target precedes its source, with speech
buckets contiguous and in `SPEECH_ORDER`. Concretely:

- **Roots** (nodes whose introduction is a fresh `NEW` node, no edge) come first
  within their speech.
- A node attached by introduction edge `new → target` is introduced after `target`.

Because there is no authored-orientation target to hit (E), this order is
**unconstrained beyond legality**: for a support edge inside one speech, either
endpoint may be the parent; for a cross-speech support edge, the earlier-speech
endpoint is necessarily the parent and the later node simply attaches onto it (edge
`later → earlier`), which is always a plain `introduce` — no reverse-order gymnastics
and no `connect` needed. The intra-speech order among independent introductions is a
determinism tiebreak (§Underdetermination), not a plausibility choice (B).

### 2. Edge inversion: introduction edges vs. `connect`, and orientation

Partition every non-comparison edge into exactly one of:

- **Introduction edge** — the single edge created when a node is introduced. Each
  non-root node has exactly one. Constraint: its target must be introduced earlier,
  and it must carry the node as `source` (`introduce` builds `new → target`).
- **`connect` edge** — every other edge (the non-forest edges: convergence
  diamonds, cross-side attacks onto already-present nodes, BD attachments added in a
  later speech, any second/third incident edge). Built by `connect(source, target,
  edge_type)` once *both* endpoints exist.

`introduce` can only grow a forest (one node + one edge); `connect` builds everything
non-forest (`action_schema_spec.md`). So the count of `connect` moves per fixture is
`#edges − (#nodes − #roots)` — the **non-forest** edges only — and lands in specific
speeches (§3).

**Orientation — RULED (E: natural-forward-with-flips).** Edge direction in the
reconstructed sequence follows **whatever direction is buildable under a legal
construction order** — never the direction authored in the fixture — and **no
`connect`-based workaround is used to force an unbuildable authored orientation.**

- **Introduction edges (the forest) are built in their natural orientation.** A node's
  one introduction edge runs `new → target`. For a support edge that means: whichever
  endpoint is introduced first is the target, and the other attaches onto it. Same
  speech → either endpoint may be the parent (a determinism tiebreak); cross-speech →
  the earlier node is the parent and the later attaches (`later → earlier`). Either
  way it is a plain `introduce`. This is what makes R2 vanish: the 43/48
  "source-earlier, unbuildable-in-authored-orientation" support edges are built as
  ordinary flipped introduction edges, and the 29/12 BD↔Impact inconsistency is moot
  because no authored orientation is targeted.
- **Attacks and comparisons happen to match authored orientation, and that is fine —
  not required.** An attack is authored attacker→target and the attacker is the
  later-speech node, so `introduce`/`connect` `source = new/acting` reproduces it
  naturally; a `weigh` authors `weighing → a`, `weighing → b`, exactly what the `weigh`
  action emits. We do not *rely* on this matching — it simply falls out of natural
  construction (E asks only for a buildable direction, and here the buildable direction
  coincides with the authored one).
- **Non-forest edges use `connect` in whatever orientation is legal**, chosen
  deterministically — again never to hit an authored direction. For a `support`
  connect, orientation is picked to **avoid closing a Support cycle** (the one legality
  constraint on connect); because E frees us from the authored orientation, we can
  always pick the cycle-safe direction, so E cannot introduce a connect-cycle
  rejection that byte-fidelity might have forced.

There is therefore **no orientation-driven `connect`** at all: `connect` is used only
for genuinely non-forest edges (a node's second-or-later incident edge), never to
force a forest edge into its authored direction.

### 3. `connect` and carriage placement across speeches

- A **`connect` edge** (non-forest edges only) is legal once both endpoints exist.
  **RULED (C): earliest-legal-speech placement.** Place each `connect` in the **first
  speech, in `SPEECH_ORDER`, by which both endpoints have been introduced and the
  connect is structurally legal and affordable**, performed by the side whose turn that
  speech is (`SPEECH_SIDE[that speech]` — equivalently, the next side to speak once both
  endpoints exist and the move is legal). The actor is not recorded in the fixture and
  does not affect the graph; C fixes it deterministically so the `(observation, action)`
  stream is reproducible. For a same-speech non-forest edge (e.g. the closing edge of a
  convergence diamond authored entirely in 1AC), that earliest legal speech is the
  endpoints' own speech, so the `connect` lands there after both introductions; its
  orientation is the cycle-safe deterministic one (E), never the authored direction.
- A **carriage action** (`extend`/`concede`) is emitted for each liveness key after a
  node's introduction speech, in that key's speech, **by the side whose speech that is**
  (`SPEECH_SIDE[key]`) — the node's own side for an own-side carriage, or the **opponent**
  for a cross-side one (a `concede`; recall `concede` is just `extend` targeting an
  opponent-owned node, `action_schema_spec.md`). Cross-side carriage is **required, not
  optional**: 8 turned-chain nodes across 5 fixtures (`F`, `r4`/`r4b`/`r4c`, `r5`) are
  AFF nodes carried by NEG at NEG speeches, so an own-side-only inverter would fail to
  reproduce them. Introduction itself stamps the introduction speech (no separate move).

### 4. Speech-budget and boundary placement

`introduce`/`weigh`/`connect` cost one slot; `end_speech` costs none;
`extend`/`concede` cost `ceil(path_length / EXTEND_COST_K)` over the root-to-impact
walk they stamp (`action_schema_spec.md` §Turn structure; the chain-level-extend +
variable-cost model that landed after this draft's first pass). **Placement:** within
a speech, emit in this order — introductions (topological over the introduction
forest), then same-speech `connect`s, then `weigh`s, then carriage
(`extend`/`concede`) — then `end_speech`. The intra-speech order among independent
moves is a determinism tiebreak (§Underdetermination), not a plausibility choice (B).

The per-speech cost an inversion needs is `introduces + connects + weighs +
carriage-cost` in that speech. Under the chain-level extend model this is feasible
corpus-wide (see the revised R1) — a whole spine is carried by one `ceil(len/K)`-priced
extend rather than one move per node, which was what previously overflowed the rebuttal
budgets.

## Underdetermination and determinism tiebreaks

Where multiple legal sequences reproduce a judge-equivalent graph, inversion must be
deterministic (so warm-start data is reproducible run-to-run). Under ruling B these
rules are **arbitrary-but-fixed for determinism only** — none is a plausibility model.

1. **Which incident edge is a node's introduction edge, and which node roots a
   component.** *Rule (Advocacy/Framework-seeded — supersedes the former
   lowest-`(speech_index, id)` `NEW` root):* each connected component is rooted at an
   **Advocacy** (or, where a component has no Advocacy, a **Framework**) — its
   first-introduced node is that Advocacy/Framework as a `NEW` root; a component with
   several Advocacies may float each, since every Advocacy is a legal `NEW` root under
   the floating-root restriction. Every other node attaches to an already-introduced
   neighbour, its introduction edge being the incident edge to the earliest-introduced
   such neighbour. Ordering within a component follows the unlock ladder's **type
   precedence** (advocacy → link → {uniqueness, impact} → {framework,
   ballot_directive}; rl_training_spec §Opening curriculum), tie-broken by ascending
   `id`. This replaces the earlier rule that rooted a component at its
   lowest-`(speech_index, id)` node and floated whatever landed there — which under
   the **floating-root restriction** (action_schema_spec §introduce → Floating-root
   restriction) would emit illegal non-root `NEW` roots. Sink nodes (a
   BallotDirective, a terminal Impact) still attach as edge **sources**, storing the
   flipped orientation the judge scores identically (E). A component with neither an
   Advocacy nor a Framework has no legal root and is not convertible (see corpus note
   below). (E unchanged: orientation still follows whatever is buildable; the
   Advocacy/Framework-seeded spanning forest is now the specific buildable one.)
2. **Intra-speech order among independent introductions/connects/weighs.** *Rule:*
   ascending `id`, after the topological constraint. Purely for determinism.
3. **`extend` vs `concede` verb** for a post-introduction liveness stamp. The two are
   the same action structurally; the label is a deterministic function of **target
   ownership** (D): `concede` when the carriage side ≠ the node's owner (a cross-side
   carry — recall `concede` *is* `extend` targeting an opponent's node), `extend`
   otherwise. This follows the labeling convention in `action_schema_spec.md`
   directly. *Ruled — see D.*
4. **Which side/speech performs a `connect`** (for the non-forest edges) whose
   endpoints exist across multiple speeches. *Rule (C, §3):* the earliest speech where
   both endpoints exist and the connect is legal, performed by that speech's side.
5. **Orientation of a non-forest `connect`.** *Rule:* pick the deterministic
   cycle-safe orientation (E: never the authored one; for `support`, whichever
   direction does not close a Support cycle). No `connect` is ever used to force a
   forest edge's authored orientation.

**Corpus consequence — `E` excluded, 43 → 42 convertible.** Exactly one fixture,
`E`, is a single component with neither an Advocacy nor a Framework (a NEG-only disad
chain). Under the floating-root restriction it has no legal root, is **not
convertible**, and is dropped from the warm-start corpus: the previously-43
convertible set becomes **42**. `E` remains a valid **oracle** fixture with an
unchanged verdict — masks live in the environment and the converter, never in the
judge (rl_training_spec §Imitation warm-start). This is distinct from the
budget-feasibility count in ruling F: `E`'s exclusion is a legality-of-construction
matter, not a budget one.

## Reachability gaps (candidate real bugs, not to be papered over)

These are fixtures (or fixture properties) that **no legal action sequence can
reproduce under current rules**. Surfaced, not worked around.

### R1. Per-speech budget overflow — RESOLVED by chain-level extend (not by A/B/E)

**Original finding (per-node extend model):** on an `introduce + extend` lower bound,
24 of 45 v2 fixtures needed more moves in some speech than its budget allowed, the
binding constraint being the rebuttal budgets (1AR/2NR/2AR = 5) — keeping a multi-node
spine live through 2AR cost 6–8 moves at one-move-per-node (e.g. `r33`/`r34`/`r35`
8/5 at 1AR & 2AR; `r32` 7/5 & 8/5).

**Now resolved** — but by the **chain-level extend + variable-cost** ruling that landed
after this draft's first pass, *not* by A/B/E. Under that model a whole spine is
carried by a single `ceil(path_length / K)`-priced `extend` instead of one move per
node, so the rebuttal load collapses and **all 45 v2 fixtures fit every per-speech
budget** (re-verified: 0/45 over budget, including `connect` moves greedily placed).
A/B/E leave this unchanged — they touch orientation, path, and fidelity, not extension
cost — but they do not re-open it either. **Open question F is therefore closed on
budget grounds for the v2 corpus** (the former residual, the v1 `NSDA24Finals` round
that overflowed on introductions alone, has since been deleted from the corpus — see
R3 / G).

### R2. Authored support-edge orientation — RESOLVED by E

**Original concern:** 43 of 48 cross-speech support edges orient `source`-earlier (not
`introduce`-buildable in authored orientation), and BD↔Impact support edges split 29
`impact→bd` vs 12 `bd→impact` — no single authored convention, so "preserve authored
orientation" was neither always feasible nor even well-defined across the corpus.

**Resolved by ruling E (natural-forward-with-flips).** Because reconstruction no longer
targets authored orientation, every such support edge is built as an ordinary flipped
introduction edge (the later node attaches onto the earlier one; §Edge inversion), and
the BD↔Impact inconsistency is moot. No `connect`-based workaround is used to force an
authored direction, and the encoder's directional bias trains on the single consistent
env-built convention rather than on the corpus's data-entry artifacts (§Reproduction
fidelity). This class of edge is now fully reproducible; R2 is **not** a reachability
blocker.

### R3. `NSDA24Finals` — RESOLVED (fixture deleted)

The single v1 fixture (`ExtensionEdge` encoding) was over budget on introductions alone
(≈84 nodes; 10/8 at 1AC, 18/13 at the block, …), making it the least reproducible round
in the corpus. It has since been **deleted from the fixture set** (its dependent tests
were retired or repointed to v2 rounds), so it is no longer a warm-start candidate and
the reachability concern is moot. Open question G is thereby closed.

### R4. Non-issues confirmed (for the record)

Measured clean across the corpus, so **not** reachability blockers: side/speech
consistency (0 mismatches), directed support cycles (0 — so the `connect` cycle rule
never blocks a fixture), and weighing well-formedness (every weighing has exactly two
comparison edges, so `weigh` inverts exactly).

## Speech ownership and turn order

The concern that "the graph does not record which side introduced what" does **not**
apply to oracle fixtures: every node carries absolute `side` **and** `speech`, and
`side == SPEECH_SIDE[speech]` universally, so the introducing side and speech are
read directly, and AFF/NEG alternation is mechanical. The relative-owner /
presumption encoding is a property of the **observation the policy sees**
(`encoder_spec.md`), not of the fixture — inversion works from the fixture's absolute
fields and lets the observation layer relativize as usual.

The genuine ownership subtleties are narrower:

- **`connect` has no recorded actor.** An edge added by `connect` is side-less; the
  environment attributes the *move* to whoever is speaking. So the *speech* a
  `connect` lands in is a real choice — it consumes that side's budget and appears in
  that side's observation stream, which is what the policy imitates. It does not change
  the graph, only the `(observation, action)` pairs, and is fixed by ruling C
  (earliest-legal-speech, that speech's side; §3).
- **Cross-side introduction targets are fine.** A NEG node introduced at 1NC
  attacking an AFF 1AC node targets an already-existing node — legal by construction,
  no ambiguity.
- **A `connect` between two same-side nodes introduced in different speeches** may be
  performed in any later speech where both endpoints exist. Which speech/side spends
  the move is a legality-neutral choice — not a plausibility one (B rules out modeling
  "the opponent has no reason to wire your spine" as a heuristic) — so it is settled by
  ruling C: the earliest legal speech and that speech's side.

## Resolved rulings

All eight original open questions (A–H) are resolved. In summary:

- **A — Fidelity: judge-equivalent, not byte-identical.** A reconstructed sequence must
  reach a graph the judge scores identically (same verdict, same liveness/path
  structure), not one that reproduces the fixture's edges byte-for-byte
  (§Reproduction fidelity).
- **B — Any-legal-path, no plausibility model.** Any legal sequence reaching a
  judge-equivalent graph is acceptable warm-start data; inversion does not model or
  approximate human construction order, and the §Underdetermination tiebreaks exist
  only for determinism (§The inversion problem). *(Amended:* tiebreak 1 now roots
  each component at its Advocacy — or a Framework where a component has no Advocacy —
  under the floating-root restriction, superseding the former lowest-`(speech_index,
  id)` `NEW` root; see §Underdetermination tiebreak 1 and its corpus note. `E` is
  thereby non-convertible — 43 → 42.)*
- **C — `connect` placement: earliest-legal-speech, that speech's side.** Each
  non-forest edge's `connect` is placed in the first speech (in `SPEECH_ORDER`) by which
  both endpoints exist and the move is legal and affordable, performed by that speech's
  side (`SPEECH_SIDE`). Deterministic; affects the observation stream, not the graph
  (§3, tiebreak 4).
- **D — `extend`/`concede` verb label: by target ownership.** The verb is a pure label
  (the two are one action; §concede in `action_schema_spec.md`), fixed deterministically:
  `concede` when the carriage side ≠ the node's owner (cross-side carry), else `extend`.
  This follows the schema's own labeling convention; it supersedes this question's
  original "always `extend` / mirror the `contested`/`conceded` status" framing, which
  predated the concede ruling and conflated the per-speech *status tag* with the *verb*
  (§Underdetermination tiebreak 3).
- **E — Edge orientation: natural-forward-with-flips.** Orientation follows whatever is
  buildable under a legal construction order, never the fixture's authored direction,
  and `connect` is never used to force an unbuildable authored orientation (§Edge
  inversion). Rationale for A/E: R2 showed authored direction encodes no reliable
  signal (43/48 cross-speech support edges unbuildable in authored orientation;
  BD↔Impact 29/12), so treating it as ground truth would train the encoder's
  directional bias on data-entry artifacts. Rationale for B: a static graph records no
  real construction order, so a "plausible" order would be invented/hardcoded strategy,
  against emergence-over-hardcoding.
- **F — Budget reconciliation: closed by chain-level extend.** All 45 v2 fixtures fit
  every per-speech budget under the `ceil(path_length / K)` extend-cost model (0/45 over
  budget, connect moves included); no budget retune or corpus trim is needed (§R1).
- **G — `NSDA24Finals` scope: closed by deleting the fixture.** The sole v1 round was
  over budget on introductions alone and irreproducible; it has been **deleted** from the
  corpus (its dependent tests retired or repointed to v2 rounds), so it is no longer a
  warm-start candidate (§R3).
- **H — hand-authored 1AC: `8node1ac.json` (pending import).** The artifact
  `rl_training_spec.md` refers to exists and is already builder-loadable v2 (no
  conversion), a single-speech 1AC opening (8 AFF nodes, one support tree), and
  **feasible** under the extend-cost model — 0 `connect` moves, 8 introduces = the 1AC
  budget, 0 extends. Importing it (proposed: a dedicated `tests/warmstart/` rather than
  the oracle-verdict corpus) is a separate, later task.

## Readiness

No open questions remain. The strategy above — judge-equivalent reproduce-by-replay
(A), any-legal-path with determinism tiebreaks (B), forest-`introduce` /
non-forest-`connect` edge partition, natural-forward-with-flips orientation (E),
earliest-legal-speech `connect` placement (C), chain-level `ceil(path_length / K)`
carriage priced per speech (F), and ownership-labeled `extend`/`concede` (D) — is
complete and internally consistent, and every reachability concern (R1–R4) is resolved
or confirmed a non-issue. Conversion code can be written against this document.

The only outstanding item is a **task, not a question**: importing `8node1ac.json` as
the warm-start 1AC fixture (H), a separate later step that does not block writing the
core inversion.
