# CDAF Warm-Start Data Spec (Phase 5, draft)

> **STATUS: DRAFT / PROPOSAL.** This document proposes how to turn the static
> fixture corpus into imitation-learning `(observation, action)` pairs. Several
> points touch debate semantics or pinned environment parameters and are flagged
> as **OPEN QUESTIONS** for a ruling; they are not decided here. Nothing in this
> spec is implemented yet, and no conversion code should be written until the open
> questions are resolved.

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

### Reproduction fidelity — PROPOSAL (open question A)

**Proposed:** "reproduce the fixture" means **judge-equivalent**, not
byte-identical:

- identical node set — same ids, roles, sides, introduction speeches, and per-speech
  liveness *keys* (which speeches each node is live in),
- identical edge *types* and *undirected* incidence (same pair, same
  support/attack/comparison type),
- identical judge verdict + trace class.

It does **not** require identical edge `source`/`target` **orientation**. Rationale:
the judge is direction-agnostic (`judge_spec.md` §2.2 — edges are read as undirected;
chains orient by node *type* and clashes by *speech recency*, never by drawn
direction), and the warm-start observations come from **env-built** orientation, not
the fixture's, so byte-fidelity to authored orientation buys nothing for the verdict.

**Why this is an open question, not a free call.** The encoder's structural attention
bias *is* directional (`encoder_spec.md` §Structural bias: `supports` vs
`supported_by` are distinct relations). So the orientation the environment happens to
build is exactly what the warm-started policy imitates. Choosing judge-equivalence
means we must also fix a **single, consistent orientation policy** for env
construction (proposed in §Edge inversion), and the user should ratify that the
policy learns *that* directional structure rather than the fixture's authored one —
which, as §Reachability shows, is itself inconsistent across the corpus and often not
introduce-reproducible in its authored direction anyway.

## The inversion problem

A fixture records only final structure: nodes (`id, role, side, speech, liveness`),
edges (`source, target, type`), and weigh pointers. It records **no action history** —
not the order nodes were introduced, not which incident edge was a node's
"introduction edge" vs. a later `connect`, not whether a liveness stamp came from an
`extend` or a `concede`, not the intra-speech move order. Inversion must recover a
*plausible legal* history that the environment would accept and that reproduces the
graph.

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

## Inversion strategy (proposal)

### 1. Node introduction: speech-bucketed reverse-spine topological order

Bucket nodes by `speech` (order buckets by `SPEECH_ORDER`). Within a speech, all
nodes share one side and are introduced in that speech's turn.

Introduction obeys one hard constraint from the environment: **`introduce(role,
target, edge_type)` creates a new node whose single edge runs `new → target`, so the
`target` must already exist.** Therefore a node introduced via an attaching
`introduce` must have its introduction-edge target already present.

**Proposed order within the whole round:** a topological order over the chosen
*introduction-edge* forest such that every introduction edge's target precedes its
source, subject to speech buckets being contiguous and in `SPEECH_ORDER`. Concretely:

- **Roots** (nodes whose introduction is a fresh `NEW` node, no edge) come first
  within their speech.
- A node attached by introduction edge `X → T` is introduced after `T`.

Because the authored support spine runs premise→conclusion (`advocacy → … → impact →
BD`) but `introduce` forces `new = source`, reproducing an authored support edge
`A → B` in-orientation requires introducing `B` **before** `A` and attaching `A` onto
`B`. Within a single speech this is free (see §2 for the orientation choice and its
tension with construction plausibility). Across speeches it is often impossible (§4).

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
at least `#edges − (#nodes − #roots)` and lands in specific speeches (§3).

**Orientation — PROPOSAL (ties into open question A/E).** Two families:

- *Attack and comparison edges invert cleanly in authored orientation.* An attack is
  authored attacker→target, and the attacker is the later-speech node, matching
  `introduce`/`connect` `source = new/acting`. A `weigh` authors `weighing → a`,
  `weighing → b`, matching the `weigh` action exactly. (Measured: every weighing has
  exactly two comparison edges from it; attacks orient attacker-as-source.)
- *Support edges frequently cannot be reproduced in authored orientation.* 43 of 48
  cross-speech support edges have `source` in an **earlier** speech than `target`
  (e.g. `impact(1AC) → bd(2AR)`); `introduce`'s `new = source` would require the
  earlier node to attach onto a not-yet-existent later node. These must be built by
  **`connect` at the later speech**, and even then `connect(source, target)` can set
  orientation freely — so authored orientation *is* reproducible for cross-speech
  support **via `connect`**, at the cost of a move in the later speech.

  For **same-speech** support edges, orientation is reproducible by `introduce` only
  in reverse-spine order (introduce the impact end first). This collides with
  construction plausibility (a debater states the advocacy first) — see open
  question E. Under the judge-equivalence proposal (A), same-speech support edges may
  instead be built in natural forward order with **flipped** orientation, which the
  judge scores identically.

**Proposed default orientation policy:** preserve authored orientation for attacks
and comparisons (always feasible); for support edges, prefer the `introduce`
introduction-edge in whatever orientation natural forward construction produces, and
use `connect` (authored orientation) only where an edge is non-forest or cross-speech.
This is the *simplest legal* policy; whether the corpus's authored orientation should
instead be preserved everywhere (forcing reverse-order construction and extra
`connect`s) is open question E.

### 3. `connect` and carriage placement across speeches

- A **`connect` edge** is legal only once both endpoints exist. **Proposed:** place
  each `connect` in the **earliest speech in which both endpoints exist**, performed
  by whichever side speaks in that speech (the actor is not recorded and does not
  affect the edge — open question C on whether the *owner* side should be preferred).
  For a same-speech non-forest edge (e.g. the closing edge of a convergence diamond
  authored entirely in 1AC), the `connect` is placed in that speech, after both
  endpoints' introductions.
- A **carriage action** (`extend`/`concede`) is emitted for each liveness key after a
  node's introduction speech, in that key's speech, by the node's owning side.
  Introduction itself stamps the introduction speech (no separate move).

### 4. Speech-budget and boundary placement

Each emitted move (`introduce`, `connect`, `weigh`, `extend`, `concede`) costs one
move against the acting speech's budget (`SPEECH_BUDGET`; 1AC/1NC/2AC = 8, block = 13,
rebuttals = 5). `end_speech` closes a turn. **Proposed placement:** within a speech,
emit in this order — introductions (reverse-spine topological), then same-speech
`connect`s, then `weigh`s, then carriage (`extend`/`concede`) — then `end_speech`. The
intra-speech order among independent moves is a tiebreak (§Underdetermination).

**This is where inversion collides with reality (see §Reachability).** The per-speech
move total an inversion needs is `introduces + connects + weighs + carriages` in that
speech, and for 24 of 45 v2 fixtures this exceeds the speech's budget — the fixtures
were authored as judge oracles, with no budget in view.

## Underdetermination and proposed tiebreaks

Where multiple legal sequences reproduce the same graph, inversion must be
deterministic (so warm-start data is reproducible run-to-run). Each rule below is a
**PROPOSAL**, not a decision.

1. **Which incident edge is a node's introduction edge** (when a node has several
   incident edges eligible). *Proposed:* prefer the edge to the target with the
   earliest `(speech_index, id)`; if the node has no eligible `new → existing` edge
   in authored orientation, introduce it as a `NEW` root and build all its incident
   edges by `connect`. *Tiebreak flagged — interacts with orientation (E).*
2. **Intra-speech order among independent introductions/connects/weighs.** *Proposed:*
   ascending `id`, after the topological constraint. Purely for determinism.
3. **`extend` vs `concede` verb** for a post-introduction liveness stamp. The two are
   structurally identical (status is derived at materialization, not from the verb —
   `environment_shell_spec.md` §Liveness stamping). *Proposed:* always emit `extend`
   (or mirror the materialized status: `concede` when that speech's derived status is
   `conceded`, else `extend`). *Flagged — this is an action-label the policy imitates,
   so it is a soft debate-semantics choice, open question D.*
4. **Which side/speech performs a `connect`** whose endpoints exist across multiple
   speeches. *Proposed:* earliest-legal speech, acting side of that speech
   (§3). *Flagged — open question C.*
5. **Root selection when a component has no authored `new → existing` orientation**
   (e.g. a support edge whose only feasible orientation is reverse). *Proposed:*
   the earliest-`(speech, id)` node becomes a `NEW` root; downstream edges become
   `connect`. *Flagged — interacts with E.*

## Reachability gaps (candidate real bugs, not to be papered over)

These are fixtures (or fixture properties) that **no legal action sequence can
reproduce under current rules**. Surfaced, not worked around.

### R1. Per-speech budget overflow — 24 of 45 v2 fixtures

On an `introduce + extend` lower bound alone (ignoring `connect` moves, which only add
more), 24 of 45 v2 fixtures need more moves in some speech than its budget allows.
The binding constraint is the **rebuttal budgets** (1AR/2NR/2AR = 5): a fixture that
keeps a multi-node spine live through 2AR must re-extend every spine node there, which
alone can need 6–8 moves. Examples (needed vs. budget):

| Fixture | Over-budget speeches (needed/budget) |
|---|---|
| `fw_weigh_lockout` | 1AR 6/5, 2NR 7/5, 2AR 7/5 |
| `r33`, `r34`, `r35` | 1AR 8/5, 2AR 8/5 |
| `r29` | 1AR 8/5, 2AR 8/5 |
| `r32` | 1AR 7/5, 2AR 8/5 |
| `AFFturnOutweighed` | 2NR 7/5 |
| … (24 total) | mostly rebuttal-speech extension load |

The **total** move count is fine (max ≈ 43 < 52 ceiling); it is the **per-speech
distribution**, specifically rebuttals, that is infeasible. The corpus and the
first-iteration budgets were designed independently — the fixtures as judge oracles
(no env), the budgets as `SPEECH_BUDGET` defaults ("tunable", per both specs). Warm-
start forces a reconciliation. **This is open question F** (retune budgets? a separate
warm-start replay budget? drop/trim infeasible fixtures? allow a node to skip re-
extension where the verdict is invariant?). It should not be silently resolved by, e.g.,
quietly raising budgets or dropping the offending extensions, because either choice
changes what behavior the warm-started policy imitates.

### R2. Authored support-edge orientation is not introduce-reproducible and is corpus-inconsistent

43 of 48 cross-speech support edges orient `source`-earlier (not `introduce`-buildable
in authored orientation), and BD↔Impact support edges split 29 `impact→bd` vs 12
`bd→impact` across the corpus — there is **no single authored convention**. Under
judge-equivalence (A) this is a non-issue (orientation is free and the judge is
direction-agnostic); under a byte-fidelity requirement it forces `connect`-heavy
reconstruction and aggravates R1. Flagged because the choice between A and byte-
fidelity is a real fork, and because the *inconsistency itself* means "preserve
authored orientation" is not even a well-defined target across the corpus.

### R3. `NSDA24Finals` is v1 and vastly over budget

The single v1 fixture (`ExtensionEdge` encoding) needs `model.convert` first, and once
converted it exceeds **every** speech's budget (10/8 at 1AC, 18/13 at the block,
13/5 at 1AR, …). As the only genuine full-length real round in the corpus it is the
most valuable demonstration *and* the least reproducible. **Open question G:** is
`NSDA24Finals` in-scope for warm-start at all, and if so under what budget?

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
  `connect` lands in is a real choice (§3, tiebreak C) — it consumes that side's
  budget and appears in that side's observation stream, which is what the policy
  imitates. It does not change the graph, but it changes the `(observation, action)`
  pairs. Flagged (C).
- **Cross-side introduction targets are fine.** A NEG node introduced at 1NC
  attacking an AFF 1AC node targets an already-existing node — legal by construction,
  no ambiguity.
- **A `connect` between two same-side nodes introduced in different speeches** may be
  performed in either side's later speech; only the owning side's speeches are natural
  (the opponent has no reason to spend a move wiring your spine), but "natural" is a
  plausibility judgment, not a legality one → tiebreak C / open question E.

## Open questions requiring a ruling (before finalize + implement)

- **A. Reproduction fidelity.** Judge-equivalent (proposed) or byte-identical edge
  orientation? Determines whether §Edge inversion's orientation freedom is available.
- **B. Any-legal-path vs. plausible construction.** Should inferred sequences just be
  *some* deterministic legal path (proposed default), or should they mimic real
  in-round construction (advocacy-first, forward spine order, construct-broad-then-
  narrow)? This is a debate-semantics call and it trades directly against A/E: forward
  plausible construction implies flipped support orientation (needs A), while authored-
  orientation fidelity implies implausible reverse-order construction.
- **C. `connect` actor/speech placement.** Earliest-legal speech + that side (proposed),
  owner-side-preferred, or something else? Affects which observations enter the data.
- **D. `extend` vs `concede` verb.** Always `extend`, or mirror the derived status?
  The policy imitates this label even though the graph is verb-invariant.
- **E. Support-edge orientation policy** (the concrete form of A/B): natural-forward
  with flips, vs. reverse-order/`connect` to preserve authored orientation.
- **F. Budget reconciliation for R1** (the headline blocker): retune `SPEECH_BUDGET`
  (esp. rebuttals), grant a separate warm-start replay budget, drop/trim the 24
  over-budget fixtures, or relax mandatory re-extension where the verdict is invariant?
  Each answer changes what the warm-started policy learns.
- **G. `NSDA24Finals` (and any future full round) scope.** In or out of warm-start,
  and under what budget, given R3?
- **H. The "hand-authored 1AC."** `rl_training_spec.md` names it as a demonstration
  source, but no dedicated 1AC artifact exists in the repo (no `*1ac*` fixture, no
  builder default found). Does it need authoring as a separate deliverable, or does
  "the fixture corpus" subsume it?
```
