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
- an **attack edge** (`defensive_attack`/`offensive_attack`, whether via `introduce`
  or `connect`) may not run between two nodes of the **same side** — a same-side
  attack is incoherent (it can never enter any attacker set; matches the judge's own
  `same-side attack (incoherent)` classification)
- an **`offensive_attack`** may not touch a **non-polarity** node — both endpoints
  must be offense-bearing (Link or Impact); offense at a uniqueness/advocacy/
  framework/ballot_directive has no polarity to flip (judge §3.4)
- a **`connect`** may not **duplicate an existing edge** (same source, target, and
  edge_type) — it would add no structure
- a **`weigh`** may not compare **different-kind** operands (Ruling 1) — a cross-kind
  weigh has no matching factor and the judge scores it inert in every state (judge_spec
  §3.4/§6.5); rejected at creation. (Same-kind BD-vs-BD is descriptive/inert;
  Uniqueness-vs-Uniqueness is consumed only when the two are competing — action_schema_spec
  §weigh.)
- a **cross-side `defensive_attack` whose target is a `BallotDirective`** is illegal
  (Ruling 2 V1a) — a BD bears no magnitude any pass consumes, so the edge can never enter
  scoring; judge-invisible in every state.
- a node may not be introduced as a **floating root** unless its role is
  **`advocacy`** or **`framework`**: an `introduce` with `target_id = NEW` is legal
  only for `role ∈ {advocacy, framework}`; every other role must attach to an
  existing node at creation. Equivalently, **every connected component contains an
  Advocacy or a Framework** — the two kinds the judge already treats as chain roots
  (judge_spec §2, rule 4). (Rationale: action_schema_spec §introduce →
  *Floating-root restriction*.)

It does **not** enforce strategic legality. Response-window compliance, whether an
extension will ultimately count, whether a new chain in a rebuttal can establish
offense, whether a spike into a conceded-but-uncontested node is inert — all of
these remain outcomes computed by the judge, not prohibitions enforced at action
time. The three structural checks above are **not** strategic: they forbid moves
that are *logically incoherent in every round state* (no possible continuation makes
a same-side attack, an offense at a non-polarity node, or a duplicate edge
meaningful), so disallowing them removes no strategic distinction. Context-dependent
inertness stays learnable: a **no-op re-extend** is legal (priced via cost, §Liveness
stamping), and window/maker-lapsed inertness stays a judge outcome.

**The Advocacy/Framework-root rule is redundancy elimination for rooted
components, and a bounded foreclosure for rootless ones.** Unlike the
same-side-attack ban (incoherent in *every* state), a floating non-root node
(neither Advocacy nor Framework) is NOT inert in every continuation: `connect` has
no same-component precondition (State schema → `connect`), so a node floated early
can be bridged into the main chain later and goes live once connected. The
justification is therefore construction **redundancy**, not inertness. For any
component that already has a legal root (an Advocacy or a Framework), building a
further node by float-then-`connect` costs two actions (`introduce` = 1, `connect`
= 1) to reach a graph the judge scores **identically** to the one an attaching
`introduce` reaches in one action — the judge is direction-agnostic, so the attach
orientation is immaterial — so removing the floating route deletes no reachable
terminal graph and no strategy for these components; it removes a strictly-costlier
redundant path to a judge-equivalent graph, the same category as an intra-speech
move-ordering restriction. `connect`'s general component-bridging capability is
**unchanged**: the rule fires at `introduce` time, so a rootless component is simply
never created for `connect` to bridge. For a component with neither an Advocacy nor
a Framework there is nothing legal to root it, so the floating route is the only
route and the rule removes the graph outright — a deliberate foreclosure
(§Governing principle: deliberate action-space foreclosure, and its list). Within
the oracle corpus the only reachable graph so removed is fixture `E`.

**The opening curriculum is not part of this layer.** The Phase-5 opening-unlock
curriculum (rl_training_spec §Opening curriculum) is a *policy-layer* mask applied
on top of this generator during the 1AC — a training scaffold, not a legality rule.
The legal-action generator here is unchanged by it and continues to enforce
structural legality only: the curriculum can only further restrict sampling, never
widen legality, and never touches the judge.

**There is no multi-terminal refusal.** Divergent chains are first-class as of
judge v11 — a same-side Support component may have several terminal impacts,
each scored as its own chain (State schema → Divergence) — so the former
"Fence A" is retired from both the generator and `validate_round`. That also
removed the generator's per-action deepcopy probe, so legality checks are now
near-O(1) structural predicates; the non-local ones are the `connect` Support-cycle
test (a cheap reachability query) and the redundant-`connect` duplicate-edge scan (a
cheap O(edges) check). Same-side-attack, offense-at-non-polarity, and
no-op-re-extend detection are all O(1) local predicates. `validate_round` at termination
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

## Governing principle: deliberate action-space foreclosure

CDAF deliberately forecloses regions of the action space in order to define the
space within which real strategy develops. Where a class of moves can only ever
produce a zero-scoring or judge-redundant graph, the environment may remove it at
legality time rather than leaving the policy to discover its uselessness through
exploration. Exploration budget spent on such moves teaches the policy nothing, and
the narrowed space is treated as the game CDAF actually models.

This is a claim about **exploration value, not judgeability.** A foreclosed move is
not "unjudgeable" — a floating disad, for instance, IS judged and collapses with
`collapse_reason = "unrooted_disad"`. The point is that discovering its uselessness
by sampling it costs exploration budget and returns no usable signal, so the
environment removes the discovery cost by removing the move.

**Authorises the one Preserved-learning-signal softening.** The structural-legality
section's Preserved-learning-signal principle (an agent structurally prevented from
a move learns nothing from it) is softened in exactly one place: the
**strictly-disconnected `unrooted_disad`** case — a NEG disad built as its own
component with neither an Advocacy nor a Framework — is now unreachable at action
time rather than scored 0 at termination. Because such a disad can only ever score
0, the exploration spent reaching it is wasted, and this principle is what
authorises removing it. Ruling 1 as a whole needs no such override: for every
rooted component it is free on redundancy grounds (action_schema_spec §introduce →
Floating-root restriction), removing only a costlier construction route to a
judge-equivalent graph.

**Scoping consequence — equilibria are relative to the foreclosed game.** A Nash
equilibrium is defined relative to an action space, so any equilibrium reached
under self-play is an equilibrium of the *foreclosed* game, not of unrestricted
policy debate. Every foreclosure must therefore be recorded in the running list
below, so equilibrium claims can be scoped correctly.

**Foreclosure list.**

| # | Foreclosure | Enforcement layer | Removes reachable terminal graphs? |
|---|---|---|---|
| i | Floating-root restriction (Ruling 1): only Advocacy or Framework may root a component | Legality mask (`check_legality`); permanent; both sides, all speeches | **Yes** — fixture `E` alone (the sole component with neither an Advocacy nor a Framework). Advocacy/Framework-rooted components lose only redundant construction routes. |
| ii | AFF-1AC opening curriculum (Ruling 2): progressive unlock ladder | Curriculum mask (policy `type_mask`); non-permanent — 1AC only, never NEG, off from 1NC | **No.** Every terminal graph stays reachable; the ladder constrains only opening *construction order*, which the judge is invariant to |

New entries are appended here as future foreclosures are ruled.

## Judge-invisible edges (Ruling 2)

Cross-side edges that touch a `BallotDirective` or a weigh tower split into one
legality-masked class and three that the judge currently reads as nothing. **Layer
split:**

- **V1a — cross-side defensive attack on a `BallotDirective` → LEGALITY MASK.**
  Enforced in `check_legality` (a BD bears no magnitude any pass consumes, so the
  edge is incoherent in every state). Both sides, all speeches.
- **V1b — cross-side support-on-BD not on a consuming walk → SILENTLY UNREAD.**
- **V2 — cross-side support off a consuming walk → SILENTLY UNREAD.**
- **V3 — comparison edges on a weigh tower not wired to a BD → SILENTLY UNREAD.**

**"Silently unread" is precise and load-bearing.** V1b/V2/V3 are NOT a "judge-time
collapse" in the sense of an emitted `collapse_reason`: the judge never reads these
edges, so there is **no trace entry, no collapse reason, and no diagnostic signal** —
the offense routed only through them simply fails to anchor/rank and contributes
nothing, indistinguishably from an edge that was never built. A future session must
NOT go looking for a collapse path for these; **none exists by design.** Emitting a
named reason (e.g. `judge_invisible_edge`) is a deliberate follow-up, filed in
judge_spec §Open questions — **not a gap.**

**TRAP — do NOT mask support-on-BD generally.** Same-side `Link→BD` and `Framework→BD`
anchor, and cross-side `Impact→BD` anchors via capture. A blanket mask breaks `r6` and
the `T1`/`T1b` capture fixtures. (Verified: `r6` relies on cross-side `Framework→BD`,
`fw(NEG)→bd(AFF)`; `T1`/`T1b` anchor a captured cross-side `Impact→BD` through
`anchor_members`, gated on live capture `eff_pol[target] == -1`.) The V1a mask is
therefore narrow by construction — a cross-side *defensive_attack* whose target is a
BD — and touches none of the anchoring routes above.

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

**Atomic single-node stamping, speech-wide batched cost.** `extend`/`concede` stamp
**only the named node** for the current speech — no path-walking, no propagation to
its chain. This makes partial-chain carriage expressible: an agent extends a link
while deliberately not extending its impact, letting the impact die. The cost is the
**marginal of a speech-wide `ceil(count / K)` batch**: the environment keeps a
counter `extends_this_speech` of carriages taken so far this speech (reset at each
speech boundary), and the next carriage costs
`ceil((count + 1) / K) − ceil(count / K)` slots — 1 on the 1st, (K+1)-th, (2K+1)-th
… carriage, 0 otherwise, so N carriages cost `ceil(N / K)` total. `K` is the named
constant beside `SPEECH_BUDGET` (`EXTEND_COST_K`, default 4). The `1-slot-per-K`
discount is scoped to the whole speech, not to any chain — the same rate whether the
K carriages are on one chain or scattered across unrelated arguments. `contested`/
`conceded` status is still derived structurally at materialization (below), never
from the verb or the named node.

**No-op re-extend costs a full slot.** A carriage of a node **already carried this
speech** changes nothing (the stamp is an idempotent set-add). It stays legal but is
charged a **full slot (cost 1)**, bypassing the `ceil(count/K)` batch discount, and
still increments `extends_this_speech`. Only **distinct** carriages — a node not yet
carried this speech — earn the `1-slot-per-K` rate. Detection reuses the same
"already carried this speech" predicate the inert classifier uses. This prices inert
re-extension at its true opportunity cost (a wasted move) instead of letting the
batch discount make it near-free — the v2 diagnostic showed the near-free case
producing tens of no-op re-extends per episode.

**Slack-budget caveat (intentional, not a gap).** A no-op re-extend pays the full
slot even in a speech with unused budget, exactly as any move does — there is no
rebate for "there was budget to spare." The one case this does not strongly deter —
a no-op late in a speech the agent would otherwise end with slack — is deliberately
left to the dormant `inert_penalty_coef` reward backstop (rl_training_spec §Reward),
to be enabled only on evidence it matters, rather than complicating the cost model.

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
- **Unaffordable actions are illegal.** Affordability is a second gate after
  structural validity: an action whose *cost* exceeds the speech's remaining
  budget is illegal — the same category as any budget-exhausted action, now
  measured against the action's cost rather than a flat 1. `end_speech` costs 0;
  `introduce`/`weigh`/`connect` cost 1; a *distinct* `extend`/`concede` carriage costs
  the marginal of the speech-wide `ceil(count / K)` batch (0 on most carriages, 1 on
  the one that starts a new K-group), so such a carriage is unaffordable only when it
  would start a new K-group with no budget left — a simple lookup on
  `extends_this_speech`, no lookahead. **Exception:** a **no-op re-extend** (a
  carriage of a node already carried this speech) costs a **full slot (1)** regardless
  of batch position, so it is unaffordable whenever no slot remains; only *distinct*
  carriages get the batch rate. `moves_used` advances by the action's cost, and the
  turn auto-advances when the budget is reached.
- **Non-terminal steps**: the env's raw `reward = 0`, `done = False` — the environment is
  learner-agnostic and does not itself form the shaped reward. It now **exposes the scalar
  potential Φ(s) in `info` each step** (see the PBRS bullet), from which the training loop
  forms the per-step shaping reward. Inert-action *detection* (now `noop_reextend` only)
  still runs per-step for the diagnostic counters and to charge the no-op its full-slot
  cost. (Phase 1 presumption-disabling shaping still applies to the *judge configuration*
  at termination, not to intermediate steps.)
- **Potential-based shaping (PBRS) — the shaping term** (replaces the retired flat
  chain-extension bonus). The environment exposes a scalar **potential Φ(s)** each step;
  the training loop forms the per-learner-step shaping reward

      F_t = λ · ( γ · Φ_L(s') − Φ_L(s) ),   λ = 0.5 (ruled)

  applied on each learner decision step (s → the learner's next decision state), with the
  **side-relative potential** Φ_AFF = Φ, Φ_NEG = −Φ. Φ is **Φ_maxdiff** (rl_training_spec
  §Reward). This distributes shaping **densely per step**, replacing the terminal-only bonus.

  **Four correctness constraints (invariance — future readers MUST preserve all four; a
  violation reintroduces the optimum-distortion that caused the retired bonus's
  bonus-dependence):**
    1. **Φ is a pure function of state** — the current graph only; never action identity,
       history, or which side built a node. Any such dependence breaks policy-invariance.
    2. **γ in F_t MUST equal the return/GAE discount** (`SemanticsConfig.discount`, currently
       0.999), read from that single source. A mismatch makes F_t non-potential and *can
       change the optimal policy*.
    3. **Φ(terminal) = 0 and Φ(empty graph) = 0**, so the telescoped shaping contributes
       zero net discounted return per trajectory — the invariance identity.
    4. **Side-relative potential** Φ_NEG = −Φ_AFF (valid because the ballot is zero-sum).

  **Why this reverses the prior "all shaping at termination / non-terminal reward = 0"
  contract:** the retired flat bonus was a terminal, *non*-potential term that distorted the
  optimum (empirically: win-rate tracked the coefficient and collapsed when withdrawn —
  Real Run 1). PBRS is per-step by construction and, under the four constraints, is
  **policy-invariant** — it changes learning *speed*, never the optimal policy — which is
  why it needs **no annealing** (nothing to withdraw).

  **Mid-round chain resolver — one implementation, two callers** (mirrors the `node_accrual`
  "single implementation, two callers" pattern, §Observation contract → Node-level accrual).
  Φ is built from a chain resolver factored out of the judge's `_build_chains`, run in two
  scopes:
    - **judge, at termination:** BD-reachable scope, `as_of = None`, applying the
      impact-weighing exclusion at the ballot — unchanged, byte-identical verdicts.
    - **mid-round, per step:** whole-graph scope, `as_of = current slot`, and — per this
      milestone — **also applying the impact-weighing exclusion rule**, so Φ zeros chains
      already outweighed by weighs made so far. (A discounted **nascent channel** at κ was
      trialed here and **NEUTRALIZED to κ = 0 on 2026-08-12** — it degraded the bootstrap
      screen and cannot inject net signal under PBRS invariance; dead end, see rl_training_spec
      §Reward. At κ = 0 the mid-round Φ is byte-identical to the extended-only potential.)
  As with node_accrual, this must be the *same* factored function the judge uses, not a
  second copy.

  `info['reward_breakdown']` at termination now splits each side into `ballot` and
  `inert_penalty` (the flat `chain_extension_bonus` key is retired); the PBRS shaping is a
  **per-step** reward, reported per step (`info['phi']` and the training loop's per-step
  shaping series), not a terminal lump.
- **Inert-action penalty** (`inert_penalty_coef`, default `0.0`; ruled 0.0 —
  **dormant**): a per-side reward hook, currently zero. The only inert class it can
  apply to is the **no-op re-extend** — the other three are now structurally illegal
  (§Governing principle) and never sampled. Detection is the pure
  `is_inert(state, action) -> (bool, kind)` predicate, now narrowed to the single
  `noop_reextend` kind, still a *sibling* of the legal-action generator and **never
  part of it** (reward-only, unchanged contract). At 0.0 the penalty is a no-op; the
  no-op re-extend is instead priced structurally via **full-slot cost** (§Liveness
  stamping). The step-time per-side/per-kind counters remain for diagnostics/logging.
  `info['reward_breakdown']` still carries a per-side `inert_penalty` key
  (−`inert_penalty_coef` × noop count; zero while dormant).
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
