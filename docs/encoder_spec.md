# CDAF Policy Encoder Spec (Phase 4)

## Scope

This spec defines the neural encoder that maps a partial round graph to node
embeddings, and the pooling that produces the graph-level representation the
critic reads. It does not define the action heads, the PPO training loop, the
self-play checkpoint pool, or reward shaping. Those are separate documents.

Depends on: `environment_shell_spec.md` for the observation contract,
`action_schema_spec.md` for the action vocabulary, `judge_spec.md` for
evaluation semantics.

## Governing decision: the policy is content-blind

The judge never reads node content. Two rounds with identical structure and
entirely different prose receive identical verdicts, so content contributes
exactly zero to reward. The learned policy therefore does not see content and
does not emit it: the action is `introduce(role, target, edge_type)`, with
content absent.

Two consequences follow and are accepted deliberately:

- The policy learns **structural strategy over argument graphs**, not debate
  in the full sense. It can learn to protect a spine, to turn rather than
  take out, to spend rebuttal budget on extension — but never that a
  particular link is unpersuasive, because nothing in its world models
  persuasiveness. This is a direct consequence of tabula rasa, not a defect.
- Shared nodes are reachable only as **structural convergence**. A
  content-blind policy attaching to an existing impact is exploiting a
  structurally advantageous attachment point, not recognizing that two claims
  are about the same thing.

Rendering prose over a finished graph is a one-way downstream step performed
post-hoc by a frozen LLM, for inspection and publication only. It never runs
during training.

## Architecture: full attention over nodes

Node representations are computed by **full self-attention across all nodes
in the round**, not by fixed-depth message passing.

Rationale: a spine can be advocacy → arbitrarily many links → impact →
framework → ballot directive, so no fixed hop count covers every chain. Any
choice of depth `k` would silently truncate context on longer chains. Full
attention has no hop limit, so chain length and graph size are both
irrelevant to whether one node's representation can reflect another's.

The usual objection to full attention — O(N²) cost — does not apply at this
scale. The action budget caps a round at 52 actions, and `extend`,
`concede`, and `end_speech` create no nodes, so realistic graphs run roughly
30–40 nodes. All-pairs attention over 40 nodes is ~1,600 pairs. Even an
order-of-magnitude budget increase leaves this negligible.

This also decouples the architecture from the speech budget ratios, which are
explicitly tunable. A depth-based encoder would need its `k` revisited
whenever budgets changed, breaking checkpoint comparability across budget
settings. Full attention has no such parameter.

### Structural bias

Attention alone sees a *set* of nodes, not a graph. Since the judge's entire
semantics live in the edges, structure is injected via a **learned additive
bias on the attention score, keyed by the directed relation between each
ordered pair of nodes**.

The relation is directed. When node A attends to node B, the bias differs
according to whether A attacks B or B attacks A — the same edge, opposite
roles. Semantic direction is asymmetric (an attack edge from A to B does not
mean B attacks A; the judge reads this direction and DF-QuAD accrual runs
bottom-up along it), while information flow under attention is already
all-pairs. No reverse-edge relations are needed: the GNN device of adding
reverse relations to carry messages backward is unnecessary when attention
propagates in every direction by construction. What must be preserved is only
the asymmetry of the relation itself.

Relation vocabulary (directed pairs):

- supports / supported-by
- defensively-attacks / defensively-attacked-by
- offensively-attacks / offensively-attacked-by
- compares-favored / compared-favored-by
- compares-unfavored / compared-unfavored-by
- no-relation

Comparison edges are split by `favors` because the asymmetry is the entire
point of the pointer: from a weigh node, "compared and favored" and
"compared and not favored" are different relations. Comparison also does not
propagate strength — it links a weigh node to the two nodes it compares —
so it should not share a bias term with the propagating relations.

### Edge recency

Edges carry one property not derivable from their endpoints: the speech in
which the edge was introduced. An edge introduced in the 2NC between two
1AC-era nodes has a speech neither endpoint shares, and response windows are
speech-relative, so this is strategically load-bearing.

Encoded as **relative recency** (how many speeches back the edge was
introduced) rather than absolute slot, bucketed. The bias table is keyed by
(relation type × recency bucket). At ~11 relations and ~4 buckets this is
~44 scalar parameters — negligible.

Edges do **not** carry liveness. Liveness is a node-level record read by the
extension gate; edges are not extended or conceded. Whether an edge is
currently effective is derivable from its endpoints' node features.

## Node features

- **role** — learned embedding over {uniqueness, link, impact, advocacy,
  framework, ballot_directive}
- **owner** — *relative* encoding (self / opponent), not absolute AFF/NEG
- **presumption flag** — whether this node's owner is the presumption-favored
  side
- **introduction speech** — categorical, plus a scalar "speeches ago"
- **liveness stamps** — 7-dim binary vector over speech slots
- **settled facts** — dropped, permanent extension failure, reachability
- **degree features** — in-degree and out-degree, split by edge type
- **σ and propagated QPN sign** — per-node accrual over the partial graph

### Why relative owner encoding

The game is almost entirely symmetric: chains build the same way from either
seat, turns work the same way, extension gates identically. Absolute encoding
would force the policy to learn that shared structure twice, once per side,
from the same sample budget.

The one genuine asymmetry — presumption is hardcoded NEG, so a wash favors
one side — is a single bit, handed to the policy directly as its own feature
rather than learned implicitly from a side label.

This also matters for the checkpoint pool: under relative encoding a
checkpoint is one agent that plays both seats, and self-play strength
transfers across sides. Under absolute encoding the same weights would see
two different input distributions, halving effective sample efficiency for no
benefit.

### Why node-level accrual is available mid-round

Per-node σ and propagated sign are computed over the partial graph. This is
legitimate where whole-round evaluation is not, and the distinction is
load-bearing:

> Node-level accrual asks "given this node's incoming attacks, what is its
> strength" — answerable at any point, over whatever attacks currently exist.
> Extension and the ballot tally ask about coverage across the round's
> *remaining* speeches, which have not happened. Only the second is ill-posed
> mid-round.

Node strength is also precisely what an agent needs to decide whether a
position requires frontlining and how — a decision real debaters make
mid-round from exactly this information.

**Single implementation requirement.** Node-level accrual must be one pure
function over nodes and edges, called by both the judge's accrual pass and
the observation layer. The encoder must not carry its own copy: a drifted
second implementation would train the policy against a slightly different
world model than the judge scores it in, which is a silent failure mode worse
than ordinary spec drift. It runs once per `step()` — tens of millions of
times across a run — so it must operate on the environment's state
representation directly, without materializing a `model.Round`.

## Global context

Current speech slot and remaining budget are round state, not node
properties. They are held as a separate global embedding and concatenated at
the pooling stage rather than replicated across every node.

Remaining budget is load-bearing for the critic: it disambiguates "large
activations because many chains survive" from "large activations because the
graph is late and therefore large."

## Pooling

Graph-level representation for the critic is **sum ‖ mean ‖ global**.

- **Sum** matches the value function's target structure — the tally is
  literally a sum over surviving chains — and preserves count information
  (three live AFF chains is not the same state as one).
- **Mean** is scale-stable against a graph that grows monotonically through
  every round, counteracting sum's drifting magnitude.
- Concatenating lets the critic learn the mix rather than committing to
  either.

Attention pooling was considered and deferred: it is more expressive and can
learn to approximate either component, but is less stable early in training
when attention weights are near-uniform. Adding it later is a small change,
not an architectural one.

## Parameter sharing

**Shared encoder** feeding both the actor heads and the critic. Standard for
PPO and more sample-efficient; the representations useful for choosing an
action largely overlap with those useful for estimating value. At this model
size, separate encoders would roughly double parameters to avoid a capacity
conflict that is unlikely to bind.

## Interface to the action heads

Action selection is factored, matching the ruled selection scheme: action
type, then target node, then the remaining structural parameter (role, edge
type, or `favors`). Each head is masked by the legal-action generator.

The target-selection head is a **pointer over node embeddings**, not over
indices: each node's embedding is scored against a learned query and softmaxed
across nodes. This handles variable node counts without a fixed output
dimension, and lets the policy generalize across rounds where the same
structure carries different node IDs.

Masking rather than propose-and-reject is what makes the factored scheme pay
off here — the legal-action generator's per-stage masks apply directly to
each head.

### Curriculum mask composition (opening unlock ladder)

During the AFF 1AC, the policy's type/role masks are further constrained by the
**opening unlock ladder** (rl_training_spec §Opening curriculum) — a *training
scaffold*, not a legality rule. Its composition with the environment's
legal-action mask is fixed:

- **AND-composition, never widening.** The curriculum mask is applied by
  element-wise AND against the legal-action generator's masks. It can only turn a
  legal action off for sampling; it can never make an env-illegal action
  selectable. Legality remains the environment's sole authority
  (environment_shell §Governing principle).
- **Keyed on speech slot and intra-1AC move index.** The mask is a function of the
  current speech slot and the move index within the 1AC only: it forces move 0 to
  `introduce(role = advocacy, target = NEW)` (masking `end_speech` at that
  decision), and from move 1 onward gates the *role* of a fresh `introduce` by the
  unlock ladder (advocacy → link → {uniqueness, impact} → {framework,
  ballot_directive}), reading which roles have been introduced so far this round.
- **No-op from the 1NC onward.** For every slot after the 1AC, and for every NEG
  slot, the curriculum mask is the identity (all-ones) — it composes to exactly the
  legal-action mask, so post-1AC behaviour is unchanged.
- **Never consulted by the judge.** Like every mask, it lives in the policy/env
  layers and is absent from `judge()`; it changes what the policy *samples*, never
  what a completed graph *scores*.

The ladder's semantic content and rationale (why BD gates on Impact alone, etc.)
live in rl_training_spec §Opening curriculum; this section specifies only how the
mask composes with the factored heads' existing per-stage masks (§Interface to the
action heads).

## Open items

1. **Attention layer count and width.** Full attention removes the hop-depth
   question but not the question of how many attention layers to stack, or
   the hidden dimension. Sweep parameters; no principled a-priori value.
2. **Recency bucket boundaries.** How "speeches ago" is bucketed for the edge
   bias table. Likely {same speech, 1 back, 2–3 back, older}, but worth
   tuning against observed response-window behavior.
3. **Sparse reward.** One bit at the end of a 52-action episode, with early
   self-play producing near-noise verdicts between two untrained policies.
   This is the principal training risk and is out of scope here; it belongs
   in the RL training spec.
