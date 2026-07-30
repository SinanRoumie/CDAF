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

**Edges** carry:
- `source`, `target` — node ids
- `edge_type` — support / attack (turn status is derived structurally by the
  judge, never declared by the agent)

**Weigh nodes** additionally carry the two compared node ids and the `favors`
pointer, per Phase 0.

**Identity merges** do not create a second node. An `introduce` action whose
target is an existing node and whose content is asserted as the same claim
attaches to that node's identity, raising its in-degree rather than adding a
sibling. Downstream attack propagation to all dependent chains follows from
this automatically via existing judge reachability.

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
  orphaned.
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

## `reset()` / `step()` contract

Gym-style, for compatibility with off-the-shelf PPO implementations.

`reset() -> observation`
Initializes an empty graph at slot 1AC with that slot's budget.

`step(action) -> (observation, reward, done, info)`

- **Non-terminal steps**: `reward = 0`, `done = False`. Binary terminal
  reward only; no per-move shaping in V1. (Phase 1 reward shaping that
  disables presumption for early iterations applies to the *judge
  configuration* at termination, not to intermediate steps.)
- **Turn advance**: on `end_speech()` or budget exhaustion, the acting side
  and slot advance per `SPEECH_ORDER`.
- **Termination**: after the final slot (2AR) completes, the environment
  hands the completed graph to the judge, which runs its six passes exactly
  as it does for oracle fixtures. The verdict maps to binary terminal reward.
  `done = True`.
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

2. **Judge invocation surface.** Whether the judge currently exposes a clean
   entry point that accepts a constructed graph object, or whether it expects
   a fixture-file format that the environment would need to serialize to.
   Affects termination-step implementation only, not the interface contract.

## Status

Ready for implementation. Item 2 is an implementation detail discoverable in
the repo and does not block the spec being committed.
