# docs/judge_spec.md — CDAF Judge Specification (V1)

This document specifies the deterministic CDAF judge. It is the bridge between the project's
argumentation design and the existing codebase. Read it alongside the `model/` package; the judge is
a new pure-Python package that imports `model/` and nothing app-side.

The judge reads a finished argument graph and returns a winner plus a decision trace explaining how
it got there. It has no learning, no UI, no opinion about argument quality, and no access to claim
text. It will later become the reward function of the RL environment unchanged, so it must stay
decoupled from everything dashboard-side.

---

## 0. Interface contract

```
judge(round) -> (ballot, trace)
```

- **Input** is a `model.Round` (its `elements`, with `nodes` / `edges` views; nodes carry
  `label`, `side`, `speech` and a class-level `ntype`; edges carry `source`, `target` and a
  class-level `etype`). The judge treats `label` as opaque — it is never read.
- **`ballot`** is binary: `AFF` or `NEG`.
- **`trace`** is an ordered list of decision records (schema in §8), emitted from day one.
- **Purity:** `judge/` imports `model/` and the standard library (numpy ok). It never imports
  `app/`, never touches the filesystem or network, never reads the clock. Given the same `Round`
  it returns the same result every time.
- **Robustness:** the judge accepts any structurally-valid `Round`, including malformed or
  incoherent graphs (a hand-author's miskey, or later an agent's output). Incoherent moves are
  **inert**, not errors — see §3.4. The judge never raises on a well-typed `Round`.

---

## 1. Pinned decisions (V1)

These are settled. They are named constants in `judge/config.py`.

| Decision | Value | Consequence |
|---|---|---|
| Base node strength | **tau = 1.0** (presumed-true) | uncontested nodes stand at full strength; attacks erode, supports restore |
| Evidence/analytic weight | identical base | no quality prior; any difference must be argued |
| Link vs impact | mechanically identical | distinction is positional (see §2); "impact" = terminal node, for weighing eligibility |
| Polarity threshold | 0.5 | a contested link's surviving magnitude >= 0.5 keeps polarity, < 0.5 flips |
| Extension | **binary, total over spine** | an argument counts only if its spine nodes are extended through every one of its side's speeches from introduction on (see §6) |
| Weighing | ballot-stage preference | never edits delta; won weighing overrides raw delta, absent/tied weighing falls back to raw delta |
| Presumption | hardcoded **NEG**, uncontestable | every indeterminate result drains here |
| Near-zero net offense | abs(N) < epsilon -> presumption | configurable small epsilon; prevents float noise from manufacturing an AFF win |

**Note on stale numbers.** Earlier project docs used tau = 0.5 and declared link magnitudes (the
0.7 / 0.165 worked examples). Those are superseded. Build oracle fixtures fresh from tau = 1.0.

---

## 2. The graph, as the judge sees it

A node's **type** encodes role, not a distinct object. Strength accrues identically at every node
(§3.1); what the strength then does depends only on the node's position in the graph.

- **Pre-world (defense-only):** Uniqueness. Can be defended against only by competing uniqueness;
  cannot bear offense.
- **Post-world (offense-bearing):** Link and Impact. Mechanically identical in V1. A node is an
  **impact** for weighing purposes iff it is **terminal** — nothing chains forward out of it. A
  mid-chain "impact" is functioning as an internal link and is treated as one. Links and impacts may
  carry offensive and defensive relations with one another through the same magnitude channel.
- **Framework / Weighing:** resolve sub-debates that gate and rank impacts (§5, §7).
- **Advocacy / BallotDirective:** structural anchors; BDs are the discovery roots (§4) and the
  validated contributors at the ballot (§7).

**Edges.** A `Support` raises its target's strength. A `DefensiveAttack` lowers its target's
magnitude. An `OffensiveAttack` is a competing-polarity claim on a link or impact. An `Extension`
marks temporal re-assertion. A `Comparison` connects a Weighing to the pair it ranks. Support and
attack are the *same accrual operation* regardless of whether the target is a chain member or
someone else's attacker — role is read off topology, not off an edge subtype.

### 2.1 Speech order and side (judge-owned)

The judge owns the canonical vocabulary; the model stores these as free strings and the judge
maps them. Confirmed against real authored rounds, the exact strings are:

```
SPEECH_ORDER = ["1AC", "1NC", "2AC", "2NC/1NR", "1AR", "2NR", "2AR"]   # 7 speeches; neg block is the single string "2NC/1NR"
SPEECH_SIDE  = {"1AC": AFF, "2AC": AFF, "1AR": AFF, "2AR": AFF,
                "1NC": NEG, "2NC/1NR": NEG, "2NR": NEG}
```

A node's side is inferable from its speech and the two always agree in authored rounds. Order by
`SPEECH_ORDER` index, **never** by element index — on-disk order is byte-fidelity, not chronology.

### 2.2 The judge is direction-agnostic (orientation principle)

Edge `source`/`target` direction is **not** load-bearing and the judge must not depend on it.
Hand-authored rounds draw the same relationship both ways (an attack appears as attacker->target or
target->attacker arbitrarily; extensions and BD links likewise), and future agent graphs will be
just as arbitrary. The judge therefore treats edges as **undirected connectivity** and recovers
orientation from two reliable signals instead:

- **Node type** orients chains: Uniqueness is the pre-world root, Impact is the terminal sink,
  BallotDirective is the ballot sink. Discovery and chain direction follow type, not arrows.
- **Speech recency** orients clashes and re-assertions: in any cross-side clash the node in the
  **later speech is the attacker/responder** (you can only answer what was already said); in any
  extension the **later-speech endpoint is the re-assertion**.

Because every speech is single-side, every cross-side attack necessarily spans two speeches, so
speech recency assigns attacker/target unambiguously. Same-side "attacks" are incoherent and inert
(§3.4). This principle is what makes the judge robust to the noise that real rounds contain.

---

## 3. The two channels (the core, kept strictly separate)

The single most error-prone point in the system. **DF-QuAD governs accrual at a node.
Multiplication governs propagation along a chain.** They are different operations and never merge.

### 3.1 Node accrual — DF-QuAD (`judge/dfquad.py`)

For a node with base tau, surviving attacker strengths {a_j} and supporter strengths {s_k} (each
already resolved, leaves first):

```
E = prod_j (1 - a_j) - prod_k (1 - s_k)        # E in [-1, 1]; supports raise (E>0), attacks lower (E<0)
sigma = tau + (1 - tau) * E    if E >= 0
sigma = tau * (1 + E)          if E < 0
```

(`prod_j (1 - a_j)` is the "no-attack" product — 1.0 when unattacked, -> 0 as attackers strengthen;
`prod_k (1 - s_k)` is the "no-support" product. So E is positive when support dominates and negative
when attack dominates. This is the canonical DF-QuAD combination function, Rago et al. 2016. An
earlier draft of this line had the two products swapped, which inverted the sign — corrected here to
match the J2 implementation.)

With **tau = 1.0**, an unattacked node stays at 1.0 (supports cannot exceed the cap; their role is
to restore strength after attack). A fully conceded lone defender (strength 1.0, no supporters)
drives its target to 0 — **terminal**. A defender itself contested down to, say, 0.6 pulls the
target only partway — **mitigation**. Terminal-vs-mitigatory thus falls out of whether the defense
survived; it is never a declared category.

**V1 accrual is attacks-only (support edges are the spine, not DF-QuAD supporters).** A `Support`
edge builds the chain (the magnitude channel of §3.3); it does **not** feed the `{s_k}` supporter
slot of the accrual above. Wiring spine supports into accrual would let a full spine restore a
conceded-attacked node to full strength and break terminal defense. In V1 a node is defended not by
supporting it but by **attacking its attacker** — the leaves-first DF-QuAD recursion lowers the
attacker's σ, which mitigates or removes its effect (this is exactly how oracle 3's mitigation
works). The `{s_k}` slot remains in the formula, reserved for a future explicit restorative-support
convention; in V1 it is always empty. Adding bare warrants to a contested node therefore does not
change its σ in V1 — a documented simplification, deferred like evidence weighting.

### 3.2 Effective polarity

A link carries a polarity (+1 / -1). An offensive attack is a competing-polarity claim that enters
the target's DF-QuAD. After accrual, read the link's surviving magnitude against 0.5: **>= 0.5 keeps
the original polarity, < 0.5 flips it.** Competing claims use symmetric bases. A link whose polarity
is genuinely unresolved is `?` (see §3.3).

**The flip is gated on the presence of an offensive attack.** Only a link that is the target of an
`OffensiveAttack` is eligible to flip. A link attacked only **defensively** keeps its original
polarity no matter how low its σ falls — a defensive attack reduces magnitude, never reverses
direction. So a link mitigated to σ = 0.3 by pure defense stays sign +1 with magnitude 0.3 (weakened
offense), and must **not** be flipped to -1. Reading the flip off σ alone, without checking for an
offensive attacker, is a bug: it would turn a side's own defensively-mitigated link into offense for
the opponent. (A link driven to σ ≈ 0 by defense is *dead*, not *turned*: sign stays +1, magnitude 0,
delta 0 — emit `MAGNITUDE`, not `POLARITY_FLIP`.)

Directional resolution of a contested link **first honors an established directional preference**
(a won weighing/framework claim about which way the link cuts); only when no such preference exists,
or it is itself contested, does it fall back to the DF-QuAD-against-0.5 computation above. (This is
the same "won preference overrides the raw number, absence falls back to the number" pattern used for
impact weighing at the ballot — applied here to polarity. It is distinct from impact weighing, which
runs later, at the ballot, over surviving offense.)

### 3.3 Chain propagation — sign (QPN) and magnitude (`judge/qpn.py`, `judge/chain.py`)

Along a serial chain, **multiply** — never run DF-QuAD along the chain.

```
sign(a) = prod_i p_eff_i        # product of EFFECTIVE polarities; any '?' link => sign '?'
mag(a)  = prod_i sigma_i        # product of every node's surviving strength along the chain
delta_a = sign(a) * mag(a)
```

`mag` is the running product of the chain's node strengths (uniqueness * link strengths * terminal
severity). Under tau = 1.0 a clean uncontested chain holds at 1.0 regardless of length — no
length-fragility. A single near-zero node collapses the product: one dead link kills the chain, with
no terminal-defense primitive required. A `?` sign means the chain cannot establish offense and
drains to presumption.

**Severity / probability convention.** The terminal impact's severity is one factor in the product.
"Probability" is not a separate declared scalar — it is already expressed as the chain product. Do
not multiply a probability term in again. In V1 every node's base is 1.0, so severity too is uniform
until something is argued; the judge never derives any factor from claim content.

### 3.4 Coherence is inert, not illegal

Channel-typed attacks: a defensive/offensive attack operates on a specific factor of the target. An
attack with no matching factor to operate on (e.g. offense aimed at a pre-world uniqueness node, or a
cross-channel attack with nothing to attenuate) **contributes nothing** — it is inert. The judge does
not reject it; it simply has no effect on any sigma. This keeps the judge robust to malformed graphs
and defers the uncertain link-vs-impact boundary to behavior rather than a hard ban.

---

## 4. Discovery and drop detection (Passes 1–2)

### Pass 1 — Discovery (outward from BallotDirectives)
Collect every `BallotDirective`. From each BD's incident node, walk the **undirected** connected
subgraph to enumerate the argument chains it anchors, orienting by node type (§2.2): Uniqueness is
the root, Impact the terminal, BD the sink. Edge arrows are ignored. **Only BD-reachable subgraphs
are evaluated.**

### Pass 2 — Drop detection + extension (temporal)
A node's **response window** is the *next speech in `SPEECH_ORDER` owned by the opposing side* after
the node's own `speech`. The attacker in any clash is the **later-speech node** (§2.2), not the
edge's `source` — direction is ignored; recency decides.

- **Answered** — an opposing node in the window speech is in a clash (attack edge, either drawn
  direction) with this node -> it goes to DF-QuAD (Pass 3).
- **Dropped** — the window speech passed with no opposing clash -> the node **locks at its strength**
  (conceded; under tau = 1.0 that is full strength). Emit `DROP`.
- **No window** — introduced in the final speech of its side, opponent never had standing ->
  **unresolved**, cannot establish offense. Emit `UNRESOLVED`.

Then apply **extension** (§6): drop any chain whose spine is not carried through every one of its
side's speeches from introduction on.

---

## 5. Framework gating (Pass 5a)

Resolve the framework sub-debate (a won framework is unattacked-or-restored and extended). Then
**binary-gate** impacts: an impact with no support path to the winning framework is excluded from the
tally entirely. In-scope or out, no continuous reweighting. Emit `FRAMEWORK_GATE` per impact.

---

## 6. Extension (binary, total over the spine)

Extension is **not** a continuous score. An argument counts iff every **spine node** — the relevant
uniqueness, link, impact, and advocacy — is re-asserted in **every one of its own side's speeches
from its introduction onward**, with the neg block (2NC/1NR) counted as one speech. A re-assertion
is a new node in the later speech joined to the prior instance by an `Extension` edge; the judge
reads the **later-speech endpoint as the re-assertion** regardless of which way the edge is drawn
(§2.2), and does not require the two endpoints to share an `ntype`.

- AFF spine speeches: whichever of 2AC, 1AR, 2AR follow introduction. NEG spine speeches: the block,
  then 2NR. An AFF argument is **not** dropped for failing to appear during NEG speeches.
- **No new chains in rebuttals** — a chain first introduced in a rebuttal speech does not count.
- Non-spine nodes need not be extended.
- **Inheritance is just extension.** When a turn flips a link, the flipped link and any inherited
  impact are ordinary nodes: they count only if extended through the turning side's subsequent
  speeches. Example: AFF impact in 1AC, NEG turn in 1NC, AFF concedes — NEG must extend **both** the
  link turn **and** the original impact, or the offense evaluates to nothing. No special turn rule;
  the ordinary spine-extension check produces this.

A chain that fails extension contributes **zero** to net offense (this replaces any separate `C_ext`
term — extension is a boolean gate, not a multiplier).

---

## 7. Weighing and the ballot (Passes 5b–6)

### Weighing (ballot-stage preference)
Weighing **never edits delta**. The delta values are fixed by §3. A won weighing claim (conceded, or
winning its own sub-clash, resolved by DF-QuAD over the for/against arguments) establishes a
preference the judge honors when comparing surviving offense — and that preference **can override**
what raw delta alone would say (a smaller-delta impact can be preferred). When weighing is **absent
or its clash is unresolved**, no preference is established and the judge falls back to **raw delta**
comparison. Meta-weighing is the same node type recursed. Emit `WEIGH`.

### Ballot (BD validation + net offense)
**Validate each BD.** A `BallotDirective` carries only `label`/`side`/`speech` (no stored
`claimed_direction` or `target_node`), so the judge derives both: its **claimed direction is its
side** (an AFF BD claims AFF offense prevails; a NEG BD claims NEG), and its **anchored node(s)** are
those it is incident to in the undirected graph (§2.2). A BD whose anchored argument resolves to `?`,
or to offense favoring the opposing side, **fails and contributes nothing**. Emit `BD_VALIDATE`.

From surviving, extended, in-scope, BD-anchored arguments, accumulate net offense:

```
N = sum(delta_a for surviving AFF args) - sum(delta_a for surviving NEG args)
```

applying won-weighing preferences to the comparison. Then the **asymmetric win condition**:

- **AFF wins** iff **all**: advocacy present; a complete chain with non-zero surviving magnitude; at
  least one in-scope impact; **and N > epsilon**.
- **NEG wins** otherwise: any AFF structural failure, framework lock-out, **N < -epsilon**, or
  **abs(N) <= epsilon** (indeterminate -> presumption, hardcoded NEG).

Emit a final `BALLOT`. The result is binary.

---

## 8. Decision trace schema (`judge/trace.py`)

One record per consequential decision, emitted as the passes run. The trace is the debugging tool:
when a verdict disagrees with your read, the first diverging record localizes the bug to one pass.

The **Core fields** are the §8 decision data — what the pass actually computed.
The **Descriptive fields** are added context the judge populates so a decision can be
reconstructed and narrated without recomputation; they are consumed by the RFD renderer
(`judge/rfd.py`) and the dashboard verdict panel. Descriptive fields never change a verdict
— a reader that ignores them still sees the full decision.

| Record | Core fields | Descriptive fields (RFD support) | Pass |
|---|---|---|---|
| `DROP` | node_id, owner, intro_speech, window_speech | — | 2 |
| `UNRESOLVED` | node_id, owner, intro_speech | — | 2 |
| `EXTENSION_FAIL` | chain_id, missing_speech, spine_node_id | — | 2/6 |
| `MAGNITUDE` | node_id, base_tau, surviving_sigma, attackers[], supporters[] | — | 3 |
| `POLARITY_FLIP` | link_id, from_sign, to_sign, sigma, via (preference/dfquad) | — | 3 |
| `INERT_ATTACK` | edge_id, reason | — | 3 |
| `CHAIN` | chain_id, sign, mag, delta | side, extended, in_scope, collapse_reason (extension_fail / sign_flip / defensive_kill / unresolved_sign / none), responsible (node id) | 3 |
| `FRAMEWORK_GATE` | impact_id, framework_id, in_scope | — | 5 |
| `WEIGH` | weighing_id, outcome (resolved/symmetric), preferred_node, via | pair[] (compared node ids), overrode (preference beat raw delta) | 5 |
| `BD_VALIDATE` | bd_id, result, reason | side (BD's claimed direction) | 6 |
| `BALLOT` | N, gates_passed[], winner | reason_class (AFF offense / NEG offense / presumption / framework lock-out / AFF structural failure), aff_sum, neg_sum, decomposition[] ({chain_id, side, delta, contributed}) | 6 |

---

## 9. Pass order (summary)

1. **Discovery** — enumerate BD-anchored chains.
2. **Drop + extension** — response windows; lock drops; cut un-extended chains.
3. **Node accrual + chain resolution** — DF-QuAD per node (leaves first); effective polarity; sign
   and magnitude products -> delta per chain.
4. *(folded into 3 for V1)*
5. **Framework gate, then weighing** — exclude out-of-scope impacts; establish weighing preferences.
6. **Ballot** — validate BDs; sum net offense with preferences; apply asymmetric win condition;
   indeterminate -> presumption.

Discovery flows outward from BDs; scoring flows inward to them. A node is scored only after its
attackers and supporters are scored.

---

## 10. Build order and gates

Build as a new `judge/` package, milestone by milestone, stopping at each gate.

| # | Build | Gate |
|---|---|---|
| J1 | `config.py` (pinned constants), `trace.py` (records) | imports clean; constants match §1 |
| J2 | `dfquad.py`, `qpn.py`, `chain.py` (the two channels) | unit tests: tau=1.0 conceded-defense -> target 0 (terminal); partial defense -> mitigation; clean N-link chain holds at 1.0; one zero node collapses the product; polarity flip at the 0.5 threshold |
| J3 | `passes.py`, `judge.py` (discovery -> ballot, full trace) | the §11 oracle rounds each produce the stated winner; trace highlights present |
| J4 | oracle harness (`tests/oracle/*.json` + expected verdicts, pytest) | all oracle rounds green; a wrong verdict's trace localizes to one pass |

Each oracle round is surgical — it isolates one mechanism — not realistic. A round you build in the
dashboard, confirm against your own read, and save is a fixture with no conversion.

---

## 11. Worked oracle rounds (build these first)

Author each from tau = 1.0. Strengths shown are post-resolution.

1. **Clean uncontested advantage.** 1AC: Advocacy -> Uniqueness -> Link -> Impact, all extended AFF
   2AC/1AR/2AR. NEG drops everything. Chain mag = 1.0, sign +. N = 1.0 > epsilon. **AFF.**
2. **Conceded terminal defense.** As (1) but NEG reads a defensive attack on the link in 1NC and AFF
   drops it; NEG extends it through the block and 2NR. Link sigma -> 0, chain mag -> 0, N ~ 0.
   **NEG (presumption).**
3. **Answered defense -> mitigation.** As (2) but AFF answers the defense in 2AC and wins it down, so
   the defender survives only at ~0.5; link sigma ~0.5, chain mag ~0.5 > epsilon. **AFF**, weakened.
4. **Link turn, extended.** NEG turns the 1AC link in 1NC; AFF concedes; NEG extends both the turn
   and the inherited impact through block + 2NR. Sign flips, NEG offense N < -epsilon. **NEG (offense).**
5. **Link turn, not extended.** As (4) but NEG fails to extend the inherited impact in 2NR. Chain
   fails §6, contributes 0. No NEG offense; AFF advantage also gone (turned). N ~ 0.
   **NEG (presumption).**
6. **Framework lock-out.** NEG wins a framework that excludes AFF's only impact (no support path).
   AFF has zero in-scope impacts. **NEG (lock-out).**
7. **Weighing overrides raw delta.** AFF impact delta = 0.8, NEG impact delta = 0.6, but NEG wins a
   conceded weighing claim that its dimension controls. Preference honored over raw delta. **NEG.**
8. **No-window unresolved.** AFF introduces a new offensive answer in 2AR (final speech); NEG never
   had standing. Marked `UNRESOLVED`, establishes no offense. Decided on the rest of the flow.
