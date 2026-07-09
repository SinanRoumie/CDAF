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
- **Advocacy:** the **shared premise** both sides litigate. Offense chains attach to it by a
  support-type dependency; an advantage (AFF) and a disad (NEG) both root in the advocacy and differ
  only by side + sign. Advocacy is **not** a valid `OffensiveAttack` target — you outweigh a proposal,
  you do not turn it; offense aimed at an advocacy is inert (§3.4). There is no "advantage edge" /
  "disad edge" type; advantage-vs-disad is derived, never stamped. (Full commitment:
  `extension_migration_spec.md` §2.1.)
- **BallotDirective:** structural anchor; BDs are the discovery roots (§4) and the validated
  contributors at the ballot (§7).

**Edges (4 types under Model C).** A `Support` raises its target's strength (and forms chain/premise
dependencies). A `DefensiveAttack` lowers its target's magnitude. An `OffensiveAttack` is a
competing-polarity claim on a link or impact. A `Comparison` connects a Weighing to the pair it
ranks. Support and attack are the *same accrual operation* regardless of whether the target is a
chain member or someone else's attacker — role is read off topology, not off an edge subtype.

**Extension is per-node state, not an edge.** The former `ExtensionEdge` is **retired**. Each node
carries a **liveness record** — the speeches it was carried through, each tagged `contested` or
`conceded` — and extension is the act of stamping that record. The judge reads liveness from the
record, never by walking edges (§6). Full model, migration, and converter:
`extension_migration_spec.md`.

### 2.1 Speech order and side (judge-owned)

The **model** owns the canonical vocabulary (in `model/speeches.py`), because per-node liveness
ordering is a model-level fact and `model/` must not import `judge/`. The judge imports it from
there. Confirmed against real authored rounds, the exact strings are:

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
- **Speech recency** orients clashes: in any cross-side clash the node in the **later speech is the
  attacker/responder** (you can only answer what was already said). (Extension no longer needs
  orienting — it is per-node state, §6, not an edge.)

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

**Attacker-liveness gate — an abandoned attack lapses (v4).** An attack feeds the accrual above
**only while the attack itself is live** — i.e. extended by its maker (read from the attacker's own
liveness record, §6; the side-agnostic union where that applies, e.g. a turn kept live by either
side). An attack whose maker **dropped it** — its liveness record fails its own extension test — is
**not** in the `{a_j}` set at all: it **lapses** and contributes nothing. Concretely, a non-unique
read in 1NC and never extended does **not** sit in the uniqueness's attacker product at σ = 1.0 and
zero it; it is simply gone. **Crucially, "the opposing side did not answer it" does not by itself
confer conceded/full strength** — concession holds only for an attack that is *also live*. Dropping
an attack and conceding an attack are different: you concede an attack the maker **keeps**; a maker
who **abandons** its own attack forfeits it regardless of whether the other side ever spoke to it.

This does **not** touch the mitigation path (§3.1, oracle 3). If the **target** answers the attack —
attacks the attacker — the attack is still *live* (its maker extended it), stays in the `{a_j}` set,
and is reduced by the leaves-first DF-QuAD recursion exactly as before. The gate removes only attacks
the **maker** abandoned, never ones the **target** answered. (The gate applies to defensive and
offensive attacks alike; for a turn, liveness is the side-agnostic union of §6.) **This is a
judge-semantics change → version bump** (v4).

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

Directional resolution of a contested link is **a clash between two same-type nodes** (the AFF link
vs. the NEG turn's competing-polarity claim), so it is resolved by the general recursive
clash-breaker, `resolve` (§6.5): a **determinate won weigh** over that link clash decides direction
outright — the link keeps the preferred side's polarity regardless of raw σ — and only when the weigh
is **absent or indeterminate** does direction fall back to the DF-QuAD-against-0.5 computation above.
This is why a won link-weigh *saves a turned link*: the weigh is the established directional
preference, consumed here, not merely an impact comparison at the ballot. See §6.5 for the full rule
and the pass-ordering it requires.

**Flipping is a sign operation and never touches magnitude (v3 invariant).** Whether the link keeps
its polarity or flips, its magnitude is unchanged by the flip — magnitude changes *only* through
defensive attack (§3.1). A winning turn does **not** drive the link's magnitude to zero; it flips the
sign and the magnitude carries through, so the flipped link becomes offense *at strength* for the
turning side. (The earlier behavior — a winning turn zeroing its target — was the turn-offense bug:
it let a turn *neutralize* the AFF link but never *generate* NEG offense from it.) This invariant must
hold symmetrically: link-wins, turn-wins, and stacked turns all preserve magnitude through every
flip. See §3.5.

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
attack with no matching factor to operate on (e.g. offense aimed at a pre-world uniqueness node, an
`OffensiveAttack` on an advocacy — a proposal is outweighed, not turned — or a cross-channel attack
with nothing to attenuate) **contributes nothing** — it is inert. The judge does not reject it; it
simply has no effect on any sigma. This keeps the judge robust to malformed graphs and defers the
uncertain link-vs-impact boundary to behavior rather than a hard ban.

### 3.5 Turn offense (v3)

A turn is not a special mechanism — it is a polarity flip (§3.2) that preserves magnitude (§3.2
invariant) over a chain whose nodes are side-agnostically live (§6). Everything about turn offense
falls out of those two rules plus the sign product (§3.3). The unified rule:

- **A turn flips its target link's polarity and preserves the link's magnitude through the flip.** The
  flipped link becomes offense for the side the *composed sign* favors, at the surviving magnitude.
- **The turned chain generates offense iff every node on it is live — sustained by ANY side.** The
  whole chain from the flipped link through to its terminal impact must be live (§6, side-agnostic).
  A turn into a **dead impact** (the target impact fell out of the round) generates nothing — there is
  nothing to inherit. This is the inheritance rule: NEG inherits the AFF's impact **iff that impact is
  live**, by whoever kept it alive.
- **Two win paths, one check (§6).** (a) AFF keeps its argument live and NEG extends the turn; (b) AFF
  drops the contention and NEG carries the whole argument with the turn as the new link. The judge
  does not distinguish them — it checks that every node on the turned chain is live via the union of
  both sides' liveness stamps. (a) and (b) are just two ways that union is satisfied.
- **Multiple turns compose by sign product (§3.3), magnitude preserved through each.** A double-turn
  is `(-1) × (-1) = +1`: the link returns to AFF polarity, and because each flip preserved magnitude,
  AFF inherits the surviving strength. No special double-turn handling — it is the sign product with
  the magnitude invariant. **Ownership of the turned chain's offense is decided by the final composed
  sign**, not by who read the last turn: one turn → NEG, two → AFF, three → NEG.
- **A turn kills the offense it flips.** When the link flips to the opponent, the original side's
  offense through that link is gone (its sign now cuts the other way) — the flip simultaneously ends
  the original chain and creates the turned chain. This is one operation, not two resolutions.

Emit `POLARITY_FLIP` (via `preference`/`dfquad`) and `CHAIN` for the turned chain with its composed
sign, preserved magnitude, and owning side. **This is a judge-semantics change → version bump (v3);
re-confirm the oracle harness.**

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
  **unresolved**, cannot establish offense. Emit `UNRESOLVED`. **Refinement (final-speech offense):**
  a final-speech node *does* engage if it continues a clash that was **contested** entering that
  speech (the attachment point's liveness status for the prior opposing speech is `contested`) —
  e.g. answering a turn the opponent carried in. A node that is **conceded-live but not contested**
  is not a legitimate site of new final-speech offense; spiking a conceded link into an impact in the
  2AR is inert (the offense had to be established while the opponent had standing). The test is
  structural, read off liveness status — never off the node's label. (Full rule:
  `extension_migration_spec.md` §4.)

Then apply **extension** (§6): drop any chain whose spine is not carried through every one of its
side's speeches from introduction on — read from each node's **liveness record**, not from edges.

---

## 5. Framework gating (Pass 5a)

Resolve the framework sub-debate (a won framework is unattacked-or-restored and extended). Then
**binary-gate** impacts: an impact with no support path to the winning framework is excluded from the
tally entirely. In-scope or out, no continuous reweighting. Emit `FRAMEWORK_GATE` per impact.

---

## 6. Extension (binary, total over the spine — read from liveness records)

Extension is **not** a continuous score, and under Model C it is **not** an edge. Each node carries a
**liveness record**: the speeches it was carried through, each tagged `contested` or `conceded`. An
argument counts iff every **spine node** — the relevant uniqueness, link, impact, and advocacy — has
its liveness record covering **every one of its own side's speeches from its introduction onward**,
with the neg block (2NC/1NR) counted as one speech. The judge reads this by record lookup, never by
walking edges. (Full model, the extend act, side-agnostic liveness, shared-node union, and the
converter: `extension_migration_spec.md`.)

- AFF spine speeches: whichever of 2AC, 1AR, 2AR follow introduction. NEG spine speeches: the block,
  then 2NR. An AFF argument is **not** dropped for failing to appear during NEG speeches.
- **No new chains in rebuttals** — a chain whose *introduction* speech is a rebuttal does not count
  (reads the node's introduction `speech`).
- Non-spine nodes need not be extended.
- **Liveness is side-agnostic — check the union of both sides' stamps.** A node stays live as long as
  *any* live argument routes through it, regardless of which side introduced it. For a **turned
  chain**, the flipped link and its terminal impact count as live iff their liveness records are
  covered by the **union** of both sides' stamps — not by the turning side alone. This yields the two
  win paths (§3.5) with no special-casing: **(a)** AFF keeps the argument live (extends the
  link/impact) and NEG extends the turn — the union is covered by both; **(b)** AFF drops the
  contention and NEG carries the whole chain with the turn as the new link — the union is covered by
  NEG alone. Either way the check is identical: is every node on the turned chain live by *someone*?
  A turn into a **dead impact** (no side kept it alive) generates nothing — nothing to inherit. No
  special turn rule — the ordinary side-agnostic spine check produces all of this.
- **Shared trunk nodes live by union.** Advocacy/uniqueness shared across a branch's paths stay live
  while any live path through them is extended; collapsing one path never un-stamps a shared node
  another live path still needs.
- **Re-engagement is allowed.** A side may extend/answer a node it had stopped extending once it is
  live again (e.g. the opponent turned it and carried it forward).
- **Extension gates attacks too, not just offense chains (v4).** Liveness governs **every** node's
  participation, including an **attack's**. An attack (defensive or offensive) contributes to its
  target's accrual (§3.1) **only while the attack is live** — extended by its maker, read from the
  attacker's own liveness record (the side-agnostic union where that applies). An attack the maker
  **abandoned** lapses and is dropped from the target's attacker set; it is **not** kept alive, and
  **not** scored "conceded," by the mere fact that the opposing side never answered it. This closes
  the gap where extension gated only offense-bearing spine chains while a dropped defensive attack
  still contested at full strength. Emit `INERT_ATTACK` (reason `lapsed…`) for the forfeited attack.
  The **answer** path is untouched: an attack the *target* answered is still live and mitigates as
  ever (§3.1).

A chain that fails extension contributes **zero** to net offense (extension is a boolean gate, not a
multiplier — this replaces any separate `C_ext` term). Emit `EXTENSION_FAIL` naming the spine node
and the missing speech.

---

## 6.5 Clash resolution — recursive weighing (the primary clash-breaker)

Weighing is **not** an impact-only, ballot-stage preference. It is the **general clash-breaker over
any two same-type nodes** — two impacts, two links (a turn's polarity clash), two uniquenesses, two
frameworks, two interpretations. Any same-type pair can be weighed; where no competing claim exists,
no weigh is authored and the question is moot. Weighing **never edits δ** — it does not change a
node's strength. It tells the judge *how to break a clash*, and a tabula rasa judge honors the
debaters' clash-breaking instruction rather than substituting its own magnitude arithmetic. Magnitude
is only the fallback for a clash the debaters did not resolve.

DF-QuAD still runs and computes every node's surviving strength (drops, concessions, attacks) exactly
as before. The change is at **clash resolution**: when two same-type nodes clash, the judge consults
weighing **first**, and reaches for raw δ **only** if weighing does not resolve.

### The rule (recursive, one definition at every depth)

```
resolve(clash):
    P = the weighing layer immediately above this clash
    if P contains exactly ONE surviving preference:      # determinate
        that preference decides the clash (overrides raw δ)
    else:                                                 # indeterminate: absent or tied
        decide the clash by DF-QuAD magnitude (raw δ)

    where a preference "survives" iff it is not dropped AND not defeated by a
    higher weighing clash — determined by resolve() applied one level up
    (meta-weighing over weighing, meta-meta-weighing over that, ...).
```

The levels, top to bottom: **meta-weighing → weighing → DF-QuAD magnitude**. A clash is
**determinate** iff the level above it yields a single surviving preference; **indeterminate** iff
that level is empty or itself unresolved — which is just *a clash one level up*, resolved by the same
rule. Magnitude is the **base case / floor**: it has no level above, always yields a comparison, and
never punts upward, so the recursion terminates.

**Well-founded:** each step climbs to strictly fewer, higher nodes in a finite graph, and the
magnitude floor guarantees a base case — so `resolve` always terminates. Implement it as an actual
recursion, not a fixed depth-2 check (a fixed check is wrong at depth ≥ 2).

### The three indeterminate cases are one case at different depths (illustrations)

- **(a) no weigh** — the weighing layer is empty → fall to magnitude.
- **(b) symmetric weigh** — both sides weigh the *same two nodes* oppositely with no meta-weigh
  breaking them → the weighing layer has no lone survivor → fall to magnitude.
- **(c) non-resolving weighs** — AFF weighs on magnitude, NEG on probability, no meta-weigh saying
  which dimension controls; or AFF says mag>prob and NEG says prob>mag with nothing resolving it →
  the weighing layer ties → fall to magnitude.

(b) and (c) are (a) one level up: "the weighing layer failed to produce a single survivor" is the
same condition as "no weigh exists." The judge does not enumerate depths — it applies `resolve`.

### Where it feeds

- **Polarity (§3.2):** a link/turn polarity clash is resolved by `resolve`. A determinate won
  link-weigh keeps the preferred side's polarity outright; indeterminate → the 0.5 σ threshold.
- **Impacts at the ballot:** the same `resolve` ranks surviving offense — determinate weigh overrides
  raw δ, indeterminate falls to raw δ.
- Emit `WEIGH` (with the pair, the resolved preference or `symmetric`, and `via`).

### What a determinate weigh does — one consequence, generalized (v5)

A determinate weigh **defeats the dispreferred member** of the clash, and — the general mechanism —
**a defeated member does not attack the winner**. That consequence is **one rule**, but its
**expression follows the channel of the clash** (§3 keeps the sign and magnitude channels strictly
separate, so "does not attack the winner" is realized in whichever channel the clash lives in):

- **Magnitude-channel clashes — uniqueness, framework, any defensive attack:** the defeated attacker
  is **removed from the winner's DF-QuAD accrual**, so a won uniqueness-weigh actually saves the
  uniqueness (the defeated non-unique no longer contests it). This is folded into the **same attacker
  gate** as the liveness rule (§3.1, v4): an attacker contributes to its target's accrual only if it
  is **(a)** live (extended by its maker) **and (b)** not defeated by a determinate weigh over its
  clash with that target — `resolve(ctx, {attacker, target})` determinate with the **target** as
  winner. A defeated attacker emits `INERT_ATTACK` (reason `defeated by weigh`).
- **Sign-channel clashes — link/turn polarity:** the defeated **turn does not flip the link** — the
  link **holds its polarity via preference** (§3.2) — and the turn is **dropped from the link's
  magnitude**. This runs in the polarity pass, which reads `offense_on`; the magnitude gate above
  leaves `offense_on` untouched precisely so this channel keeps ownership of the turn.
- **Impact clashes:** the dispreferred impact's **chain is excluded at the ballot** (§7) — the same
  defeat expressed at the tally rather than in accrual.

All three are the same rule; they differ **only** in which channel "does not attack the winner" is
expressed in. **Do not collapse them into one literal prune point.** Concretely, do **not** prune a
defeated turn from `offense_on` before the polarity pass runs: that would make the link look
**unattacked** (so it would never enter the flip path), **lose the `POLARITY_FLIP via=preference`
trace record** that explains *why* the link held, and **conflate the sign and magnitude channels** —
which §3 forbids. The turn must be consumed in the sign channel (polarity), not silently deleted from
the magnitude channel's attacker set. Before v5 only the link and impact expressions existed, so a
won uniqueness-or-framework weigh was **decorative** (the `WEIGH` resolved but nothing changed); v5
adds the magnitude-channel expression and no more — one rule, three channel-faithful expressions, not
one prune site.

### Pass-ordering requirement

Because a determinate weigh must be able to **decide polarity**, the weighing towers must be resolved
**before** the clash resolution that consumes them — not two passes later. Resolve each weighing
sub-debate to determinate/indeterminate first (it needs only the weighing nodes' own drop/concession
status, not the main-chain polarities), then feed determinate weighs into clash resolution (polarity,
then impacts). This avoids a dependency cycle (weighing needs strengths; polarity now needs weighing;
strengths need polarity): the weighing towers depend only on their own sub-debate, so they resolve
independently first.

**This is a judge-semantics change → version bump.** A large class of clashes that previously fell to
magnitude may now be weigh-decided, so re-confirm the oracle harness against the new semantics rather
than assuming prior verdicts hold.

---

## 7. The ballot (Pass 6)

### Ballot (BD validation + net offense)
**Validate each BD.** A `BallotDirective` carries no stored `claimed_direction` or `target_node`
(the model's nodes carry `label`/`side`/`speech`/`liveness`), so the judge derives both: its
**claimed direction is its side** (an AFF BD claims AFF offense prevails; a NEG BD claims NEG), and
its **anchored node(s)** are those it is incident to in the undirected graph (§2.2). A BD whose
anchored argument resolves to `?`, or to offense favoring the opposing side, **fails and contributes
nothing**. Emit `BD_VALIDATE`.

From surviving, extended, in-scope, BD-anchored arguments, accumulate net offense:

```
N = sum(delta_a for surviving AFF args) - sum(delta_a for surviving NEG args)
```

applying won-weighing preferences to the comparison. Then the **asymmetric win condition**:

- **AFF wins** iff **all**: advocacy present; a complete chain with non-zero surviving magnitude; at
  least one in-scope impact; **and N > epsilon**.
- **NEG wins** otherwise. The `reason_class` distinguishes *why*: `AFF structural failure` (an AFF
  chain existed but collapsed / missing gate), `framework lock-out`, `NEG offense` (N < -epsilon), or
  `presumption` (abs(N) <= epsilon, or no AFF offense ever existed — indeterminate drains to NEG).
  "AFF structural failure" and "presumption" are distinct outcomes and must not be conflated: the
  former means AFF built offense that failed, the latter means the round is indeterminate.

Emit a final `BALLOT` (carrying N, reason_class, and the offense decomposition). The result is binary.

---

## 8. Decision trace schema (`judge/trace.py`)

One record per consequential decision, emitted as the passes run. The trace is the debugging tool:
when a verdict disagrees with your read, the first diverging record localizes the bug to one pass.
Each record has **core** fields (the decision data) and, for some, **descriptive** fields that the
RFD/panel reads — judge-populated, consumed downstream, and never able to change a verdict.

| Record | Core fields | Descriptive fields (RFD support) | Pass |
|---|---|---|---|
| `DROP` | node_id, owner, intro_speech, window_speech | — | 2 |
| `UNRESOLVED` | node_id, owner, intro_speech | — | 2 |
| `EXTENSION_FAIL` | chain_id, missing_speech, spine_node_id | — | 2/6 |
| `MAGNITUDE` | node_id, base_tau, surviving_sigma, attackers[], supporters[] | — | 3 |
| `POLARITY_FLIP` | link_id, from_sign, to_sign, sigma, via (preference/dfquad) | — | 3 |
| `INERT_ATTACK` | edge_id, reason | — | 3 |
| `CHAIN` | chain_id, sign, mag, delta | side, owner, extended, in_scope, collapse_reason, responsible | 3 |
| `FRAMEWORK_GATE` | impact_id, framework_id, in_scope | — | 5 |
| `WEIGH` | weighing_id, outcome (resolved/symmetric), preferred_node, via | pair[], overrode | 5 |
| `BD_VALIDATE` | bd_id, result, reason | side | 6 |
| `BALLOT` | N, gates_passed[], winner | reason_class, aff_sum, neg_sum, decomposition[] | 6 |

---

## 9. Pass order (summary)

1. **Discovery** — enumerate BD-anchored chains.
2. **Drop + extension** — response windows; lock drops; cut un-extended chains.
3. **Node accrual** — DF-QuAD per node (leaves first) → surviving σ. (No polarity yet.)
4. **Weighing towers** — resolve each weighing sub-debate to determinate/indeterminate via `resolve`
   (§6.5); these depend only on their own drop/concession status, so they settle before clash
   resolution and cannot cycle with polarity. **Then consume the towers against accrual (v5):** drop
   every attacker defeated by a determinate weigh over its clash with its target (§6.5) and re-accrue,
   so a won uniqueness/framework/defensive weigh actually removes the defeated attacker. (The link
   case is consumed in pass 5's polarity channel; the impact case at the ballot.)
5. **Clash resolution** — resolve same-type clashes (§6.5): determinate weigh decides, else raw δ.
   This is where **effective polarity** is set (a won link-weigh keeps polarity; else the 0.5 σ
   threshold). Then chain sign/magnitude products → delta per chain.
6. **Framework gate** — exclude out-of-scope impacts.
7. **Ballot** — validate BDs; rank surviving offense with `resolve`; sum net offense; apply the
   asymmetric win condition; indeterminate → presumption.

Discovery flows outward from BDs; scoring flows inward to them. A node is scored only after its
attackers are scored. **Weighing resolves before polarity** so a won weigh can decide a link clash
(§6.5 pass-ordering). This ordering changed with the recursive-weighing upgrade — **version bump**.

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
