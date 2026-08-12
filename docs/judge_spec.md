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
| Link vs impact | identical accrual / polarity / turn channel, but **not interchangeable for chain completeness** | a scored chain's path needs ≥1 literal `Link` node (§2, §3.3.1); the distinction is otherwise positional — "impact" = terminal node, for weighing eligibility |
| Polarity threshold | 0.5 | a contested link's surviving magnitude >= 0.5 keeps polarity, < 0.5 flips |
| Extension | **binary, total over spine** | an argument counts only if its spine nodes are extended through every one of its side's speeches from introduction on (see §6) |
| Weighing | ballot-stage preference | never edits delta; preference is the weigh's explicit `favors` pointer (legacy default: own-side member cross-side, inert same-side); won weighing overrides raw delta, absent/tied falls back to raw delta |
| Presumption | hardcoded **NEG**, uncontestable | every indeterminate result drains here |
| Near-zero net offense | abs(N) < epsilon -> presumption | configurable small epsilon; prevents float noise from manufacturing an AFF win |
| Framework selection | **live-set cardinality** | exactly one live framework gates; zero or many is a **wash** (no gating, §5.2). Never order-dependent |
| Framework magnitude | **none** | a framework's σ is consumed at the liveness threshold only (§3.6); it is never a spine rep and never a δ term |

**Note on stale numbers.** Earlier project docs used tau = 0.5 and declared link magnitudes (the
0.7 / 0.165 worked examples). Those are superseded. Build oracle fixtures fresh from tau = 1.0.

**Note on uniqueness (v9).** These pinned decisions predate the uniform-uniqueness migration:
uniqueness is no longer a root-only node — every post-world node carries its own uniqueness attack
surface. Superseded, see §12.

---

## 2. The graph, as the judge sees it

A node's **type** encodes role, not a distinct object. Strength accrues identically at every node
(§3.1); what the strength then does depends only on the node's position in the graph.

- **Pre-world (defense-only):** Uniqueness. It is a **spine rep** — it sits at the front of a chain
  and multiplies into magnitude (§3.3) — but it bears **no offense**: it is never turn-eligible and
  never the source or target of an effective `OffensiveAttack` (§3.4). Its only attack role is
  **defensive**: a uniqueness may defensively attack another uniqueness *or* a link — both express the
  non-unique ("already inevitable regardless of your link"), never flipping it (§3.2). **Both routes
  go through the SAME poison-gate threshold (§12.4.3), not a proportional discount (Ruling A,
  non-unique unification):** a *surviving* non-unique (σ ≥ threshold) zeroes the contested link's
  contribution all-or-nothing — whether it lands on the link directly, or on the link's satellite
  uniqueness (driving *that* below threshold). Competing non-uniques break by weighing (§6.5) like any
  clash. (This drops the earlier "uniqueness clashes only with competing uniqueness" restriction: the
  non-unique can land on the link directly when no separate uniqueness node exists.)
- **Post-world (offense-bearing):** Link and Impact. They share one magnitude/sign/turn channel —
  **mechanically identical for accrual, polarity, and turns** — but are **not interchangeable for
  chain completeness**: a scored chain's path must contain **at least one literal `Link` node**
  (`kind == "link"`). A path of impacts alone is **not a live offense carrier**; the resolver returns
  `extended = False` with `collapse_reason = "no_link_premise"` (§3.3.1). This holds **even where a
  mid-chain "impact" functions positionally as an internal link** — the requirement is on node
  **kind**, not position. A node is an **impact** for weighing purposes iff it is **terminal** —
  nothing chains forward out of it. Links and impacts may carry offensive and defensive relations with
  one another through the same magnitude channel. **They are the only turn-eligible nodes** (§3.4): an
  effective turn requires an offense-bearing node at each end.
- **Framework:** a **scope instruction**. It says which offense the ballot may count (§5). It is
  **offense-inert**: it carries no polarity, contributes no magnitude factor, and cannot terminate a
  chain. Its strength σ is contested like any node's (§3.1), but σ on a framework is consumed at the
  **liveness threshold**, never as a magnitude multiplier (§3.6). A framework is **not** a valid
  `OffensiveAttack` target — you defeat a framework, you do not turn it; offense aimed at a framework
  has no polarity to flip and is inert (§3.4), the same as offense aimed at an Advocacy. A framework
  relates to a chain in exactly **one** way: support-reachability, which §5.3 reads as **anchoring**
  (scope). There is no separate "premise" or "is-about" relation — the judge does not model aboutness,
  and it must not, because aboutness is label-driven and the label is opaque (§0).

  The two things debaters say about a framework map to the two attack channels, and the second is a
  **two-edge** construct that needs no new node or edge type:
  - *"Their framework is flawed"* is a `DefensiveAttack` on the framework node. It lowers σ. Drive σ
    below the threshold and the framework leaves the live set and cannot gate (§5.1). Nothing new is
    required.
  - *"Their framework is harmful"* (the framework argument) is **one Link node wearing two hats**, not
    an attack *on* the framework as such. As a **spine rep** it roots its own offense chain,
    Link → Impact, anchored to a framework the reading side can win (typically its own) — this chain
    carries magnitude through its Link and Impact and contributes to N like any disad. **Separately**,
    that same Link draws a `DefensiveAttack` onto the target framework, lowering its σ toward the
    threshold. A node that is a chain member *and* an attacker on someone else is the ordinary
    "attack your attacker" topology of §3.1, read off type and position (§2.2); the framework argument
    is that pattern with a framework as the thing attacked. The offense stands on **its own anchor** and does
    **not** require the criticized framework to be live: "util is racist" generates offense under the
    reader's framework whether or not util is anywhere in the round. Anchoring the criticism to the
    framework it criticizes would be self-defeating — the criticism would share that framework's fate
    and go out of scope when it loses — so debaters bring their own framework to house it, which is
    real practice and emerges from anchoring alone.

  A framework may therefore be a **chain anchor** (scope, gating on liveness) and the **target of a
  defensive attack** (unseating, on σ) at once, while never being a spine rep and never carrying a
  magnitude factor.

  **What a framework is not.** *"Framework"* in debate practice is polysemous: it names a scope gate
  (util, deontology, "evaluate the discourse first") and it names a procedural contention with a ballot
  directive (theory's fairness/education → "reject the team"). These are structurally distinct objects
  and the graph keeps them apart. The discriminating test is: **with no other offense in the round, can
  this node alone produce a ballot?** If yes it is an Impact plus a `BallotDirective`, not a Framework.
  Fairness and education are **Impacts**: they carry magnitude ("minor abuse" vs "game-ending"), they
  are weighed against one another, they terminate chains, and they pair with a BD. A value/criterion is
  a **Framework**: you cannot win on it alone; it selects which offense counts. A theory shell is
  structurally a contention — violation (Link) → fairness (Impact) → reject the team (BD) — with no new
  node types. That the formalization forces this split is a result, not a gap.
- **Weighing:** ranks a same-type pair through `Comparison` edges; the general clash-breaker (§6.5).
  A Weighing declares **which** member of the pair it prefers via an explicit `favors` pointer;
  absent one, preference defaults structurally (§6.5 legacy default).
- **Advocacy:** the **shared premise** both sides litigate. Offense chains attach to it by a
  support-type dependency; an advantage (AFF) and a disad (NEG) both root in the advocacy and differ
  only by side + sign. Advocacy is **not** a valid `OffensiveAttack` target — you outweigh a proposal,
  you do not turn it; offense aimed at an advocacy is inert (§3.4). There is no "advantage edge" /
  "disad edge" type; advantage-vs-disad is derived, never stamped. (Full commitment:
  `extension_migration_spec.md` §2.1.)

  **NEG-offense rooting is ENFORCED (rule 4), not merely described — pure reachability, no
  mandatory-uniqueness.** A scored NEG offense chain must root in a **shared premise** reachable
  from its spine over `Support` edges, in one of two ways matching the two kinds of NEG offense:
  a **disad** roots in the **AFF Advocacy** (the plan both sides litigate) — reached EITHER by a
  direct `link → advocacy` Support edge OR by the **fusion idiom** (r27) `advocacy → uniqueness →
  link`, one hop up through the disad's satellite Uniqueness; a **framework argument** roots in a
  **Framework** — its offense is evaluated under a framework it reads (`imn → fneg`), not a link into
  the plan (§5.3), so the advocacy requirement does not apply to it (a framework argument is not a
  disad). A scored NEG
  chain reaching **neither** an AFF Advocacy nor a Framework is unrooted (floating, disconnected
  from any premise): the resolver collapses it with `collapse_reason = "unrooted_disad"` (§3.3.1,
  the disad analog of `no_link_premise`), scoring 0. There is **no mandatory per-Link uniqueness**
  requirement (the former two-shape rule is retracted): a disad Link needs its own Uniqueness only
  if the opponent chooses to **contest** it by reading a non-unique (§3.4, non-unique unification).
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

**Terminality follows node type, not introduction order.** An Impact is the terminal sink of its
offense chain (§2). It stops being *terminal* only when **another Impact** chains forward past it (a
genuine multi-impact chain). A **Link never affects an impact's terminality**, regardless of
introduction order: by the type orientation above a Link is always a *premise upstream* of the impact
it supports (uniqueness → link → impact → BD), never downstream — so a later-introduced convergent or
premise link supporting an impact must not disqualify that impact from being terminal. Because two
Impacts share a node type, type alone cannot order an impact→impact chain; **speech recency** is
retained *only* for that residual case (the later-introduced impact is the downstream one). This is
the sole use of recency for chain direction; it is not a general terminality signal. (Reading a
later *link* as "forward" was a bug — it leaked introduction order into a structural judgment and
silently dropped the chain of any impact that gained a late premise link, a state a legal
`introduce(link → impact, support)` in a later speech can reach.)

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
`OffensiveAttack` is eligible to flip, and only **offense-bearing** nodes (Link, Impact) are eligible
at all — an `OffensiveAttack` touching a Uniqueness, Advocacy, Framework, Weighing, or BD is inert and
flips nothing (§3.4). A link attacked only **defensively** keeps its original
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

### 3.3.1 Redundant paths and convergence to a shared impact (per-path aggregation)

§3.3's `mag = prod_i sigma_i` describes **one** serial root→impact path. When two or more Support
paths converge on the **same terminal impact** (an impact reachable from the advocacy by more than one
distinct spine), the impact-component chain is **not** a single flat product over the *union* of every
spine node — that would let a dead node on **one** redundant path (σ = 0) collapse an impact a clean
sibling path still fully carries. The component chain is built **per path**, then aggregated to one
object:

- **(a) Per-path liveness (extension).** Enumerate the distinct root→impact spine paths inside the
  component. Apply the extension gate (§6) **per path**: a path is *complete* iff every node on its own
  spine is extended. The **impact survives iff at least one complete path is fully extended.** A path
  dead at any single node — a de-linked link (σ = 0) or a dropped uniqueness (failed extension) —
  removes only **that path**; it never poisons a sibling. Extension is a property of a path, not of
  the union of all spine nodes in the component. (A **non-unique** is the one exception: it is a
  claim about the shared post-world *state*, so it poisons a convergent impact across **all** live
  paths, not just its own — §12.4.3.)

- **(b) Same-sign redundancy → max, ONE chain per impact.** Among the **surviving** paths that carry
  the **same sign** into the impact, the impact's magnitude is the **maximum** of their path magnitudes
  — not their product, not their sum. Redundant support neither compounds nor attenuates (two clean 1.0
  links onto one impact is one 1.0 impact — not 2.0, not 1.0 × 1.0). This is emitted as **exactly one
  chain object per impact-component**: the net-offense sum (§7) must see each impact **once**. The
  per-path enumeration changes only how `mag` / `sign` / `extended` are **computed** for the existing
  component-keyed chain; it does **not** split the chain into per-path objects. (A per-path-object
  implementation would double-count a two-link impact as +2; keeping one object per impact, with a max
  over surviving same-sign paths, is precisely the guard against that.)

- **(c) Convergence sign-conflict (EQUAL magnitude only).** When surviving paths carry **opposite**
  signs into one shared impact — a clean AFF path (+1) and a **live turned** path (−1) at **equal**
  magnitude — the impact's AFF offense is **washed**: the shared impact is rendered
  **non-resolved-+1**, i.e. its sign becomes `?` (UNRESOLVED) or the turned −1 — **never** a resolved
  +1 chain driven to zero magnitude. This distinction is load-bearing for `reason_class` (§7): a washed
  impact must not leave a resolved, extended, sign-+1 chain, or the ballot reads `AFF structural
  failure` (offense established, then zeroed) instead of the correct `presumption` (the round washed;
  no side established prevailing offense, and the clean chain is intact — it was cancelled, not
  collapsed). The turned path scores **for** NEG only through an anchored NEG `BallotDirective` (§3.5,
  §7); absent that BD it banks 0 and the impact simply washes to a tie. **Scoped to equal magnitude
  only.** Unequal-magnitude convergence (a partial turn that does not fully cancel the clean path) is
  **out of scope for V1** and is not governed by this rule.

### 3.4 Coherence is inert, not illegal

An attack operates on a specific factor of its target; an attack with no factor to operate on
**contributes nothing** — it is inert. The judge does not reject it, does not raise, and leaves no σ
changed; it emits an `INERT_ATTACK` record (§8) so the trace shows the edge was seen and deliberately
given no effect. `INERT_ATTACK` is a **record**, not an edge type — there are still only four edge
types (§2). "Inert" is a verdict the judge reaches about an ordinary edge, never a kind of edge an
author draws. This keeps the judge robust to malformed graphs (a hand-author's miskey, an agent's
output) and defers uncertain boundaries to behavior rather than a hard ban.

**One rule generates every inert case for offense: a turn requires an offense-bearing node at each
end.** An `OffensiveAttack` is a competing-polarity claim; it can only flip something that carries
polarity and magnitude, i.e. a **Link or Impact** (§2). If either endpoint is anything else, there is
no polarity to flip and the edge is inert. This single principle **subsumes** the previously
enumerated cases:

| `OffensiveAttack` between | Effect |
|---|---|
| Link ↔ Link, Link ↔ Impact, Impact ↔ Impact (cross-side) | **turn** — flips, offense at surviving magnitude (§3.5) |
| Uniqueness ↔ anything | inert — uniqueness bears no offense (§2); the non-unique is *defensive* |
| Advocacy ↔ anything | inert — a proposal is outweighed, not turned |
| Framework ↔ anything | inert — no polarity; defeat it defensively or by weigh (§5.1) |
| Weighing / BallotDirective ↔ anything | inert — you meta-weigh, you do not turn |

The rule is **direction-agnostic** (§2.2): it keys on "is either endpoint non-offense-bearing," never
on which end is `source`. "Uniqueness attacks Link" and "Link attacks Uniqueness" are the same
undirected edge and both are inert. A cross-channel attack with nothing to attenuate, and a
`Comparison` over a mixed-type pair (§6.5), are inert for the same reason: no matching factor.

**`DefensiveAttack` is the asymmetric counterpart and is NOT governed by this rule.** A defensive
attack lowers magnitude and never flips, so it is coherent against any node that carries either a
magnitude factor or a liveness threshold. In particular a `DefensiveAttack` on a **Framework** is
**not** inert — it lowers σ and is exactly how a framework is unseated (§5.1, §5.4) — and a
`DefensiveAttack` from a **Uniqueness onto a Link** is not inert — it is the non-unique (§2). The
asymmetry is the point: a uniqueness can *mitigate* a link but can never *turn* one. **Ruling A
(non-unique unification):** a `DefensiveAttack` from a Uniqueness onto a Link does **not** apply the
ordinary proportional DF-QuAD discount other defensive attackers apply; it is diverted into the
poison-gate threshold (§12.4.3) so it produces the SAME all-or-nothing effect as a non-unique
zeroing the link's satellite uniqueness. A surviving non-unique (σ ≥ threshold) zeroes the link;
below threshold it has no effect.

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

**Uniqueness under v9.** A turn capturing a chain now also captures the uniqueness nodes wired to
its post-world nodes, joining the capturing side's anchor set. Superseded, see §12.5.

### 3.6 Each node's σ is consumed exactly once

Strength accrues identically at every node (§3.1). What a node's surviving σ then *does* is fixed by
its type, and **no node's σ is read in two channels**:

| Node type | σ is consumed as |
|---|---|
| Advocacy, Uniqueness, Link, Impact (the **spine reps**) | a magnitude factor in the chain product (§3.3) |
| Framework | a **liveness threshold**: σ >= `POLARITY_THRESHOLD` keeps the framework in the live set (§5.1); below it the framework is out and gates nothing |
| Weighing | survival of the preference (dropped / defeated / standing, §6.5) |
| BallotDirective | not consumed. A BD is a structural anchor, validated by the offense it anchors (§7), never by its own strength |

This is the §3 channel discipline applied to the gating layer. Letting a framework's σ both clear a
threshold **and** multiply into the chains it gates would count the same contested strength twice: a
framework won at σ = 0.6 would govern the round *and* silently discount every impact under it by 40%.
**Gates are binary.** A framework that survives governs completely; one that does not, does not govern
at all. A barely-won util still means you evaluate consequences, and it does not make the extinction
impact smaller.

The corollary that decides the framework channel: because a framework carries no magnitude, a
framework clash has **no magnitude floor** to fall back to (§5.1, §6.5).

---

## 4. Discovery and drop detection (Passes 1–2)

### Pass 1 — Discovery (outward from BallotDirectives)
Collect every `BallotDirective`. From each BD's incident node, walk the **undirected** connected
subgraph to enumerate the argument chains it anchors, orienting by node type (§2.2): Uniqueness is
the root, Impact the terminal, BD the sink. Edge arrows are ignored. **Only BD-reachable subgraphs
are evaluated.**

### Pass 2 — Drop detection + extension (temporal)
A node's **response window** is the range of speeches in which an opposing answer is still *legal*.
The attacker in any clash is the **later-speech node** (§2.2), not the edge's `source` — direction is
ignored; recency decides.

- **Default (next-speech only):** the window is the *single* next speech in `SPEECH_ORDER` owned by
  the opposing side after the node's own `speech`. An argument introduced in speech N can be answered
  only in the immediately following opposing speech.
- **1AC exception (extended window):** a node introduced in **1AC** has a window covering **both** of
  NEG's constructive speeches — its latest legal response is **2NC/1NR**, not just 1NC. (`window_close`
  returns `2NC/1NR` for a 1AC node, the next opposing speech otherwise.)

**Response-window enforcement (attack validity/timing).** An attack that arrives **after its target's
window closes** is **inert** — the argument was settled/conceded as of the window's close, so a late
"answer" is not evaluated (no DF-QuAD contribution). This is enforced at attack-gathering (Pass 1),
alongside the same-side / turn-ineligibility / lapsed-attacker filters, and emits a **`WINDOW_CLOSED`**
trace record `{edge_id, attacker_id, attacker_speech, target_id, target_speech, window_close}` so a
late response is visible, not silent. **This is distinct from the no-new-offense-in-rebuttals rule
(§6):** that governs whether a *rebuttal move may introduce new offense*; this governs *response
validity/timing on an existing argument*, independent of whether the attacking move introduces offense.

- **Answered** — an opposing node **within the response window** is in a clash (attack edge, either
  drawn direction) with this node -> it goes to DF-QuAD (Pass 3).
- **Dropped** — the window passed with no in-window opposing clash -> the node **locks at its strength**
  (conceded; under tau = 1.0 that is full strength). Emit `DROP`. Because a late attack is now inert
  (above), a dropped/locked node is no longer silently reduced by an out-of-window "answer" — the
  Pass 2 lock and the Pass 3 magnitude channel agree.
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

Frameworks are scope instructions (§2). This pass decides which framework, if any, governs the round,
then binary-gates chains against it. **In-scope or out; no continuous reweighting.**

### 5.1 Selecting the governing framework

```
live = [F for F in frameworks
          if sigma[F] >= POLARITY_THRESHOLD          # survived accrual (§3.6)
          and maker_extension_ok(F)]                 # its own side carried it (§5.4)

defeated = set()
for each Comparison ranking a pair of Frameworks {F1, F2}:
    if resolve({F1, F2}) is determinate:             # §6.5, the same recursion impacts use
        defeated.add(dispreferred)
        live.discard(dispreferred)

winning_framework = live[0] if len(live) == 1 else None
```

Four things this pins.

- **A framework weigh defeats through `resolve`, not through attacker pruning.** Two frameworks
  usually do not attack one another; they clash through a `Comparison`, so there is no attacker set to
  prune and the v5 weigh-defeat machinery (which prunes `attackers_by_target`) finds nothing to do. The
  determinate weigh removes the dispreferred framework from the **live set**. This is the same
  `resolve` that decides impacts and link polarity, meta-weighs and recursion included. A framework
  weigh that is computed but never consumed is a **decorative weigh**, and is a bug of the same class
  as the pre-v5 decorative uniqueness weigh.
- **Selection is never order-dependent.** `winning_framework` is a function of the live set's
  **cardinality**, never of element index or iteration order (§2.1). It is an invariant, and the judge
  asserts it: `len(live) != 1` implies `winning_framework is None`. Selecting "the first surviving
  framework in iteration order" makes the judge a non-function of the round and is disqualifying for a
  reward function.
- **The framework channel has no magnitude floor.** §6.5's fallback is "decide the clash by DF-QuAD
  magnitude." Frameworks bear no magnitude (§3.6), so there is nothing to fall back to: an
  indeterminate framework clash (no weigh, symmetric weigh, or a weighing layer with no lone survivor)
  yields **no defeat**. σ *eliminates* (below threshold, not live) but never *ranks*. Two live
  frameworks with no weigh between them are both simply live. **The judge does not pick the stronger
  one.** Picking a framework nobody won is the intervention tabula rasa forbids, and σ is not a claim
  about which framework should govern.
- **Defeat removes a framework from the live set. It does nothing else.** In particular it does not
  exclude anything (§5.3).

### 5.2 The framework wash

`winning_framework is None` is a **wash**: **no gating.** Every chain is in scope, and the round
proceeds to impact weighing and the ballot exactly as an unframeworked round does.

**A wash is not a drain to presumption.** Weighing and BallotDirectives sit on the layers above content
and still do their work. A judge given no winning framework decides on the best-weighed piece of
offense; presumption remains only the ordinary backstop for `abs(N) <= epsilon` (§7). This is the
in-round behavior: when nobody wins framework, judges vote on the impact debate, they do not vote NEG
for want of a framework.

A wash arises when the live set has any cardinality but one:

| live | how | `FRAMEWORK_SELECT.via` |
|---|---|---|
| 0 | no frameworks were read | `no_frameworks` |
| 0 | every framework fell below σ threshold or failed maker-extension | `none_survived` |
| 0 | a cycle of determinate weighs defeated all of them (A>B, B>C, C>A) | `all_defeated` |
| 1 | one framework was ever live; no weigh needed | `sole_survivor` |
| 1 | a determinate weigh reduced the live set to one | `weigh` |
| >= 2 | multiple live, no determinate weigh separating them | `multiple_live` |

Only the two `live == 1` rows produce a governing framework.

### 5.3 The gate

A chain's **framework anchors** are the frameworks reachable **from its impact terminal(s)** by a
`Support` path whose interior nodes are never an `Advocacy` or a `BallotDirective` (§2.2, §4).
Anchoring is the relation *this impact is evaluable under this framework* — the impact reaching a
framework through the conductive part of a chain (the Uniqueness/Link/Impact spine). `Advocacy` and
`BallotDirective` are the two node types that are **not conductive spine**: an Advocacy is the shared
premise both sides litigate (§2.1), sitting *behind* the spine; a BD is terminal, sitting *past* the
framework. Neither conducts scope, so the anchor walk may **arrive** at one but may not **expand
outward** from it — they are **absorbing, not traversable** (arrival ≠ traversal). A chain may have
**several** anchors: an impact that supports *directly* into both util and social value anchors to
both.

**Uniqueness under v9.** Uniqueness nodes count as chain members for `anchor_members` and
reachability; a turn capturing a chain brings its wired uniqueness nodes into the capturing side's
anchor set. Superseded, see §12.5.

```
in_scope(chain) = (winning_framework is None)                     # wash: ungated
                  or (winning_framework in anchors(chain))
```

That is the whole exclusion rule. There is exactly **one** exclusion condition.

**Defeat is not exclusion.** Losing a framework weigh removes a framework from the live set; it does
nothing to the chains anchored to it. A chain anchored to a defeated framework is out of scope for
exactly one reason — it is not anchored to the framework that *won* — and that reason applies equally
to a chain anchored to nothing at all. So: if util is defeated but the impact also links into social
value, and social value wins, **the impact is in scope**. Rejecting a framework is not rejecting
everything that ever touched it. Reading "util is racist" is a reason to reject util; it is not a
reason to reject the argument that linked into util.

The framework argument does **not** depend on this. Its offense (§2) is anchored to the reader's **own**
framework, not to the framework it attacks, so it stands or falls on that anchor and is untouched
by the attacked framework's fate. "Util is racist" keeps generating offense whether util is live,
defeated, or absent — the framework argument's ballot weight lives under the reader's framework. The attack on
util is a **separate** `DefensiveAttack` on σ(util) whose only job is to unseat util from the live set
(§5.1). Scope and unseating are two edges doing two jobs; neither is the other.

Because exclusion requires a winner, `framework lock-out` (§7) is reachable **only** when
`winning_framework is not None`. A wash never locks anyone out.

Emit `FRAMEWORK_SELECT` (once, with `via` and the live set), `FRAMEWORK_DEFEAT` (once per defeated
framework), and `FRAMEWORK_GATE` (once per chain).

### 5.4 Framework liveness: maker-extension to gate

A framework must be extended by **its own maker**, through every one of that side's speeches from
introduction on (§6), to enter the live set and gate the round. *Going for your framework is
extending it.* A framework the maker abandons cannot gate. This mirrors the v4 attacker-liveness gate:
an attack applies only while its maker keeps it live, and an abandoned attack **lapses** rather than
being scored off the opponent's silence. This is a genuine gate **separate from σ** — a kicked
framework can sit at σ = 1.0 and still fail to gate, because its maker stopped carrying it.

There is no second, union-based liveness condition for frameworks. A framework serves no "premise"
role for any chain (§2): a chain reaches a framework only by support-anchoring, which is scope, and a
framework contributes no magnitude to any chain, so there is nothing for union-liveness to keep alive.
The only reason another side's stamp on a framework ever mattered was the discarded "rooted-at-premise"
model; under the two-edge framework argument it does not arise.

The kick comes out right from the one gate plus ordinary anchoring. AFF reads `F_util`. NEG reads
`F_sv` (its own framework) plus a framework-argument link whose impact anchors to `F_sv` and whose
`DefensiveAttack` targets `F_util`. AFF kicks `F_util` in the 1AR. `F_util` fails **maker**-extension,
so it leaves `live` and gates nothing — AFF cannot keep its own gate by dropping it. NEG's framework-
argument offense is anchored to `F_sv`, never to `F_util`, so it is entirely unaffected by whether `F_util` is
live, defeated, or kicked; it keeps its magnitude (`link × impact`; the framework contributes no
factor) and generates N. AFF cannot kick out of NEG's link either — the link is a spine rep NEG
extends. Both halves hold, and neither needs a framework-specific rule beyond the one maker-extension
gate.

### 5.5 Framework families (deferred to a milestone after v6)

**Status: specified, not yet built.** Families change what "the live set" and "winning framework"
*are* — they become sets rather than single nodes — which touches the exact selection code v6 repairs.
Building families onto a selection pass that does not yet route through `resolve` means debugging two
things at once. So the single-framework case (§5.1–§5.4) is the base case, landed and validated first;
families generalize it in a later milestone. This section fixes the design so it is ready.

A real framework is usually a **stack** of values, not one node, and losing one plank should not
collapse the stack. A **family** is a set of framework nodes composed by non-conjunctive aggregation
(losing a member weakens the family's reach but does not kill it — this is what rules out chain-style
composition, where any dead spine node kills the chain).

- **Identity.** A family is a **connected component of `Framework → Framework` `Support` edges**. A
  lone framework is a one-member family; the single-framework case is the base case, not a special
  case.
- **Liveness.** A family is **live iff at least one plank is live** (clears the σ threshold, §3.6, and
  its maker extends it, §5.4). Kicking or defeating one plank removes that plank; the family gates on
  while any plank stands.
- **Scope is the union of live planks' scopes.** A chain is in scope under a governing family iff it is
  anchored to **any** live plank of that family. This is the mechanism that makes "losing one node
  doesn't kill the framework" true: a dead plank simply stops contributing its slice of scope, and
  chains anchored to the surviving planks stay in.
- **Selection** is §5.1 lifted from nodes to families: build the live families, apply weigh-defeat
  through `resolve`, and `winning_family = the sole live family if exactly one, else None (wash)`.
  Still cardinality, never order.
- **Weighing operates at either grain, resolved by the existing recursion.** A `Comparison` may rank
  two **families** (which stack governs) or two individual **planks** (which value wins within or
  across families). When a plank-level weigh and a family-level weigh conflict, that is a **meta-weigh**
  — the same `resolve` recursion (§6.5) that already climbs meta-weighing levels handles it with
  nothing new in the weighing engine. The only generalization is that the objects `resolve` compares
  may be sets. A determinate plank weigh removes the dispreferred **plank** (which may or may not empty
  a family); a determinate family weigh removes the dispreferred **family** (all its planks). Which
  grain a given weigh targets is read from what its `Comparison` connects, not stamped.

Every §5.1–§5.4 invariant survives the lift: no magnitude floor (families bear no magnitude either);
defeat removes from the live set and excludes nothing directly; the single exclusion condition becomes
"in scope iff no winning family, or anchored to a live plank of the winning family"; lock-out still
requires a winner. Emit `FRAMEWORK_SELECT` with the winning family's members (or `null` + wash `via`),
and `FRAMEWORK_DEFEAT` per defeated plank or family.

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
- **No new offense in rebuttals** — checked **per root→impact path** (§3.3.1), not per component.
  A path establishes no new offense — and is dropped from the live carriers, exactly like an
  extension failure — if **any offense-bearing node on it** (a `Link` or `Impact`: the sign channel
  of §3.2; `Advocacy` and satellite `Uniqueness` carry magnitude, not offense, and are exempt) was
  **introduced in a rebuttal speech**. A sibling branch off a shared trunk built entirely in
  constructives is unaffected — the disqualification is scoped to the tainted path, and the OR/max
  aggregation over surviving paths (§3.3.1) is unchanged. (This supersedes the earlier chain-level
  rule that read the *component's earliest* member: grafting a fresh impact or link onto a chain
  rooted in a constructive is new offense in a rebuttal and does not count, even though the
  component's earliest node is a constructive.)
- Non-spine nodes need not be extended, with one exception: a **Framework** must be extended by
  its own maker to enter the live set and gate the round (§5.4). A framework serves no premise role
  for any chain, so it has no union-liveness condition — anchoring is scope, not a magnitude
  dependency (§2, §5.4).
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
rule. Magnitude is the **base case / floor**: it has no level above and never punts upward, so the
recursion terminates. For offense-bearing nodes the floor always yields a comparison. For
**frameworks** the floor is **empty** (§3.6): they bear no magnitude, so an indeterminate framework
clash yields *no defeat* rather than a δ ranking. Termination depends on the floor not recursing, not
on the floor producing a winner.

**Well-founded:** each step climbs to strictly fewer, higher nodes in a finite graph, and the
magnitude floor guarantees a base case — so `resolve` always terminates. Implement it as an actual
recursion, not a fixed depth-2 check (a fixed check is wrong at depth ≥ 2).

### The preference a weigh expresses (`favors`)

A weigh does not merely name a pair; it names **which member it prefers**. That preference is an
explicit **`favors`** pointer the Weighing node carries, aimed at one of the two nodes its
`Comparison` edges connect. `resolve` reads `favors` directly: the preferred member of the pair *is*
`favors`. This is what makes **own-side weighing** meaningful (Phase 0): a debater may weigh two of
their own impacts and say which controls, and the judge honors it — the preference is the pointer,
not the side.

**Legacy default (no explicit `favors`).** A round authored before the `favors` channel carries no
pointer; the judge derives the preference structurally, preserving prior verdicts byte-for-byte:
- **Cross-side pair** (exactly one member on the weighing's own side): `favors` defaults to that
  own-side member (an AFF weigh of {AFF-link, NEG-turn} prefers the AFF link). Identical to
  pre-`favors` behavior.
- **Same-side pair** (both members on the weighing's own side, or neither): no preference is
  derivable from side alone, so the weigh is **inert** — it expresses no clash-break and falls
  through to magnitude. Unchanged; a legacy same-side weigh does **not** become live.

`favors` is consumed only to identify the preferred member; everything else in §6.5 — survival,
meta-weighing, the determinate/indeterminate split, the magnitude floor — is unchanged. Pair identity
is still the two `Comparison` targets (direction-of-`Comparison`, §6.5 `weigh_pair`), orthogonal to
`favors`.

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
- **Frameworks (§5.1):** the same `resolve` ranks a framework pair. A determinate weigh **defeats** the
  dispreferred framework, removing it from the live set; indeterminate yields **no defeat**, because the
  framework channel has no magnitude floor (§3.6). This is not an exception to `resolve` — it is what
  `resolve` says when the floor is empty. A framework weigh must be **consumed** by framework selection;
  a weigh whose only trace is a `WEIGH` record is decorative and is a bug.
- **Channel-specific expression of weigh-defeat.** One rule ("a determinate weigh defeats the
  dispreferred member, and a defeated member does not attack the winner"); three expressions, which are
  deliberately **not** collapsed to one prune point. *Magnitude clashes* (uniqueness, defensive) prune
  the defeated attacker from the target's accrual. *Sign clashes* (link, turn polarity) hold the link's
  polarity by preference and drop the turn from the magnitude channel. *Scope clashes* (impacts,
  frameworks) **exclude**: a defeated impact's chain is excluded at the ballot; a defeated framework is
  removed from the live set. **Frameworks sit with impacts, not with uniqueness** — scope defeat
  expresses as exclusion, because scope is what a framework is.
- A `Comparison` over a **mixed-type** pair (a framework against an impact, say) has no clash to break
  and is **inert** (§3.4).
- Emit `WEIGH` (with the pair, the resolved preference or `symmetric`, and `via`).

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

applying won-weighing preferences to the comparison.

**Turned offense does not "emigrate" to the opponent's sum.** A turn is a **negative delta on the
introducing side's own sum**, scoring for the opponent only through an **opponent `BallotDirective`
anchored to that chain**; absent that BD it contributes **0** to the ballot sum. `delta_a` already
carries the polarity flip (§3.2, §3.5): a turned AFF chain has `sign = −1`, so its delta is a
**negative** term in the **AFF** sum above — never a positive term added into the NEG sum. It scores
for NEG only when a NEG BD is anchored to it and validates (a determinate turn → `NEG offense`); with
no such BD the turned chain is incident only to an AFF BD, which rejects opposite-side offense, so it
banks nothing and is orphaned to presumption (r10). The effect on `N` matches "the offense changed
hands," but the mechanism is a signed self-side delta gated on an opponent BD, not a literal transfer
into the opposing sum.

Then the **asymmetric win condition**. It is stated over **AFF-OWNED offense**, not AFF-*introduced*
offense. A chain's contribution belongs to the side its composed sign **favors** — its `owner`/favored
side (§3.5) — which is the introducing side for an ordinary chain and the **opponent** for a captured
(turned) chain. AFF can win on offense it **captured** from a NEG chain by turning it, exactly as NEG
wins on a captured AFF chain (§3.5, r4). A link turn is affirmative offense and carries a ballot even
when AFF has dropped its own advantage, provided AFF still defends the plan.

- **AFF wins** iff **all four**:
  - **a live AFF advocacy** — a reachable `Advocacy` node extended through AFF's speeches (§6). This
    reads the advocacy **node**, not a surviving advantage chain: AFF may drop its own advantage and
    still hold the plan, so the advocacy gate must not be tied to a validated AFF-side offense chain.
  - **a complete AFF-owned chain** — a BD-validated chain whose **favored side is AFF** (`owner == AFF`,
    **not** `side == AFF`) with non-zero surviving magnitude. A captured NEG disad (`side == NEG`,
    `owner == AFF`) satisfies this; a turned-**away** AFF chain (`side == AFF`, `owner == NEG`) does not.
  - **at least one in-scope AFF-owned impact** — that chain is in scope of the winning framework (§5),
    read over `owner == AFF` chains (the same owner-side quantifier as `framework lock-out`, below).
  - **and N > epsilon** — net offense favors AFF past the floor.
- **NEG wins** otherwise (presumption). NEG never satisfies these gates — it wins whenever AFF does
  not — because presumption is the tabula-rasa default and **AFF carries the burden**. So NEG capturing
  an AFF impact wins by the *absence* of an AFF win (AFF's offense turned away → gates unmet), needing no
  owner-side gate of its own; AFF capturing a NEG impact must still clear all four. The owner-side rule
  makes **whose offense counts** symmetric; presumption keeps **who bears the burden** asymmetric. A NEG
  mirror of the AFF-captures-disad round therefore resolves NEG by presumption / `NEG offense`, never by
  a NEG structural gate.

**Balloting a captured turn (§3.5 / r4).** A `BallotDirective` may anchor to **any node on the chain
it directs the ballot toward**, and a **turning link that captured a chain is on that chain** — so a BD
on the turning link anchors the captured chain, exactly as a BD on the captured impact does. Both are
legal authorings; neither is privileged. Mechanically the turning link joins the captured chain only by
an `OffensiveAttack`, which is not a chain-membership (spine) edge (§3.3) — union-find keys on same-side
`Support`, so the link is absent from the chain's `members`. Anchoring therefore reads a distinct
**`anchor_members`** set (= `members` ∪ the live turning links that captured a member), used **only** by
BD incidence; `union-find`, per-path aggregation (§3.3.1), and the chain's `side`/`sign`/`mag`/`owner`
continue to read `members` alone, so anchoring adds BD **reach** and never changes accounting. **No
walk crosses an `OffensiveAttack` edge** — `anchor_members` is a one-hop record set at chain-build time,
not a traversal; framework anchoring still uses the Support-only impact→framework walk (§5.3).

Anchor-membership is gated on **live capture**: a turning link earns anchor reach on the chain
containing the node it attacks **only if that node actually flipped** (`eff_pol == -1`). A **defeated or
washed** turn — one whose target kept its polarity — earns **no** anchor reach, even though it remains
in `offense_on` (which is populated pre-resolution and so is not by itself evidence of capture). Note
this gate is about **reach**, not **direction**: whichever side a BD claims, validation still checks the
chain's `owner` (`_favored_side`), so a BD anchored to a captured chain scores only if the chain's
offense favors that BD's side.

**The `N > epsilon` floor is load-bearing and unchanged.** Recognizing AFF-owned offense in the three
structural gates does **not** let a bare or washed turn win: a captured chain that nets to a tie or
worse (`N <= epsilon` — outweighed at the polarity clash, or cancelled by opposing NEG offense) fails
the floor, and NEG wins by presumption even though the owned chain satisfies advocacy + complete +
in-scope. The floor is the only gate that reads **net** offense; the other three read structure. (This
is what stops "any turn wins": the owner-side change adds a *path* to an AFF win, never removes the
net-offense requirement on it.)

The `reason_class` distinguishes *why* NEG won: `AFF structural failure`,
  `framework lock-out`, `NEG offense` (N < -epsilon), or `presumption`. The two indeterminate-looking
  labels are **disjoint** — no overlap, so no tiebreak is needed:
  - **`AFF structural failure`** requires that AFF **established** offense that then **failed**: a
    **complete, extended, still-AFF-favoring** chain (sign +1, resolved — not turned to the opponent)
    existed but was **driven to zero magnitude** or **gated out of scope**, so it reached the ballot
    contributing nothing. Offense was built, then lost. ("Advocacy present" is too coarse a proxy for
    "offense established" — a bare advocacy, or a chain whose impact was never resolved, has not
    established offense; the judge tests for the chain itself.)
  - **`presumption`** is every other NEG win at `abs(N) <= epsilon`: **no** such chain was ever
    established — offense never legitimately existed (an unresolved / no-window impact, or a chain that
    failed extension) — **or** a complete in-scope AFF chain did contribute but the round netted to a
    **tie** (`abs(N) <= epsilon`; offense reached the ballot, it just did not prevail).
  A **flip is never `AFF structural failure`.** When AFF's link is turned, the offense becomes the
  opponent's: it **scores for NEG** if that side anchored a `BallotDirective` (a determinate turn with
  a NEG BD → `NEG offense`), and is **orphaned to presumption** if no such BD exists (the turned chain
  is incident only to an AFF BD, which rejects opposite-side offense — nobody established scoring
  offense). A turned-away chain is not AFF's offense collapsing; it is offense changing hands or going
  nowhere. So the structural-failure collapse modes are *driven-to-zero* and *gated-out-of-scope*,
  **not** *flipped*.

  `framework lock-out` requires `winning_framework is not None` **and** no in-scope **AFF-owned**
  impact (the same owner-side quantifier as the win condition, so a captured disad in scope of the
  winning framework is *not* locked out — it reaches the ballot and either wins on `N > epsilon` or, if
  washed, drains to `presumption`, never to lock-out). A **wash** (§5.2) can never produce lock-out,
  because a wash gates nothing. A round that returns the
  right winner via the wrong `reason_class` is a **failing** round: the harness asserts on the
  `(winner, reason_class)` pair, never on the winner alone. A verdict that is correct by presumption
  where it should be correct by lock-out is a judge that will drift, and a winner-only fixture cannot
  see the difference.

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
| `WINDOW_CLOSED` | edge_id, attacker_id, attacker_speech, target_id, target_speech, window_close | — | 1 |
| `CHAIN` | chain_id, sign, mag, delta | side, extended, in_scope, collapse_reason, responsible | 3 |
| `FRAMEWORK_SELECT` | winning_framework_id \| null, live[], defeated[], via | reason | 5 |
| `FRAMEWORK_DEFEAT` | framework_id, weighing_id, preferred_id | — | 5 |
| `FRAMEWORK_GATE` | chain_id, impact_id, framework_id \| null, in_scope | anchors[] | 5 |
| `WEIGH` | weighing_id, outcome (resolved/symmetric), preferred_node, via | pair[], overrode, favors_source (explicit/legacy_default/null) | 5 |
| `BD_VALIDATE` | bd_id, result, reason | side | 6 |
| `BALLOT` | N, gates_passed[], winner | reason_class, aff_sum, neg_sum, decomposition[] | 6 |

`FRAMEWORK_SELECT.via` is one of `weigh`, `sole_survivor` (both yielding a winner) or `no_frameworks`,
`none_survived`, `all_defeated`, `multiple_live` (all washes). It is emitted **exactly once per round**,
including when there are no frameworks at all: a silent framework pass is how a decorative weigh hides.

**Chains are named by their impact terminal**, never by an incident node. Naming a chain by whatever
node the renderer reached first produces traces that call one chain a `Framework` and a structurally
identical one a `BallotDirective`, which is how a correct-verdict-wrong-reason round becomes
unreadable. This binds the RFD renderer as well as the trace.

---

## 9. Pass order (summary)

1. **Discovery** — enumerate BD-anchored chains.
2. **Drop + extension** — response windows; lock drops; cut un-extended chains.
3. **Node accrual** — DF-QuAD per node (leaves first) → surviving σ. (No polarity yet.)
4. **Weighing towers** — resolve each weighing sub-debate to determinate/indeterminate via `resolve`
   (§6.5); these depend only on their own drop/concession status, so they settle before clash
   resolution and cannot cycle with polarity.
5. **Clash resolution** — resolve same-type clashes (§6.5): determinate weigh decides, else raw δ.
   This is where **effective polarity** is set (a won link-weigh keeps polarity; else the 0.5 σ
   threshold). Then chain sign/magnitude products → delta per chain.
6. **Framework gate** — build the live set (σ threshold + maker-extension); apply framework
   weigh-defeat through `resolve` (§5.1); select the governing framework by live-set **cardinality**;
   gate chains (§5.3). Zero or many live → **wash**, no gating (§5.2).
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
   **NEG (AFF structural failure).** (A complete, extended, still-AFF-favoring chain was driven to zero
   magnitude — offense established, then lost — which is `AFF structural failure` per §7, not
   `presumption`. Earlier drafts wrote "presumption" loosely; §7 is authoritative.)
3. **Answered defense -> mitigation.** As (2) but AFF answers the defense in 2AC and wins it down, so
   the defender survives only at ~0.5; link sigma ~0.5, chain mag ~0.5 > epsilon. **AFF**, weakened.
4. **Link turn, extended.** NEG turns the 1AC link in 1NC; AFF concedes; NEG extends both the turn
   and the inherited impact through block + 2NR. Sign flips, NEG offense N < -epsilon. **NEG (offense).**
5. **Link turn, not extended.** As (4) but NEG fails to extend the inherited impact in 2NR. Chain
   fails §6, contributes 0. No NEG offense; AFF advantage also gone (turned). N ~ 0.
   **NEG (presumption).** (The AFF chain was turned away and the turn was not carried, so *nobody*
   established offense — the chain failed extension, so no complete AFF chain stood. A flip is never
   `AFF structural failure`, §7.)
6. **Framework lock-out.** NEG wins a framework that excludes AFF's only impact (no support path).
   AFF has zero in-scope impacts. **NEG (lock-out).**
7. **Weighing overrides raw delta.** AFF impact delta = 0.8, NEG impact delta = 0.6, but NEG wins a
   conceded weighing claim that its dimension controls. Preference honored over raw delta. **NEG.**
8. **No-window unresolved.** AFF introduces a new offensive answer in 2AR (final speech); NEG never
   had standing. Marked `UNRESOLVED`, establishes no offense. Decided on the rest of the flow.

### Framework-channel oracle rounds (§5)

Each isolates one clause of §5 and each asserts on `(winner, reason_class)`.

17. **Framework weigh lock-out.** AFF: advocacy → link → impact, impact anchored to `F_aff`, `F_aff`
    supports the AFF BD. NEG: mirror, anchored to `F_neg`. NEG authors a Weighing with a `Comparison`
    over `{F_aff, F_neg}` preferring `F_neg`. No attacks anywhere; all nodes live throughout.
    `resolve` is determinate for `F_neg` → `F_aff` defeated → `live = {F_neg}` → `F_neg` gates → AFF's
    chain is not anchored to it → no in-scope AFF impact. **NEG (framework lock-out).**
    *This round currently returns AFF, because framework selection never calls `resolve`.*
18. **Framework wash.** As (17) but **no weighing node**. Both frameworks live, `len(live) = 2`,
    `winning_framework = None`, no gating. Both chains in scope, both δ = +1.0, N = 0.
    **NEG (presumption)** — but by the *wash* path, with `FRAMEWORK_SELECT.via = multiple_live`, not by
    lock-out and not by any σ tiebreak. Give the two frameworks **different σ** (attack one down to
    0.7, still above threshold) and the verdict must not move: σ eliminates, never ranks.
19. **Dual-anchored impact survives its framework's defeat.** AFF's impact has support paths to **both**
    `F_util` and `F_sv`. NEG weighs `F_sv > F_util` (determinate). `F_util` is defeated; `live = {F_sv}`;
    the AFF impact is anchored to `F_sv` → **in scope**. **AFF (AFF offense).** Rejecting a framework is
    not rejecting the argument that linked into it.
20. **Framework argument: offense independent of the attacked framework.** AFF reads `F_util`. NEG
    reads `F_sv` (its own framework) and a framework-argument link ("util is racist") whose impact (racism) anchors
    to `F_sv`, and whose `DefensiveAttack` targets `F_util`. AFF **kicks** `F_util` in the 1AR (stops
    extending it); NEG never drives σ(`F_util`) below threshold. `F_util` fails **maker**-extension →
    leaves `live` → gates nothing; `live = {F_sv}` → `F_sv` gates; NEG's racism impact is anchored to
    `F_sv` → in scope, δ = `link × impact` (the framework contributes no factor). **NEG (NEG offense).**
    Two assertions carry the round: σ(`F_util`) appears in **no** chain product; and the verdict is
    **invariant to deleting `F_util` from the graph entirely** — the framework argument's offense does not depend on
    the framework it attacks, only on its own anchor `F_sv`.
21. **Permutation determinism.** Round (17) under N random permutations of the element list. Ballot and
    `reason_class` identical across all of them. *This is currently red: `winning_framework` is chosen
    by iteration order.*
22. **Offense aimed at a framework is inert.** An `OffensiveAttack` targeting a Framework node. Emit
    `INERT_ATTACK`; σ unchanged; no `POLARITY_FLIP`. Same treatment as offense aimed at an Advocacy
    (§3.4).
23. **Framework argument unseats by defensive attack.** As (20) but AFF **keeps** `F_util` extended
    throughout (no kick), and NEG's framework-argument link drives σ(`F_util`) **below** `POLARITY_THRESHOLD` via
    its `DefensiveAttack`. `F_util` leaves `live` on the **σ** path (not maker-extension); `live =
    {F_sv}` → `F_sv` gates; NEG's racism offense (anchored to `F_sv`) is in scope. **NEG (NEG
    offense).** Assert `F_util` left `live` with `FRAMEWORK_SELECT` reflecting `none_survived`-style σ
    elimination for it, and that the same link both attacked `F_util` and rooted the scoring chain (the
    two-hat spine rep). Contrast with (20): (20) unseats by maker-extension at full σ; (23) unseats by
    σ at full extension. Both must reach the same verdict by different mechanisms.
24. **Non-unique lands on the link.** AFF: advocacy → link → impact, no separate uniqueness node. NEG
    reads a `DefensiveAttack` **from a Uniqueness node directly onto the AFF link** ("inevitable
    regardless of your link"), conceded and extended. The link's magnitude drops; drive it to σ ≈ 0 and
    the chain dies. Assert the link is **dead, not turned** (sign stays +1, magnitude → 0, emit
    `MAGNITUDE` not `POLARITY_FLIP`): a uniqueness can mitigate a link but never manufactures offense
    from it. **NEG (AFF structural failure)** at σ ≈ 0 (a complete, extended, still-AFF-favoring chain
    driven to zero magnitude — §7, not `presumption`; the earlier "presumption" wording was loose), or
    **AFF weakened** at partial mitigation.
    *v9: superseded, see §12 — uniqueness is now a distinct per-post-world node, not "no separate
    uniqueness node"; this verdict is unchanged, only the construction.*
25. **Offense aimed at a uniqueness is inert.** An `OffensiveAttack` targeting a Uniqueness node (either
    drawn direction). Emit `INERT_ATTACK`; σ unchanged; no `POLARITY_FLIP`; the uniqueness is not
    turned. Same treatment as offense aimed at an Advocacy or Framework (§3.4). Confirms uniqueness is
    not turn-eligible.
    *v9: superseded, see §12 — uniqueness is now wired per post-world node; verdict unchanged.*

---

## 12. Schema migration: uniform uniqueness nodes (v9)

**This is a judge-semantics change → JUDGE_VERSION 8 → 9.** It invalidates cross-run
comparisons with any prior (v8 and earlier) version; every run must record the judge version it
was scored under. Pre-v9 oracle fixtures are not valid under v9 and are redrawn, not
reinterpreted, before re-ruling (migration process tracked outside this spec).

**Supersedes.** This section overrides the root-only uniqueness model wherever prior text assumes
uniqueness lives only at a chain root:

- §1's "impact = terminal node, positional only" still holds, but **every** post-world node now
  carries the uniqueness attack surface (§12.2.1), not only the terminal impact.
- §11 rounds 24–25 ("no separate uniqueness node"; a single uniqueness landing on the link) are
  re-expressed under the uniform model: uniqueness is always a distinct node, one wired per
  post-world node. Their **verdicts are unchanged**; only their construction is superseded.
- Any prior handling of uniqueness as a one-off root node (context in §3.5, §5.3).

### 12.1 Motivation

The pre-v9 schema treated uniqueness as a distinct node type reserved for the root of a chain.
This is a structural primitive that isn't supported by how uniqueness actually functions in
debate: non-uniqueness attacks are legal against any step in a causal chain, not only its origin.
Restricting the attack surface to the root encodes a strategic distinction (where non-unique
arguments may be run) that should instead be left for self-play agents to discover, per the
emergence-over-hardcoding principle.

This migration removes that restriction by making uniqueness a uniform, repeatable node type wired
to every post-world node in a chain, rather than a one-off root node. Pre-world is not a separate
field or a bare attribute — it is represented by the uniqueness node itself, which is why no
additional "pre-world node" type is needed.

### 12.2 Node types

#### 12.2.1 Post-world (mechanism) nodes

A post-world node represents a claimed state produced by one or more inbound mechanism edges.
Every post-world node in a chain — not only the terminal impact — has the same
three-attack-surface structure:

- **Non-unique**: attacks the wired uniqueness node. Argues the claimed post-world state would
  obtain independent of the mechanism.
- **Delink**: attacks the node directly. Argues the mechanism does not produce the claimed state.
- **Turn**: attacks the node's sign. Argues the mechanism produces the opposite of the claimed
  state.

Internal nodes carry link/probability strength; the terminal node in a chain additionally carries
impact magnitude. This is a positional attribute difference, not a difference in node type — all
post-world nodes share the same attack surface and evaluation pass regardless of position.

#### 12.2.2 Uniqueness nodes

A single reusable node type. One uniqueness node is wired to each post-world node, representing
that node's pre-world / status quo baseline. Uniqueness nodes:

- Are attackable via non-unique arguments (evidence that the status quo baseline does not hold).
- Do not themselves require separate delink/turn surfaces — they are the pre-world claim, not a
  mechanism claim.
- Follow the standard v4 binary extension rule with no type-based exemption (see §12.4).

Pre-world for node N is defined as: the uniqueness node wired to N. In the ordinary single-parent
case this will typically restate or depend on node N−1's post-world value, but it is represented as
its own node so it can be independently attacked, extended, and evaluated — not collapsed into an
attribute of the edge.

### 12.3 Edges and multi-parent structure

#### 12.3.1 Single-parent (advantage-style) chains

Standard case: node N has one inbound mechanism edge from node N−1, plus one wired uniqueness node.
Evaluation is a straightforward per-node pass down the chain.

#### 12.3.2 Multi-parent (disad-style) links

A disad link node has two inbound mechanism edges — one from the advocacy branch ("plan enacted")
and one from the uniqueness-of-trend branch ("status quo heading in direction D") — that combine to
produce the disrupted-trend post-world claim. This is an AND-join into the same per-node
evaluation, not a distinct node type or a distinct aggregation rule. DF-QuAD/QPN propagation logic
is unchanged; only the number of inbound contributions being joined at that node changes.

Cardinality ruling: one uniqueness node attaches to the joined post-world claim, not one per parent
branch. The post-world state is singular even though multiple mechanisms feed it.

**This multi-parent structure is one COMMON authoring of a disad Link, but it is no longer
mandatory (§2, rule 4).** A satellite uniqueness on a disad Link is OPTIONAL — the author draws one
when they want to pre-empt a non-unique, or omits it and lets the claim stand (tabula rasa) until
the opponent contests it (§3.4). What IS enforced is only that the disad's spine REACH the shared
AFF Advocacy over Support edges — directly (`link → advocacy`) or via the fusion idiom
(`advocacy → uniqueness → link`). A NEG Link that instead support-attaches DIRECTLY to an AFF Link
(e.g. "nuclear power solves warming AND causes meltdown risk": one causal chain, two terminal
impacts) still roots per rule 4, because the AFF Link itself roots in the Advocacy — so the disad's
spine reaches the Advocacy one hop further along. This is a **Support** edge ("builds on"),
categorically distinct from the turn (an **Attack** edge, "refutes", §3.4/§3.5) between the same two
Links. A scored NEG chain reaching neither an AFF Advocacy nor a Framework collapses `unrooted_disad`.

### 12.4 Evaluation order

Per-node evaluation, applied in topological order down the chain:

1. Check the node's wired uniqueness node. If it is zeroed (non-unique argument extended against it
   per §12.4.1), the node's inbound mechanism contribution is zeroed — and, per §12.4.3, if that
   node is a link feeding a shared impact, the impact's offense (for the chain's owning side) is
   zeroed across ALL live paths, not this node only.
2. Apply defensive attacks (delink) at this node.
3. Run QPN sign/magnitude propagation using this node's resolved value as input to the next node.

Zeroing a node's contribution via non-unique propagates forward through standard QPN
multiplication. For a DELINK this composes with per-path OR / max aggregation (§3.3.1): zeroing one
path's link does not affect a parallel independent path to the same impact (r29). A NON-UNIQUE does
NOT compose this way — it is a claim about a shared post-world STATE, not one causal pathway, and
poisons the convergent impact for its owning side across all live paths (§12.4.3).

#### 12.4.1 Extension rule for uniqueness nodes

Uniqueness nodes follow the same v4 binary extension rule as every other node. There is no
type-based or function-based exemption. A non-unique argument must be extended in the final speech
to zero its wired uniqueness node; if dropped, the uniqueness node is treated as standing and the
chain evaluates at full, non-zeroed value.

This ruling is scoped specifically to uniqueness nodes. It does not reopen or resolve the parked
general question of whether extended-but-unanswered defense requires re-mention to survive; that
question remains parked and is not settled by the "binary rule, no exemption" phrasing here.

#### 12.4.2 Final-speech gate, no liveness grandfathering

Contested-liveness (both sides actively engaging an argument) satisfies extension while speeches
remain in the round. It does not substitute for presence in the literal final speech. If an
argument — including a uniqueness node or a non-unique attack against one — was live earlier in the
round but is absent from the relevant side's final speech, it is dropped under the standard binary
rule regardless of how contested it was previously.

#### 12.4.3 Non-unique is state-level: convergence poisoning and the delink kick-out

Stated for a generic chain and its owning side; by the orientation principle (§2.2) it applies
identically whichever side (AFF or NEG) owns the chain and whichever side runs the non-unique
against it — the same way every other rule in this spec is side-agnostic.

A non-unique attacks a post-world state's uniqueness, not a single link. Because every link feeding
an impact converges on the SAME state, a non-unique that zeroes the wired uniqueness of ANY live
link feeding an impact (or the impact's own uniqueness) zeroes that impact's offense for the
chain's owning side in full — even when a parallel, independently-clean link also reaches it. A
second pathway to a non-unique state does not restore the state's uniqueness (r33, r32). This is
the one place §3.3.1's per-path OR / max aggregation does NOT apply.

**Two entry points, one gate (Ruling A, non-unique unification).** A non-unique reaches this gate by
either route, at the same threshold and with the same all-or-nothing effect: (i) attacking a link's
(or impact's) **satellite uniqueness**, driving *that* below threshold (the wired route above); or
(ii) attacking the **Link directly** — a `DefensiveAttack` from a Uniqueness onto a Link (§3.4,
r24), which poisons the link iff the *attacking* non-unique **survives** (σ ≥ threshold). Route (ii)
is diverted out of ordinary DF-QuAD accrual (it does NOT apply the proportional magnitude discount a
non-uniqueness defensive attacker would); it is state-level exactly like route (i), so it too
poisons a convergent impact across all live paths, subject to the same kick-out below.

**Kick-out (the only escape).** The chain's owning side may neutralize a non-unique by CONCEDING A
DELINK on the non-unique'd link, driving that link's σ→0. A severed (dead) link no longer feeds the
impact, so its non-unique has no live carrier to the shared state; the impact is then evaluated on
the remaining clean links with their own intact uniqueness (r31). A non-unique feeding an impact
zeroes it only while its link is live; delinked to σ→0 (or dropped by extension failure, §6), the
non-unique lapses with the link. A non-unique on the impact's OWN uniqueness has no such escape —
there is no link to sever — so it is unconditional (r32). This holds symmetrically: NEG kicks a
non-unique on its disad exactly as AFF kicks one on its advantage.

Reason class follows §7 for any complete, extended chain of either side driven to zero magnitude
(structural failure of the owning side's chain); it is never special-cased to one side.

### 12.5 Anchor and reachability accounting for uniqueness nodes

Uniqueness nodes count as chain members for `anchor_members`, per-path extension checks, and
reachability-from-captured-chain purposes. This follows from §12.4.1: since uniqueness nodes are
subject to the standard binary extension rule and can be dropped, they need the same
liveness/reachability tracking that governs whether any other chain member counts as extended.
Treating them as satellite nodes outside this accounting would be inconsistent — there would be no
mechanism for the judge to determine whether a uniqueness node was reachable and therefore eligible
to be extended in the first place.

Practical effect on the anchor-walk fix (§5.3): turning links currently join `anchor_members` of
the chain they capture. Uniqueness nodes wired to captured post-world nodes are included in that
membership, so a turn capturing a chain also brings its wired uniqueness nodes into the capturing
side's anchor set. This extends the existing fix's scope, not its logic — no modification to
union-find or aggregation is required.

## Open questions (parked; not settled here)

- **Unextended-BD gating of a disad's evaluation (raised on r36, 2026-08-12).** In `r36` the NEG
  disad's `BallotDirective` was **not extended**, yet the NEG disad is still evaluated (it loses
  on net offense — the verdict is AFF +1.000, hand-confirmed, and is unaffected either way). A
  **stricter judging philosophy** would hold that a disad whose ballot directive was never
  extended should not be evaluated at all — i.e. an unextended BD should gate its own chain out of
  the ballot, the way an unextended spine node gates the chain via §6. This is a genuine
  judge-semantics question about whether BD validation (§7) should carry a maker-extension
  requirement analogous to §5.4's framework-liveness gate. **Parked — the current judge does not
  gate on it, and it changes no adjudicated verdict on the corpus. Do not implement without a
  ruling.**
