# CDAF — progress & known issues

Running record of what's built and what's deliberately deferred. See
`docs/judge_spec.md` and `docs/extension_migration_spec.md` for the specs.

## Milestones done

- **Model / serializer / converter (E1)** — pure `model/`; per-node `liveness`;
  schema v2; lossless v1→v2 converter (dict-level).
- **Judge J1–J3 + RFD** — deterministic verdict + decision trace + templated RFD.
- **Extension migration (E1/E3a/E3b)** — extension is per-node liveness; builder
  renders glow/strip and authors via traced-path extend/un-extend.
- **Judge reads liveness (E2)** — the §6 extension check reads the liveness
  record (not edges); §4 final-speech refinement; legacy `Extension` type deleted;
  §11 verdict-oracle harness in `tests/oracle/test_oracle.py`.
- **Recursive weighing clash-resolution (judge v2, §6.5)** — weighing is the
  general same-type clash-breaker, resolved by one well-founded recursion
  (`judge/resolve.py`): meta-weighing → weighing → magnitude floor. Passes
  reordered (§9) so weighing towers settle before the polarity clash they decide
  (anti-cycle: `resolve` reads only the weighing tower, never main-chain polarity).
  A determinate link-weigh now saves a turned link. `JUDGE_VERSION = 2`.

## Known issues / deferred

### Framework σ-regime: no "weakened but live" framework in V1 (not a bug)

V1 accrual is attacks-only with τ=1.0 (§3.1): a live conceded attacker drives its
target to σ=0, and the `{s_k}` restorative-support slot is empty, so there is no
way to author a framework that is **weakened below full but still above the live
threshold while carrying live offense**. A framework is therefore effectively
binary in V1 — full σ, or σ≈0 (below threshold, not live). Consequence for the
oracle: the two framework-unseat mechanisms are isolated in **separate** rounds
rather than shown together at an intermediate σ — **r20** isolates the
maker-extension kick at **full σ** (no σ-attack drawn), **r23** isolates the
two-hat `DefensiveAttack` unseat at **full extension** (σ→0). Both reach the same
verdict by different mechanisms; the "attacked to 0.7, still above threshold"
picture from §11.18's prose is not cleanly constructible in V1. Closes when V2
populates the `{s_k}` restorative-support slot (§3.1), which makes intermediate σ
authorable. Not a bug — a documented V1 limitation.

### Builder must enforce Comparison direction (follow-up, not yet built)

`Comparison` is the one direction-bearing edge type (§2.2 carve-out): it must point
FROM the ranking weighing TO each ranked node, or the weigh has no well-formed pair
and is **silently inert**. The builder must enforce this when authoring a weigh
(create the edge weighing→member; reject/normalize a backwards draw) so a human
cannot draw a dead weigh and get a silent no-op. Judge-side this is already handled
(a malformed weigh is inert); the gap is the authoring UI. Do in a later E-stage.

### Turn offense (deferred to its own milestone)

### Turn offense (deferred to its own milestone)

**The judge cannot currently award NEG offense FROM a turn.** When NEG turns an
AFF link (an `OffensiveAttack`) and AFF concedes, two current mechanics combine to
make the turn *collapse the AFF chain* rather than *establish NEG offense*:

1. **Turns zero the target.** An offensive attack enters the target's DF-QuAD and
   drives its sigma toward 0, so the chain magnitude collapses to ~0 (delta ~0)
   instead of transferring at full strength to NEG.
2. **Extension is read node-side.** `node_extension_ok` checks each spine node's
   *own* side's speeches (the link's side is AFF), and AFF conceded — so the chain
   fails extension regardless of whether NEG carried the turn forward.

Consequently oracle rounds §11.4 (turn extended) and §11.5 (turn not extended)
both resolve to **NEG via the AFF chain's collapse** (`EXTENSION_FAIL` /
"AFF structural failure"), and are **not** distinguished as offense-vs-presumption.
The winner (NEG) is correct in both; the *reason* is not turn offense.
`test_r4_link_turn_collapses_aff_chain_neg` documents this explicitly and asserts
only the collapse, not turn offense.

**What awarding turn offense needs (out of scope until its own milestone):**
- **Offense-side extension** — check the chain against the side whose offense it
  establishes (for a flipped/`-1` chain, the opposing side), not the node's
  introduction side, so NEG carrying a turned AFF link counts.
- **Turn-preserves-magnitude** — an offensive turn should flip polarity while the
  magnitude survives (offense transfers to NEG at strength), instead of driving
  the target's sigma to 0.

These are downstream (DF-QuAD / chain / extension) changes and were explicitly
left unchanged in E2.

### §11 verdict labels vs §7 (RESOLVED — §7 redrawn, code proxy corrected)

§7's two indeterminate-looking labels used to **overlap** (a round with a partial
AFF structure but no established offense — e.g. round 8's unresolved 2AR impact —
matched both "AFF structural failure" and "presumption" with no tiebreak), and the
code used the coarse proxy *advocacy-present* for *offense-established*. Both are
now fixed:

- **§7 redrawn disjoint.** `AFF structural failure` requires a **complete, extended,
  still-AFF-favoring** chain (sign +1) driven to **zero magnitude** or **gated out
  of scope** — offense established, then lost. `presumption` is every other NEG win
  at |N|≤ε: no such chain ever established (unresolved / no-window impact, or failed
  extension), a chain **turned to the opponent** (a flip is never structural failure
  — it becomes NEG offense with a NEG BD, else orphaned), or a complete in-scope
  chain that merely tied.
- **Code proxy corrected.** `_reason_class` now tests for the chain itself
  (`side==AFF ∧ extended ∧ not unresolved ∧ sign==+1`) instead of advocacy-present.
- **Oracle re-labelled per §7:** r2/r13/r16/r24 → "AFF structural failure" (built
  then collapsed); r5/r8/r10 → "presumption" (turned away / never established); §11.2
  and §11.24's loose "presumption" wording tightened to match §7.

The r10 finding that drove the "a flip is never structural failure" clause: a turned
chain is validated against the flip-favored side's BD when that side authored one
(round 4 → NEG offense); round 10 authors only an AFF BD, so the NEG-favoring turn is
correctly orphaned → presumption. Not a turn/BD bug — the ballot re-anchors correctly.
