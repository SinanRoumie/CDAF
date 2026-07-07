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

## Known issues / deferred

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

### §11 verdict labels vs §7 (documentation, no code change)

Per `judge_spec.md` §7, an AFF chain that *existed and collapsed* is an **"AFF
structural failure"**, distinct from **"presumption"** (no AFF offense ever
existed). The judge follows §7. §11's worked rounds use the looser word
"presumption" for some collapses (e.g. round 2: a conceded terminal defense
collapses the AFF chain → the judge reports "AFF structural failure", which is the
§7-correct label). §11 can be tightened to match §7's vocabulary; no code change
is needed.
