# docs/extension_migration_spec.md — Per-Node Liveness (Model C)

This specifies the migration of extension from a structural mechanism (re-asserted nodes joined by
`ExtensionEdge`) to per-node state (each node records the speeches it was carried through). It is the
analogue of `judge_spec.md` for this change: pin the decisions first, review, then build in stages.

Motivation: extension-as-node made liveness something the judge *reconstructs* by walking edges
across duplicated nodes, so a single un-drawn re-assertion silently broke a chain (the NSDA24Finals
failure). Under Model C, liveness is state the judge *reads*. Extension becomes a single act that
cannot be half-done, collapse becomes the natural divergence of per-node records, and the resource
model gets a clean attachment point.

---

## 1. Pinned decisions (review these first)

1. **Liveness is per-node state.** A node keeps its introduction `speech` (unchanged) and gains a
   **liveness record**: which speeches it was asserted live in.
2. **Extension is an act, not an object.** Extending walks a chosen path and stamps the current
   speech onto each spine node's record. No new nodes, no edges. Atomic over the path — cannot be
   half-done.
3. **Liveness is side-agnostic.** A node stays live as long as *any* live argument routes through
   it, regardless of which side introduced it. A NEG turn sustains the liveness of the AFF link it
   turns; AFF abandoning that link does not kill it while NEG carries it.
4. **The unit of extension is a root-to-impact path.** A branchy argument (one advocacy/uniqueness,
   several links/impacts) is extended one path at a time. "Extend the whole argument" = extend each
   path. Collapse = stop extending the paths you drop.
5. **Shared trunk nodes live by union.** A node shared across paths (advocacy, uniqueness) stays
   live as long as any live path through it is extended; it dies only when the last such path dies.
   Dropping one path never un-stamps a shared node another live path still needs.
6. **Re-engagement is allowed.** A side may extend/answer a node it had stopped extending, once the
   node is live again (e.g. because the opponent turned it and carried it forward).
7. **Rich per-speech status.** The record stores, per live speech, not just "live" but the node's
   status that speech: **conceded** (standing unanswered) or **contested** (under active clash).
   This one field does triple duty: the resource-model cost hook, the conceded/contested
   distinction whose absence caused the original bug, and the new-argument threshold (§4).
8. **`ExtensionEdge` retires.** Its only job — "this re-asserts that" — becomes the liveness record.
   Genuine re-assertion-with-addition decomposes into a stamp (carry the node) plus an ordinary new
   node + support edge (the added warrant). The `ExtensionEdge` type is removed from the model.
9. **New offense in a final speech is gated by the no-window rule, refined by status (§4).**

---

## 2. Data model change

A node currently carries `label`, `side`, `speech`. Add one field:

- `speech` — **unchanged**: the introduction speech (first entry into the round).
- **`liveness`** — a mapping from speech label to status, e.g.
  `{"1AC": "contested", "2AC": "conceded", "1AR": "conceded", "2AR": "conceded"}`.
  Introduction stamps the first entry automatically. Each extension act adds an entry. Status per
  entry is `contested` or `conceded`, set by whether an opposing attack was live against the node
  that speech (the judge's drop detection already computes this; the builder can default new stamps
  and let the judge annotate).

Notes:
- `speech` stays for backward compatibility and because "introduction speech" is still meaningful
  (the no-new-chains-in-rebuttals rule reads it).
- Order liveness by `SPEECH_ORDER` index, never by insertion order.
- Edges no longer include `ExtensionEdge`. All other edge types are unchanged.

### 2.1 Advocacy as shared premise (structural commitment)

The advocacy is the shared subject both sides litigate, not a truth-claim either side flips. This
pins three things (all consistent with inert-not-illegal and role-from-structure):

- **Offense chains depend on the advocacy by a support-type edge, not a side-typed edge.** Both an
  advantage (AFF) and a disadvantage (NEG) attach their root (uniqueness) to the advocacy as a
  premise. There is **no** "advantage edge" / "disad edge" type — that distinction is redundant with
  data the model already carries and would create a second, conflictable source of truth.
- **Advantage vs. disadvantage is derived, never stamped.** An advantage is AFF-owned offense rooted
  in the advocacy; a disad is NEG-owned offense rooted in the advocacy. The judge reads side off the
  nodes and sign off polarity, routes δ into the AFF or NEG sum accordingly, and they meet at net
  offense / weighing. The advantage/disad character emerges from side + sign the way link vs. impact
  emerges from position.
- **Advocacy is not a valid OffensiveAttack target.** You outweigh a proposal; you do not turn it.
  An offensive edge into an advocacy is **inert** (contributes nothing), exactly as offense aimed at
  a pre-world uniqueness node is inert. A disad does not attack the advocacy — it grants it and
  builds competing offense off it, resolved at the impact/weighing level, not by an edge into the
  advocacy.

Burden consequence (derived, not encoded): AFF must establish net-positive offense rooted in the
plan; NEG wins by attacking that offense (defense/turns at link/impact level) **or** by rooting
competing offense in the same shared premise (a disad). Neither route is an edge into the advocacy.

---

## 3. The judge — §6 rewrite (reads records, not edges)

Replace the edge-walking extension logic with record reads.

**Complete extension (binary, total over spine).** A BD-anchored chain counts iff every spine node's
`liveness` contains **every one of that node's side's speeches from its introduction onward**. AFF
side-speeches: 1AC, 2AC, 1AR, 2AR. NEG side-speeches: 1NC, 2NC/1NR, 2NR. A gap in any spine node's
record ends the chain's eligibility as of that gap. Emit `EXTENSION_FAIL` naming the node and the
missing speech.

**No new chains in rebuttals.** A node whose *introduction* speech is a rebuttal cannot anchor
offense. (Reads `speech`, not liveness.)

**Side-agnostic liveness.** When checking whether a node is live in a speech, its record is the union
of every party's extensions through it. A turn's spine stamps the turned node live even though the
introducing side stopped extending it.

**Shared-node union.** A trunk node's liveness is sustained by any live path through it; the judge
never needs special handling — the node's record simply keeps getting stamped while any live
extension includes it.

Everything downstream (DF-QuAD accrual, chain product, δ, weighing, ballot) is unchanged. Only the
source of "is this live" changes: record lookup, not edge traversal.

---

## 4. New-argument handling (final-speech offense)

This refines, not replaces, the existing no-window rule.

- **Baseline (unchanged):** a node introduced in the final speech of its side has no response window
  (no opposing speech follows), so it cannot establish offense — `UNRESOLVED`, inert. Drawing it is
  allowed; it simply counts for nothing. (Oracle round 8.)
- **Refinement — legitimate continuation:** a final-speech node *does* engage if it attaches (attack
  or support edge) to a node that was **contested and in play entering that speech** — i.e. the
  attachment point's `liveness` for the immediately prior opposing speech is `contested`. Answering a
  turn qualifies, because the opponent kept that clash actively in play.
- **Conceded-live is NOT sufficient.** A node that is live only by concession (status `conceded`, not
  `contested`) is present but is not a legitimate site of *new offense* in the final speech. AFW
  cannot convert a conceded piece of defense into new terminal offense in the 2AR — that offense had
  to be established earlier, while the opponent had standing to answer. Spiking a conceded link into
  an impact in the 2AR is inert.

The test is structural and content-blind: *was the attached clash actively contested entering this
speech?* Turn-answer -> yes. Conceded-link spike -> no. New argument from nothing -> no. The judge
reads this off `liveness` status; it never reads the node's label.

Trace: emit `UNRESOLVED` for inert final-speech offense; a continued clash resolves normally.

---

## 5. The one-time converter (existing rounds)

Existing saved rounds encode extension as duplicate nodes joined by `ExtensionEdge`. Migrate them
losslessly:

1. For each `ExtensionEdge`-connected component of same-claim nodes, collapse to a **single** node.
2. Its introduction `speech` = the earliest speech in the component.
3. Its `liveness` = the set of all speeches present in the component (status defaulted, then
   re-annotated by the judge on load, or defaulted to `contested` where an opposing attack existed
   that speech and `conceded` otherwise).
4. Rewire any edges that pointed at a collapsed duplicate to point at the single surviving node.
5. Discard the duplicates and all `ExtensionEdge`s.

The old structure contains exactly the information the new one needs, so this is a re-encoding, not a
guess. Bump the schema version. Expect rounds to shrink substantially (NSDA24Finals from ~84 nodes).
Re-encoding NSDA24Finals is also the first correctness check — the extension bug may resolve here.

---

## 6. Build order (staged; each its own prompt and gate)

Strict dependency order — correctness is proven before the UI is touched.

| Stage | Build | Gate |
|---|---|---|
| E1 | Model field (`liveness`) + serializer round-trip + version bump + the §5 converter | existing saved rounds (incl. NSDA24Finals) load, convert, re-save; graphs shrink; round-trips clean; `ExtensionEdge` gone |
| E2 | Judge §6 rewrite (record-based extension) + §4 new-argument threshold | extension unit tests pass (clean chain counts; a gap fails; side-agnostic turn keeps opponent node live; shared-node union holds); new-argument tests pass (contested continuation counts, conceded-live spike inert, new-from-nothing inert) |
| E3 | Builder: extend act = select path + pick speech + stamp; nodes visibly show live speeches; collapse is visible on the graph | build and collapse a branchy argument in the dashboard; liveness renders; judging it matches E2 |

E1 imports nothing app-side and keeps `model/` pure. E2 keeps the judge pure. E3 is app-side only.

---

## 7. Resolved decisions (were open; now locked)

- **Converter status defaulting:** infer `contested`/`conceded` per speech from whether an opposing
  attack existed against the node that speech — `contested` if so, `conceded` if not. Not a flat
  default.
- **Rich liveness now:** `liveness` stores per-speech status strings from the start (not a bare
  speech set). It is the resource-model hook and the §4 threshold input, and migrating the schema
  twice is worse.
