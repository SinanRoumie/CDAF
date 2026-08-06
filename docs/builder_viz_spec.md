# docs/builder_viz_spec.md — Builder & Visualization (Model C, Stage E3)

E3 reworks the dashboard's authoring and display for per-node liveness (Model C). It replaces the
old speech-column extension scheme (a node re-appearing in a later column) — that encoding is gone,
because extension is now state on the node (`extension_migration_spec.md`). E3 is builder-and-view
only: it changes how rounds are drawn and shown, and must produce and read the **same** liveness data
the E1 model and E2 judge use — never a parallel representation.

Prereqs: E1 (model + converter) done; E2 (judge reads liveness) done. Read `extension_migration_spec.md`
and `judge_spec.md` §2 first.

---

## 1. Layout model (hybrid: time on X, argument structure on Y, persistence on the node)

- **Horizontal position = introduction speech.** A node anchors at the speech it was introduced in
  (1AC left → 2AR right), as **soft bands**, not hard columns. A node **never moves** after
  introduction; its later life is shown by glow/strip, not by drifting right or re-instantiating.
  This holds for **every** node including the advocacy (introduced in a constructive like anything
  else — it is not a special root).
- **Vertical = argument structure (intra-speech).** Within a speech, an argument branches: the
  advantage/disad progresses **horizontally** toward its impact, and the argument's internal parts
  (advocacy, uniqueness, link, impact) fan **vertically** off that spine. No reserved lanes — branches
  fan freely.
- **Two branch kinds, kept distinct (do not conflate):**
  - *Argument branches* — structural, within a single speech (advocacy/UQ/L/! fanning). Vertical.
  - *Clash branches* — temporal, spanning speeches (answer / extend / turn across the round). Carried
    by liveness + edges, **not** by position.
- **Speech bands are an authoring scaffold.** In the **builder**, speech bands are firmer — they tell
  you which speech you're authoring into. In the **read-only view**, they are soft left-to-right
  placement zones; branches fan freely and time is read from glow/strips. Builder and view share the
  same layout, bands just firmer in the builder.

---

## 2. Persistence rendering — glow (always) + strip (on select)

Persistence is shown **on the node**, never by motion or duplication.

### 2.1 Glow — ambient layer, always on
- Every node carries a glow whose **brightness encodes how close to that side's FINAL speech the node
  stays live** — not raw speech count. Reaching the side's last speech (AFF 2AR / NEG 2NR) is the
  strategically meaningful thing; a branch that dies in the 2NR vs. one carried to the 2AR must read
  differently.
- **No glow = dropped** (not live past introduction / liveness ends early).
- Glow is driven by the node's **liveness record**, which is **side-agnostic**: a node kept alive by
  the *opponent's* turn glows because it is live, even though its introducing side abandoned it. Glow
  follows whoever sustained liveness, never "did its own side extend it."
- A collapse reads at a glance: the carried branch blazes toward the right/final speech; dropped
  branches fade.

### 2.2 Strip — diagnostic layer, on select
- Selecting a node reveals a **liveness strip**: seven cells in `SPEECH_ORDER`
  (1AC, 1NC, 2AC, 2NC/1NR, 1AR, 2NR, 2AR), each showing that speech's status:
  **filled-contested / filled-conceded / empty-not-live**.
- The strip is the debugging read — it shows *which* speeches (and gaps: e.g. live 1AC–2AC then dead)
  and contested-vs-conceded per speech, which glow (a scalar) cannot. This is the view that would have
  made the NSDA link's `{1AC, 2AC}` gap obvious.
- Default is glow-only for a clean canvas; the strip appears on select.

### 2.3 Retained encodings
- Side by node **border** (AFF green / NEG red). Type by node color/shape. Legends retained.
- Strip cells ordered by `SPEECH_ORDER` (from `model/speeches.py`).

---

## 3. Authoring interactions

- **Extend (the core new act):** select a path (advocacy→…→impact, a turn's spine, or a single
  defensive node) and a speech, and **stamp** — the act writes that speech into each selected node's
  liveness record. One atomic act over the chosen path; cannot be half-done. This replaces drawing
  re-assertion nodes + ExtensionEdges entirely.
- **Collapse** is not a separate feature: you extend the paths you keep and don't extend the ones you
  drop; dropped paths' liveness simply stops, and their glow fades. The theory case (shell with many
  impacts → one) is just extending one path forward while others end.
- **Status (contested/conceded)** is not hand-set: it is derived from whether an opposing attack
  exists against the node that speech (same rule as the converter, `extension_migration_spec.md` §7).
  The builder may show it; the judge/derivation owns it.
- **Re-engagement** is allowed: a side may extend/answer a node it had stopped extending once it is
  live again.
- **Retire the ExtensionEdge tool** from the palette. Remove the ExtensionEdge type from the model
  now that nothing authors it (final step of the migration). Existing v1 files still convert on load.
- Node creation, edge creation (support/attack/comparison), and selection/inspection otherwise as
  today, adapted to the new layout.

---

## 4. Invariants (do not break)

- Builder output and view input are the **same liveness data** the E1 model serializes and the E2
  judge reads. No parallel representation. A round built here, saved, and judged must be identical to
  the same round loaded from JSON and judged.
- `app/` imports `model/` and `judge/`; nothing in `model/`/`judge/` imports `app/`. Judging a live
  graph uses the same convert→judge path as loading a file.
- The judge is not re-run on edits; judging is the explicit "Judge Round" action (existing).

---

## 5. Staging (E3 is large — build in two prompts, gate each)

| Stage | Build | Gate |
|---|---|---|
| E3a | New layout engine (intro-speech X anchor, vertical argument fan, soft bands) + node rendering (glow always, driven by liveness-to-final-speech, side-agnostic; strip on select) | load a converted round (originally NSDA24Finals, since deleted from the corpus): nodes sit at intro speech, glow reflects liveness depth, selecting a node shows its 7-cell strip; a node kept alive by an opponent turn glows |
| E3b | Extend/collapse interaction (select path + speech → stamp liveness) + retire ExtensionEdge (palette + model) | build a small round, extend one branch to 2AR and drop another; glow/strips update; Save→Load→Judge round-trips; judging the built round == judging its saved JSON |

E3a is display-only (safe to build and eyeball). E3b changes authoring and finalizes the migration.

---

## 6. Acceptance (the payoff)

After E3 you can hand-build a surgical oracle round in the builder — a clean AFF chain extended to
the 2AR, NEG dropping everything — see the glow confirm the extension visually, judge it, and get
AFF at magnitude 1.0. That round, saved, becomes a fixture. The visualization is what lets you
*confirm the argument you built is the argument you meant*, which is the thing that was missing when
extension was invisible.
