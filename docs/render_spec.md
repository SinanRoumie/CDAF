# CDAF Render Spec v1.5

Status: **accepted, v1.5**. All semantic rulings closed. M0 built and committed at v1.3; M1 built and validated at v1.5.

v1.1 amended RS10, RS11, RS13, RS16, RS27, and O3 after a code survey found RS10 described a function the judge does not have.

v1.2 amended RS10, RS11, RS13, retired RS11b, and closed O2 and O4 after a corpus survey found the v1.1 register predicate structurally incapable of classifying cross-side-rooted NEG offense.

v1.3 restated RS10c, split RS13, added RS13c, and closed O5 after the RS10c check caught a judge-internal inconsistency about whether a BallotDirective conducts.

v1.4 retired RS13 and RS13b entirely after the v1.3 assert fired on a legal post-mask round.

v1.5 adds RS5b (advocacy polarity), rules impactless material flat (RS16d), and closes O1 after M1 produced readable prose across three resolutions. Corrections are marked inline as *correction notes* rather than silently overwritten.

Rule IDs are prefixed `RS` to avoid collision with judge spec rules (`r6`, `r26`, `r27`, …). Sub-rules use the `a`/`b`/`c` suffix convention established in `RULINGS_v9.md` (V1a/V1b).

Open items in §12 are empirical, not unresolved: they require rendered output to answer and cannot be settled by ruling.

---

## 1. Scope and role

The renderer converts a completed or partial CDAF graph into a readable round transcript. It is an **inspection and publication artifact**. It is not part of the environment, not part of the reward, and not part of any training loop.

**RS1.** The renderer is a pure function of graph state. It has no state of its own across speeches beyond the frozen content of prior speeches and the established flow order (§4).

**RS2.** The renderer is a *frozen* model at temperature 0. It is never fine-tuned, never updated mid-experiment, and never consulted by the policy, the environment, or the judge.

**RS3.** Rendering is post-hoc. A round is played to completion (or to the speech being inspected) before any rendering occurs. Rendering never gates, filters, or influences move selection.

### 1.1 Non-goals

- The renderer does not evaluate arguments. It has no notion of who is winning.
- The renderer does not repair graphs. Incoherent topology renders as incoherent prose; that is diagnostic output, not a defect.
- The renderer does not select moves. Content-aware action selection is explicitly out of scope and is rejected on throughput, determinism, and tabula rasa grounds (§9.1).

---

## 2. Governing constraints

### 2.1 Type blindness

**RS4.** The renderer's permitted inputs are exactly:

- node type
- edge type
- side (AFF/NEG)
- speech index
- local neighborhood (incident nodes and edges)
- anchor component membership (§3)
- frozen content of prior speeches

**RS5.** The renderer receives no argument-type label. There is no "kritik" flag, no "theory" flag, no "disadvantage" flag. Type blindness is load-bearing for the structural unity claim: if a type-blind renderer produces text a coach would recognize as a kritik versus a policy disad, that is evidence *for* structural unity. If the renderer requires the label to produce recognizable output, that is evidence *against*, and the negative result is itself worth reporting.

**RS5b.** Advocacy polarity is derived from side, and side alone. An Advocacy node on AFF affirms the resolution; an Advocacy node on NEG opposes it. This is stated explicitly in the render prompt, not left to inference.

A NEG Advocacy is a legitimate move — a counterplan or an alternative. The node is legal; only its polarity is constrained.

Rationale: side is already a permitted renderer input under RS4, so this adds no information channel and no type label. M1 v1 left polarity to inference and got it wrong inconsistently: in one round a NEG orphan advocacy rendered as advocating the AFF's position. Policy resolutions mask the defect because "advocacy" defaults to "the plan," a shared referent both sides can point at. A value resolution removes the default, and an unanchored NEG advocacy must then be derived from side alone.

**RS6.** Spec and code use *offense anchored to X*. The terms "disadvantage", "advantage", "kritik", and "theory shell" do not appear as structural identifiers. They are emergent descriptions of rendered output, not properties of the graph.

### 2.2 Judge blindness

**RS7.** The renderer sees graph state as of the end of the speech being rendered. It receives no judge evaluation, no scores, no presumption state, no outcome, and no future speeches.

Rationale: a transcript that leaks the result cannot be read as a round. Judge reasoning, where wanted, is a separate annotation layer composed over the finished transcript — never renderer input.

### 2.3 Content immutability

**RS8.** Content is frozen at the speech in which it is generated. No later speech mutates the content of an earlier node or edge. A debater cannot rewrite the 1AC in the 2AR.

**RS9.** Rendering is speech-atomic. All nodes and edges introduced in a speech are presented to the renderer together, conditioned on frozen prior content, before any content for that speech is generated.

---

## 3. Registers

Content domain is determined by anchoring, not by type.

**RS10.** The renderer performs no *duplicate* analysis: it never recomputes anything the judge computes. Where the judge has no corresponding concept, the renderer may walk the graph, but only by borrowing an existing judge traversal's semantics exactly.

Register derivation is three reads:

1. **Component** — `resolve_chains` (`judge/passes.py:957`). Union-find over same-side Support edges. Advocacy and BallotDirective are ordinary traversable members here; the union unions *through* them. Chains are grouped into components by shared union-find root (see RS16b).
2. **Framework reach** — `_framework_anchors` (`judge/passes.py:1277`). Directed walk seeded from the component's impact terminals in which Advocacy and BallotDirective are absorbing — arrived at, not expanded through. Returns reachable Framework node ids.
3. **Advocacy reach** — RS10b. Not a judge concept; computed by the renderer.

**RS10b.** Advocacy reach is `_framework_anchors`' traversal with the collector changed. Identical seeding (component impact terminals), identical absorption (Advocacy and BallotDirective arrived at, not expanded through), identical side-agnosticism. It collects reachable **Advocacy** node ids instead of Framework node ids. Nothing else differs.

*Why not `_neg_offense_rooted`.* `_neg_offense_rooted` (`passes.py:784`) closes the RS13 rooted-at-all question exactly, but cannot do RS11 for two independent reasons. It conflates advocacy-rooting with framework-rooting into one boolean, and it is asymmetric by construction — it searches for an *AFF* advocacy specifically, so AFF components cannot use it. Deriving advocacy-reach as `rooted=True AND frameworks=empty` would work on the present corpus and fail on shapes not yet generated: the two walks have different semantics, and a NEG Advocacy is traversable in `_neg_offense_rooted` but absorbing in `_framework_anchors`.

**RS10c.** RS10b must mirror `_framework_anchors` exactly — identical seeding, identical absorption, identical side-agnosticism, differing only in what it collects. Divergence from `_framework_anchors` is drift and is a hard failure.

Divergence from `_neg_offense_rooted` is **expected, not a failure**, and is recorded rather than corrected.

**BallotDirective absorbs for register.** A BallotDirective is a ballot instruction, not an inferential link. Reasoning *through* one treats "vote NEG" as a premise. This follows the BD-trap precedent, which already ruled BD absorbing for anchoring purposes.

*Correction note.* v1.2's RS10c required `advocacy_reach ∪ framework_reach` to agree with `_neg_offense_rooted` for NEG components, and made disagreement a hard failure. Corpus survey found 3 disagreements, all traced to two structural differences intrinsic to the judge rather than to renderer drift: `_framework_anchors` absorbs at BallotDirective while `_neg_offense_rooted` walks through it, and RS10b seeds from terminal impacts while `_neg_offense_rooted` seeds from all members. The check did its job — it surfaced that two judge traversals disagree about whether a BD conducts — but the failure it detected was not the renderer's.

The seed-set difference is downstream of the same ruling: under absorption, a root reachable only through a non-terminal BD member is correctly not reachable.

*Judge-side finding, out of scope for rendering.* `_neg_offense_rooted` (`passes.py:784`) treats BallotDirective as a conductor, which appears inconsistent with the BD-trap precedent. This affects `unrooted_disad` collapse decisions and is logged here for the judge workstream. It is not addressed by this spec and no judge file is modified on its account.

*Correction note.* Earlier drafts of RS10 described a single union-find "from the impact terminal over Support paths whose interior nodes are never Advocacy or BallotDirective." That rule is real but governs **BallotDirective anchoring**, not register. It was misapplied here. No single judge function performs it for register purposes, and none should be written; register requires both steps above plus the precedence rule in RS11.

**RS11.** Register keys on **reachability**, not membership, and is side-agnostic. Advocacy takes precedence.

| Condition on component | Register |
|---|---|
| Advocacy reach (RS10b) non-empty | Substantive offense about the advocacy's consequences |
| Advocacy reach empty, framework reach non-empty | Reason to prefer that framework |
| Both empty | — (RS13 assert) |

One rule, both sides, no special-casing. AFF components reach their own advocacy directly. NEG offense reaches the AFF advocacy through a cross-side Support edge. A component reaching both renders substantive; a component reaching only a framework renders as an advantage of reading that framework.

Register is total: every legal component derives exactly one register.

*Correction note.* v1.1 keyed register on "Advocacy in the component's `members`." Membership comes from a same-side union-find, and a disad is NEG offense *about the AFF's advocacy*, so the advocacy a disad roots into is always cross-side and never in its own component. The v1.1 predicate was therefore structurally incapable of classifying a NEG disad correctly — not merely inaccurate at the margin. Corpus survey confirmed: `r36` and `shape2_link_turn` are scored by the judge as live rooted NEG offense (`collapse_reason=None`, extended, sign +1, magnitude 1.0) while v1.1 classified them unanchored. 32 of 178 post-mask run exports carry the same shape.

This is a correction to the implementation of Ruling 4, not a change to it. Ruling 4 holds that a chain not connected to the advocacy is not a disad to the advocacy. These chains *are* connected, via cross-side Support. RS11 v1.1 failed to see the connection; the ruling always covered them.

**RS11b.** *Retired in v1.2.* RS11b held that a NEG component must not acquire substantive register by unioning into the shared Advocacy. That is backwards: cross-side rooting into the advocacy is precisely what makes NEG offense substantive. The property it guarded was the wrong one. Superseded by RS10c, which checks the renderer's walk against judge semantics rather than forbidding a legitimate shape.

The survey result RS11b prompted remains valid on its own terms: zero components span both sides across 47 fixtures, confirming the same-side union gate at `passes.py:980`.

**RS12.** Framework-anchored offense is available to both sides and is not a disadvantage to the framework's own proponent. It is a warrant to prefer. A link supporting a framework without connecting to an advocacy renders as an advantage to reading that framework.

**RS13.** *Retired in v1.4.* There is no assert. Every component that derives no register renders under RS13c.

*Correction note.* RS13 was specified three times and fired on legal output every time. v1.1 keyed on same-side membership and fired on legal cross-side-rooted disads (`r36`, `shape2_link_turn`). v1.2 fired on 55 legal unfinished arguments. v1.3 reserved the assert for rounds containing no Advocacy anywhere and fired on `B4post_negwin_seed1_u25_ep000`, a legal post-mask round the policy actually produced.

The root error was constant across all three: the assert was trying to encode *"this input is stale."* Staleness is provenance, not structure, and the renderer sees only structure. `E.json` and `B4post_negwin_seed1_u25_ep000` are structurally identical; only their origin differs. No structural predicate can separate them — which is the same reason O5 could not be closed by survey.

Retiring the assert is therefore not a relaxation. It is the recognition that the renderer was never in a position to make the judgment the assert claimed to make.

**RS13b.** *Retired in v1.4.* Survey-before-enforcement existed to protect a wired assert. With no assert, there is nothing to enforce and nothing to survey against.

**RS13c.** Every component with no derivable register renders with an incomplete marker — advocacy reach empty and framework reach empty, regardless of what the rest of the round contains.

Such a component is a legal, unfinished argument. The mask governs legality at the moment a move is made; it does not guarantee that every terminal-impact component is anchored at round end. A debater may read an impact and never link it. Under RS21c, chains built across speeches are already legal; a component unanchored at round end is simply one the debater started and never finished.

A round containing no Advocacy at all renders as a round in which the AFF never advocated. That is readable and diagnostic. It is not a crash.

The renderer emits the component's claim and marks it as never connected. It does not invent a register, and it does not halt on any input.

Rationale: asserting would halt on legal rounds; skipping silently would hide them. Marking makes "the policy built 23 impacts it never linked" visible in prose — the same diagnostic value as RS27. Expect roughly a fifth of rendered components to carry this marker on the pre-W1a corpus (58 of 262). That is ugly, and it is an accurate picture of those rounds.

*Training-workstream finding, out of scope for rendering.* `B4post_negwin_seed1_u25_ep000` contains no Advocacy node. If the first move is always an advocacy, then either that invariant is not enforced in the action space or the AFF no-op'd every speech. Both are findings for the training workstream. The second would be a stronger statement of AFF collapse than anything currently in the diagnostics.

---

## 4. Flow order

**RS14.** Speech organization is derived from topology, never from within-speech construction order. Identical graphs constructed in different orders render identically. This is both cache-stable and faithful: real speeches are organized by argument, not by order of invention.

**RS15.** Components are ordered by **first appearance in the round**. Once established, a component's position in the flow is fixed for the remainder of the round.

**RS16.** Per-speech rendering procedure:

1. Obtain components per RS16b.
2. Walk components in established flow order.
3. Within each component, render all new material touching it: new nodes, new edges, extensions.
4. Within a component, traverse anchor outward to terminal per RS16c.
5. Tie-break on node id.

**RS16b.** *Component* (spec) is not *chain* (code). Since the v11 divergence, `resolve_chains` emits one chain per terminal impact, and several chains may share one `members` component (`passes.py:1027-1044`). The renderer's component is the grouping of chains by shared union-find root.

A divergent multi-terminal component renders as **one flow section with multiple terminals**, not one section per terminal. One advantage with two impact scenarios is read under one heading.

**RS16c.** Multi-terminal traversal: DFS from the anchor, node-id ordered, each node rendered exactly once. The shared spine renders once and then branches. This is what a debater does and it prevents duplicate rendering of shared links.

**RS16d.** Impactless material renders flat. Nodes carrying no terminal impact form no chain — `resolve_chains` skips them — so they are not components and derive no register. They render in a single unattached bucket, ungrouped.

Grouping them by argument would require re-running the same-side union-find in the renderer, which RS10 forbids. More importantly, material with no impact is not an argument, and rendering it as though it were would misrepresent the graph.

Validated at M1: a round with zero components (12 frameworks, 4 weighing, 10 uniqueness, 2 orphan impacts, 1 BD) rendered as fluent but hollow framework prose — an abstract debate about how to weigh, with nothing to weigh. Mechanical in places, notably eight weighing nodes rendered as an enumerated list, but legible throughout. A round with no substantive spine reading as one is the faithful outcome.

**RS17.** Cross-side attacks render inside the target's component, not in a section of their own. A NEG defensive attack on an AFF link appears within the discussion of that AFF chain. Line-by-line structure falls out of the partition; no separate rule is required.

**RS18.** Extensions attach to the component they extend.

---

## 5. Content structure

**RS19.** Every node carries two separately generated fields:

- **claim** — the tag. Short, frozen at creation, restated on every extension.
- **warrant** — the explanation of the claim. Generated once, never restated.

*Correction note (pending fix → resolved at m1-v3; resolution at foot of note).* "The tag ... restated on every extension" carries an unstated premise the M0/M1 implementation breaks: a node's claim is **its own** and a restatement happens **only on an explicit extension**. Surfaced on `W2_B4_nearmiss_seed3_u240_ep002` (three Advocacy nodes `n1`, `n8`, `n35`; `n1` and `n8` chained through shared link `n2`), where the M1 renderer collapsed two distinct 1AC advocacies into one. This is a topology misrepresentation — the one failure the render layer exists to prevent. Two causes, one per layer.

**Cause A — template layer (`templates.py:_NODE_STEM`): one mechanism, two faces.** The stem maps a node *kind* to a single canned string, and that string is wrong in two ways at once, both repaired by the same edit:

- *It does not vary across nodes.* Every Advocacy node emits the identical claim "we advocate the plan," so two structurally distinct advocacies — plan and second plank, plan and counterplan, plan and alternative — are textually indistinguishable in the prompt. Distinct node ids are distinct commitments and must render as distinct commitments; the claim slot must be **per-node**. The fix realizes this as a per-node discriminator scoped **per side per speech**, phrased as parallel rather than ranked — advocacy nodes in a component are unordered, so "one advocacy" / "a separate advocacy" / "a third, distinct advocacy", never a "first"/"second" that invites the model to read the later node as derivative — and internal node ids never surface in prose.
- *The string it produces is wrong on its face.* "we advocate the plan" names a policy plan, so a NEG advocacy — a counterplan or alternative (RS5b) — or any advocacy under a value resolution is misrendered even when the round has exactly one advocacy node. The stem must be **type-neutral with polarity from side**: AFF advocacy affirms, NEG advocacy is a competing advocacy — RS5b restated at the stem, no new channel. Conditioning further on *register* to name a framework-register advocacy an "interpretation" is **rejected**. Register is a judge-side gating property; reading a claim's content-type off it would place semantic inference in the render layer. RS5b's side-based polarity is defensible precisely because side is a structural fact about who submitted the node — register is not that kind of fact. ("Interpretation" is additionally an emergent argument-type in the class RS5/RS6 forbid.) The framework case stays neutral; if a future round needs interpretations to read as interpretations, that is a content-authoring problem, not a template one. The framework-advocacy case is near-empty besides: across the 2327-round corpus, of 332 framework-register components exactly **1** (in `W2_B4_nearmiss_seed2_u236_ep001`, 3 Advocacy members) contains any Advocacy member at all, and even there advocacy reach is empty — consistent with RS11 (framework register ⟺ empty advocacy reach). Verified against the corpus, not inferred from the spec.

**Cause B — prompt layer (system-prompt rule 6): independent of the stem.** Rule 6 lets the model infer "restatement" when two claims read alike; that inference is available for every same-kind node pair, not only advocacies, and it would survive a correct stem. The skeleton already carries extensions as their own event type. The prompt must state that distinct node ids are distinct commitments and that **only an explicit extension event** licenses treating a claim as a restatement of a prior one. This is the general fix; Cause A's per-node stem is the local one.

*Scope (unverified surface).* Blast radius (2327-round corpus): **55.8%** of rounds carry >1 Advocacy node; **82.0%** carry some component with >1 node of a single kind. The 82.0% is an **unverified surface** — the set of sites where the restatement inference is *available* — not a measured collapse rate. Collapse is confirmed only for advocacy, observed directly on the round above. For other kinds (two impacts, two frameworks in one component) the firing rate is **unmeasured**: their edge wiring may carry enough distinguishing context that the model keeps them apart. Measuring the non-advocacy firing rate is logged as a follow-up; no claim is made here that the collapse fires at that rate.

*Measured (m1-v4), provisional — counts over-report, see below.* A first pass sampled 20 sites per kind (two same-kind nodes co-introduced in one component in the 1AC; empty prior, faithful since the 1AC has none), rendered under R2 and LLM-judged, scoring **impacts 3/20 and frameworks 6/20** as collapsed. The scorer used too broad a criterion. The correct one: a collapse is *N distinct nodes rendered as fewer than N distinct commitments*; it is **not** a collapse when the two same-kind nodes are joined by a support edge, because one impact reached through another (or two frameworks where one grounds the other) renders as a connected progression — that is fidelity. Convergent structure is legitimate topology, not a defect. Checking the wiring of the flagged sites: 2 of 3 impact sites carry an inter-node support edge (`n14`→`n12`) and 2 of 6 framework sites do (`n6`→`n4`, `n7`→`n6`), so those are not collapses; that leaves 1 impact and 4 framework genuine among the flagged. The denominators also still include inter-wired pairs that should not be sites, so the rate is not recomputable by hand — a corrected measurement (sites = two same-kind nodes with no support path between them) is pending. Treat 3/20 and 6/20 as loose upper bounds, not the firing rate. What holds regardless of the recount: same-kind collapse does occur for non-advocacy kinds, and it is measured *with* the Cause-B rule-6 prompt fix already active but non-advocacy stems still per-kind (only advocacy got the per-node stem, Cause A) — so it is the residual the prompt fix leaves; driving it lower would need per-node stems for those kinds too (not attempted). Caveats on the sample: n=20 (wide CIs), one judge pass per site, and kind is confounded with workspace (impact sites all W2_B4, framework sites all W3_B4), so no cross-kind comparison is claimed.

*Re-measured (corrected criterion, same 20 sites/kind, no re-sampling).* A collapse is now (1) two same-kind nodes not both articulated as distinct claims **or** (2) a causal step between them asserted/skipped rather than explained — one impact reached through another must render the step, not just a trajectory; a chain must explain every link; parallel (sibling) pairs must render both pathways as two. Result: **impact 3/20** (by relation: direct 3/13, parallel 0/7) and **framework 8/20** (chain 4/4, direct 1/6, parallel 3/10). The impact count coincided at 3/20 old and new **with zero overlap in which sites** — a complete turnover, not stability (do not read "3 → 3" as an unchanged result): all three sites the earlier pass flagged — the two direct-edge pairs and the one parallel pair — **pass** under audit (the render did articulate both impacts and did explain the fossil-dispatch / demand-to-blackout step), and three *different* sites failed, each where a step is asserted or skipped. So the old node-only criterion both over-counted (faithful trajectories flagged) and under-counted (unexplained steps missed); the real failure mode is edges, not node-merger, and it concentrates where a step exists to skip — every chained framework pair failed (4/4), while parallel pairs rarely fail (impact 0/7). Frameworks rose 6→8 under the stricter edge check. Within-workspace figures only (kind confounded with workspace, as above); n=20 each, one judge pass per site, measured with rule 6 active and non-advocacy stems still per-kind. The failures are **edge-side** — a causal step between two named nodes stated but not explained — so, unlike the advocacy bug, a per-node stem would not touch them; that idea is shelved for these kinds (impact parallel sites were 0/7, no node-merger problem worth fixing). Any lever is prompt-level and is deferred pending the chained-relation diagnosis below.

*Chained relations tested at scale (m1-v4).* The 4/4 chained-framework rate above was small-sample. Pulling every chained same-kind 1AC site in the already-selected workspaces (no new workspaces; two same-kind nodes with a multi-hop directed support path between them): **framework 58/106 (55%)** and **impact 6/21 (29%)** collapse. So 4/4 was noise — but chained relations still fail far above direct-edge (framework 1/6 earlier) and parallel (impact 0/7), and the rate rises with chain length (framework k=3: 8/10; impact k≥3: 3/4). This is a structural finding about the render layer, and it is an **edge/mechanism** failure, not node-merger: the dominant mode is *asserted-no-mechanism* — the connection is stated and the mechanism omitted (framework 42 of 58 failures; e.g. "the link between the moratorium and improved resource efficiency is structural, not incidental", or two impacts "mutually reinforcing rather than independent") — with *skipped-middle* second (framework 9, impact 3) and node-merger rare (7). Caveat on interpretation: many same-kind "chains" route through other kinds (framework→impact→framework, impact→framework→…→impact); for frameworks that is a weighing-lens relation, not a harm escalation, so a share of the 55% is the causal-step criterion applied to non-causal topology — flagged, not resolved. Fix deferred; diagnosis under review; not a node problem.

*Restricted to same-kind causal chains, floating nodes excluded — the 55% mostly dissolves.* **Limit on the criterion (not just a filter):** the edge-explanation requirement applies to *causal* relations only. A framework relation is causal only when a framework connects **directly** to another framework; `framework→impact→framework` is a weighing-lens relation and is **not a site**. A later reader who applies edge-explanation to weighing relations will reproduce the inflated number below. Re-running with every interior node the same kind as the endpoints: **framework 9 sites** (of 106 unrestricted — 97 routed through impacts), **impact 17** (of 21). Same-kind collapse: **impact 4/17, framework 3/9**. Floating nodes (a node with **no incoming edges** — unfilled, not implicated in the advocacy structure; scoring the render for a relation that isn't there) inflate further: 2 of the 9 framework sites contain one and **both** collapsed, so excluding them gives **framework 1/7**; impacts had 3 floating sites, **none** collapsed, so impact holds at **4/14** excluding floating. So the unrestricted **55%** was almost entirely (a) the criterion applied to non-causal weighing-lens pairs and (b) floating unfilled nodes; genuine same-kind non-floating framework chains collapse **1/7**, near the direct-edge rate (1/6) — **no special chain effect survives** for frameworks. Both denominators are tiny and the impact failures overlap (3 of the 4 are sub-chains of one round, `u25_ep001`), so distinct failing rounds are **impact 2, framework 1** — provisional, not a structural rate. Report both: 55% is the unrestricted figure; 4/14 (impact) and 1/7 (framework) are the same-kind non-floating figures.

**Second limit — the criterion has two forms.** Edge-explanation is not one obligation. For **impact→impact** the render owes a *causal mechanism* (how the first harm produces the next). For **framework→framework** the relation is *entailment*, not causation: the render owes *why the first principle commits you to the second*, and demanding a causal mechanism there is the criterion misapplied — the same error as applying it across weighing pairs. A reader applying the causal form to framework edges will over-count. Re-scored under the entailment form, the one surviving framework case (`u489`, `n14→n12→n10`) **still collapses**: all three principles are articulated, but the irreversibility→proportionality step is asserted via "connects directly to" with no reason given — a genuine entailment failure, not a misapplication, so **1/7 stands** under the correct criterion.

What the surviving failures show (diagnosis, no fix): the render articulates every node but marks the edge with a relational connective — "compounds", "mutually reinforcing", "further specified by", "a third principle follows", "connects directly" — and substitutes that connective for the reason (mechanism for impacts, entailment for frameworks). Assert-via-connective, not skip-the-node.

**Provisional, and closed.** These figures are not a structural rate and are not a basis for designing a fix. The impact 4/14 rests on 4 failures across just 2 rounds — 3 of them overlapping sub-chains of a single round (`u25_ep001`) — and the framework 1/7 on a single round (`u489`). The sample is not widened and measurement stops here; if assert-via-connective matters it will resurface in content-authored rounds where there is something real to compare against.

*Resolved (m1-v3).* Cause A fixed in `templates.py` (`advocacy_stem`, per-node `stem_override`) and `linearize.py` (ordinal scoped per side per speech); Cause B in `llm.py` (rule 6 — distinct ids are distinct commitments, restatement only on an explicit extension event). `PROMPT_TEMPLATE_VERSION` → `m1-v3` ships with the code because it is what makes the fix take effect: under the RS30 cache key an unbumped version serves the stale collapsed render. Re-render of `W2_B4_nearmiss_seed3_u240_ep002` under R2 shows `n1` and `n8` as two distinct 1AC advocacies (energy/climate and resource-equity planks sharing the moratorium mechanism). Two follow-ups remain open: (a) the non-advocacy same-kind firing rate is still unmeasured (above); (b) the per-side-per-speech ordinal scope does not disambiguate a distinct advocacy introduced in a *later* speech from a same-side advocacy in an earlier one — `n35`, a distinct 2AC advocacy in this round (own node id, liveness `{2AC}`, not an extension of `n1`), rendered thin as "We affirm the moratorium", a restatement of the plan text in different clothes. Cross-speech advocacy distinctness is unhandled; not fixed here.

*Correction (ordinal scope, m1-v4).* The scope recorded above — "per side per speech" — was wrong, and it broke exactly where predicted, on `n35`. The per-side constraint was right (it stops a NEG advocacy inheriting AFF's count); the per-speech reset was not thought through. An advocacy is a commitment in the *round*, not the speech: `n1` and `n35` are as distinct as `n1` and `n8`, and the boundary between them is temporal, not structural, so the discriminator should not care. Ordinal scope is therefore **per side, per round** (`linearize.py`; `PROMPT_TEMPLATE_VERSION` → `m1-v4`), ordered by round appearance — `n35` is discriminated against `n1` and `n8` (→ "a third, distinct advocacy") the same way `n8` is against `n1`. Re-render under R2 confirms `n35` renders as a distinct third advocacy (federal siting-coordination policy), not a restatement; the cross-speech phrasing "a third, distinct advocacy" did not reintroduce the collapse by a cross-reference route. This closes follow-up (b) from the Resolved note above. Follow-up (a) — the non-advocacy firing rate — is now measured; see the Scope paragraph.

**RS20.** The warrant explains the claim. It does not explain the edge. Inter-node inference is carried by chain sequence: link → link → impact reads as an argument without connective prose.

**RS21.** Generation order is forward: anchor → link → impact. Generation order and presentation order are identical; no reversal step.

Rationale: the intuition that debaters reason backward describes *content → strategy* — how arguments are built before a round. CDAF inverts this. The topology is fixed first and content fills it, so the pipeline is *strategy → content*. Backward traversal within a chain does not follow from the debater intuition and is not adopted.

Forward generation is safe because of RS9. Rendering is speech-atomic, so the renderer sees the terminal impact and every intervening link slot before generating any content. It knows it has *n* links to travel from anchor to a known terminal and paces the inferential burden accordingly. The burden-distribution problem belongs to incremental node-at-a-time generation (rejected, §9.3), not to forward traversal.

Forward generation additionally keeps impacts canonical. Impacts are the stock element and links are the contingent one; generating the impact first would permit a bespoke terminal that then reverse-constrains the links. Forward traversal against a visible terminal puts the contingent work in the links, where it belongs.

**RS21b.** Node identity determines content identity. Distinct nodes receive distinct content; shared nodes are genuinely shared.

- Two link nodes supporting one impact node = one impact with two links. This is ordinary and correct.
- Two distinct impact nodes = two materially different impacts. The renderer must not render the same impact twice under different node ids.

The renderer sees the full graph and can enforce this directly. Where converging paths are rendered in the same speech, the shared terminal is generated once, on the first path traversed by RS16 order; subsequent paths generate against it as already-frozen content. This constrains later links to actually reach the established terminal.

**RS21c.** Chains spanning speech boundaries are generated under partial information, and this is accepted rather than corrected.

RS9 guarantees visibility of the current speech's subgraph only; RS7 forbids visibility of future speeches. A chain whose links appear in one speech and whose terminal impact appears in a later one therefore has its links generated against an unknown terminal, and RS8 freezes those links before the impact arrives. The impact must fit what was already said.

This is faithful. A chain built across speeches genuinely was underdetermined when its links were written, and a debater adding an impact in the 2AC is constrained by the 1AC's link language. Where the fit strains, the strain is diagnostic in the RS28 sense.

Rejected alternative: extending renderer visibility from the current speech's subgraph to the full graph. This would let the renderer see a terminal not yet introduced, violating RS7 and leaking future speeches into the transcript. Judge blindness is the more valuable property.

**RS22.** The claim/warrant split is the attachment seam for prep pool grounding (§8, M3). At M3, claims become retrieval keys and warrants become retrieved text. No other field changes.

---

## 6. Budgets

**RS23.** Per-event word budgets:

| Event | Budget | Shape |
|---|---|---|
| New node | 50 | claim 15 / warrant 35 |
| Edge introduced in same speech as both endpoints | 0 | syntax only ("…which means…", "…that turns…") |
| Support edge, one or both endpoints pre-existing | 50 | cross-application |
| Defensive attack edge | 50 | refutation |
| Extension | ~15 | claim restatement only, no warrant |

**RS24.** Speech length is the sum of its event budgets. There is no independent speech-level cap.

**RS25.** A bare edge — an edge targeting two existing nodes, introducing no new node — is by definition later than both endpoints and receives the full 50-word budget. Register keys off edge type:

- **Support** → cross-application. "Extend our link from the 1AC — it applies here too, because…" Endpoint node content is untouched; the connective work is new and belongs to the speech in which the edge was made.
- **Defensive attack** → refutation. Claim is the takeout; warrant is why the target fails.

**RS26.** Extensions restate the claim in full every time, including repeated extensions of the same node across multiple speeches. No decay to bare reference. Rebuttals are compressed restatement; this is faithful, and it makes extension a visible textual event in the transcript.

Consequence: a rebuttal introducing no new nodes still renders. A 2AR extending eight nodes produces roughly 120 words plus new material. This directly serves the `extension_fail` diagnostic — the transcript shows, in prose, whether a policy is carrying offense through the round or rebuilding it each speech.

---

## 7. Diagnostics

**RS27.** Judge-invisible edges render. A debater made the move; it merely did not land. The edge is rendered and marked with its judge-emitted reason.

**RS27b.** The renderer never computes visibility. It consumes only invisibility records the judge already emits, and marks exactly those. A renderer-side visibility analysis would be a second implementation drifting from the judge — the same failure that RS10 guards against, one layer up.

Current vocabulary is therefore exactly what the trace names:

- `InertAttack(edge_id, reason)` — `judge/trace.py:95`
- `WindowClosed(edge_id, …)` — `judge/trace.py:104`

Both are attack-edge reasons. Visibility is computed, not stored on edges, and there is no canonical judge record enumerating every invisible edge with a name.

*Correction note.* Earlier drafts implied RS27 would surface the full ~33% of edge-creating moves producing edges the judge never reads. It cannot. That set includes Support edges — same-speech zero-budget edges, cross-side fusion never traversed, edges into dead components — for which no named judge record exists. Producing that readout requires a **judge-side invisibility enumerator**, which is work on judge code and a separate workstream. It is not started while pod experiments are running against the tree.

**RS28.** Cross-application prose that visibly strains is signal, not failure. A renderer struggling to justify a late edge indicates a move that scores structurally while meaning nothing.

**RS29.** Semantic coherence scoring, where run, is **offline, on completed graphs, as a reported metric only**. It never enters the environment, the reward, or move selection. Divergence between coherence and structural score indicates a spec gap, resolved by judge revision at Yaz's ruling — never by reward shaping.

---

## 8. Milestones

**M0 — template renderer, no LLM.**
Node type × edge type → canned sentence with placeholder claim slot. Deterministic, instant, free. Purpose is to isolate linearization — flow order, component partition, extension lines, invisible-edge markers — so speech structure is debugged independently of content quality. Deliverable: typed placeholder skeleton per speech.

**M1 — one LLM call per speech.**
The M0 skeleton is the prompt. Eight calls per round. Content thin, prose fluent, output readable.

**M2 — two-stage per speech.**
Call 1 emits structured content as JSON (claim, warrant) per new node and edge. Call 2 writes prose from it. Sixteen calls per round. The structured intermediate permits regenerating a single node without redoing the speech. This is where readability is expected to land.

**M3 — prep pool grounding.**
Each side receives a fixed prep pool before the round begins. Rendering becomes assignment rather than generation: nodes map to pre-written cards, and only connective glue is generated. Keyword or tag matching over a structured card file is sufficient at single-resolution scale; embeddings only if retrieval quality proves to be the bottleneck.

---

## 9. Rejected designs

**9.1 Content-aware action selection.** Rejected on three independent grounds:

- **Throughput.** PPO self-play requires millions of environment steps. An LLM call per candidate action on 16 CPU vCPUs is several orders of magnitude off feasible.
- **Determinism.** A coherence model inside the environment makes it non-stationary and non-reproducible. Every oracle fixture ceases to be a fixed point.
- **Tabula rasa.** A coherence checker shaping move selection is a hardcoded opinion about argument quality wearing a neural network as a disguise.

The legitimate concern behind the proposal is served by RS29.

**9.2 Retrieval influencing move selection.** Same rejection as 9.1. Retrieval at render time (M3) is unaffected.

**9.3 Node-at-a-time rendering in construction order.** Superseded by RS9 (speech-atomic) and RS14 (topology-derived order). This design is the source of three problems, all resolved by its rejection rather than by any compensating mechanism:

- *Retroactive mutation* — resolved by RS8 plus RS25. Late edges generate new prose in their own speech rather than modifying frozen endpoints.
- *Uneven inferential burden* (one link silently carrying five links' worth of inference, or a chain failing to reach its impact) — resolved by RS9. A renderer that sees the whole speech subgraph knows how many slots it has and what terminal it must reach.
- *Duplicate or incoherent convergence* — resolved by RS21b.

**9.4 Backward (impact-first) generation within a chain.** Rejected by ruling. Justified originally by the debater intuition of reasoning backward from impact to link; that intuition describes content → strategy and does not transfer to CDAF's strategy → content pipeline. See RS21.

**9.5 Coherence-filtered BC corpus.** Parked. Not a time-priority item, and it invites the training-data contamination the project has otherwise avoided.

---

## 10. Reproducibility

**RS30.** Cache key: `hash(canonical_graph, model_id, prompt_template_version, render_mode)`.

Graph hash alone silently serves stale renders on model switch or prompt revision.

**RS31.** Canonicalization excludes within-speech construction order (per RS14). Identical graphs built in different orders share a cache entry.

**RS32.** Frozen model version, temperature 0, recorded in every render artifact.

**RS33.** Multi-model comparison is *same graph, different renderers* — rendering quality isolated with strategy held fixed. Models generating strategy is a different architecture (LLM-as-agent rather than LLM-as-renderer) and is not reachable from this design.

---

## 11. Citations

**RS34.** Current mode is pure analytics. No citations of any kind until M3.

**RS35.** Two guards, both required from M1:

- Explicit prohibition in the prompt.
- Regex validator on output for author-year patterns.

Rationale: debate transcripts in training data are saturated with `Smith '19`. A renderer told nothing about citations will emit fabricated ones from stylistic habit alone. Fabricated cites in a published artifact are a publication hazard.

**RS36.** The prompt states that warrants are reasoning-only. Analytic warrants and carded warrants have materially different shapes, and the renderer defaults to the latter absent instruction.

**RS37.** At M3, citations are real and come from the real prep corpus. There is no intermediate mode with plausible-looking synthetic cites.

---

## 12. Open — empirical, resolved by reading M1 output

**O1.** ~~Whether 50 words suffices for a warrant carrying a long chain's inferential load.~~ **Closed at M1.** The failure mode cannot arise as specified: every node carries its own 35-word warrant, so inferential load distributes across the topology rather than concentrating in one warrant. This confirms the RS21 forward-generation rationale empirically. Budget-as-target also holds — measured speech lengths landed at 415/400, 653/615, 911/880 against their targets, with neither padding nor truncation.

*Caveat recorded.* W1a produces no chains of 5+ serial links, so the extreme case is untested. The argument that it cannot bite is structural, not measured.

**O2.** ~~Whether a digest mode (~200–300 words/speech) survives contact with the node-count budget.~~ **Closed by ruling.** It does not, and this is accepted. A 7-node 1NC at ~350 words is fine. Variable speech length is fine and expected. There is no digest-mode cap and no length trimming. Corpus node counts per speech: 1AC=237, 1NC=83, 2AC=19, 2NC/1NR=11, 1AR=2, 2NR=16, 2AR=34 (oracle corpus totals) — rebuttals are thin, so rebuttal length is almost entirely RS26 extension lines.

**O3.** ~~Whether rendering surfaces edge-legality gaps the current mask does not cover.~~ **Rescoped by RS27b.** Rendering can only surface gaps the judge already names, so RS27 is an instrument for `InertAttack` and `WindowClosed` only. The broader edge-waste readout is blocked on a judge-side invisibility enumerator and is not an M0 or M1 question.

**O4.** ~~Whether any component acquires the wrong register via cross-side fusion through the shared Advocacy (RS11b).~~ **Closed.** Surveyed at zero across 47 fixtures, but the question was malformed — see retired RS11b. The real defect ran the other way and is fixed in RS10b/RS11.

**O5.** ~~Whether the post-mask export set is genuinely post-mask, resolved by surveying asserts across the exports.~~ **Closed as unresolvable by this instrument.** The survey returned 58 no-register components across 262. Under RS13c that count is consistent with the mask being live — unfinished arguments are legal — but consistency is not demonstration. A legal unfinished argument and a pre-mask floating root are structurally indistinguishable at round end. Mask verification requires a run-config record, not a corpus survey. Do not re-attempt via survey.

**O6.** Whether M1 readability tuning is contaminated by edge soup. Every available export predates the edge-soup fix (`5c87129`), with up to 2.4× redundant cross-side edges per pair. Under RS25 each redundant edge draws a full 50-word cross-application, so a soup-heavy round renders as the same cross-application restated repeatedly. Accepted for M0, where repetition is visible at zero cost and arguably demonstrates RS28. Revisited before M1 **if** exports can be regenerated from saved checkpoints without new pod time; if regeneration requires new runs, the diagnostic reading stands and M1 proceeds on the existing corpus.

---

## 13. Standing foreclosures touched by this spec

None proposed. RS13 records that unanchored chains are already foreclosed in the action space; this spec relies on that foreclosure but does not extend it.

Any legality mask suggested by render diagnostics is surfaced as a decision, not implemented — foreclosing changes the action space mid-experiment.
