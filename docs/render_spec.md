# CDAF Render Spec v1 (draft)

Status: **accepted**. All semantic rulings closed. No code written against this document yet.

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

**RS10.** A chain's register is derived from its anchor component. Anchor membership is the judge's existing union-find from the impact terminal over Support paths whose interior nodes are never Advocacy or BallotDirective (both absorbing, not traversable). The renderer performs no independent graph analysis.

**RS11.** Two registers:

| Anchor | Register |
|---|---|
| Advocacy | Substantive offense about the advocacy's consequences |
| Framework | Reason to prefer that framework |

**RS12.** Framework-anchored offense is available to both sides and is not a disadvantage to the framework's own proponent. It is a warrant to prefer. A link supporting a framework without connecting to an advocacy renders as an advantage to reading that framework.

**RS13.** Unanchored chains are illegal in the action space as of current runs. The renderer therefore has no unanchored register and no fallback branch. Encountering an unanchored chain is an **assertion failure**, not a render path.

Rationale for RS13 as an assert rather than a render: older fixtures predate the legality mask, and hand-authored graphs enter through the builder. A loud failure identifies stale input. A silent one fabricates content the semantics say cannot exist.

---

## 4. Flow order

**RS14.** Speech organization is derived from topology, never from within-speech construction order. Identical graphs constructed in different orders render identically. This is both cache-stable and faithful: real speeches are organized by argument, not by order of invention.

**RS15.** Components are ordered by **first appearance in the round**. Once established, a component's position in the flow is fixed for the remainder of the round.

**RS16.** Per-speech rendering procedure:

1. Compute anchor components over the full graph.
2. Walk components in established flow order.
3. Within each component, render all new material touching it: new nodes, new edges, extensions.
4. Within a component, traverse anchor outward to terminal.
5. Tie-break on node id.

**RS17.** Cross-side attacks render inside the target's component, not in a section of their own. A NEG defensive attack on an AFF link appears within the discussion of that AFF chain. Line-by-line structure falls out of the partition; no separate rule is required.

**RS18.** Extensions attach to the component they extend.

---

## 5. Content structure

**RS19.** Every node carries two separately generated fields:

- **claim** — the tag. Short, frozen at creation, restated on every extension.
- **warrant** — the explanation of the claim. Generated once, never restated.

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

**RS27.** Judge-invisible edges render. A debater made the move; it merely did not land. The edge is rendered and marked invisible with its named reason.

Rationale: this makes the transcript a direct readout of the edge-waste workstream (~33% of edge-creating moves produce edges the judge never reads). The prose shows what the policy believes it is doing; the marker shows what the judge reads.

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

**O1.** Whether 50 words suffices for a warrant carrying a long chain's inferential load. Unknowable before M1 output exists.

**O2.** Whether a digest mode (~200–300 words/speech) survives contact with the node-count budget. A dense 1NC may exceed it substantially. M0 yields node counts per speech and answers this cheaply.

**O3.** Whether rendering makes any additional edge-legality gap visible that the current mask does not cover. RS27 is the instrument.

---

## 13. Standing foreclosures touched by this spec

None proposed. RS13 records that unanchored chains are already foreclosed in the action space; this spec relies on that foreclosure but does not extend it.

Any legality mask suggested by render diagnostics is surfaced as a decision, not implemented — foreclosing changes the action space mid-experiment.
