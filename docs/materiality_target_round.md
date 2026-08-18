# Materiality — mentor-meeting flagship round (Step 1 finding)

**Status:** identified, pending Yaz's confirmation before UI is built around it.

## The round

**`runs_verify/B4post/exports/B4post_negwin_seed1_u25_ep000.json`**

This is the zero-component round described in `docs/render_spec.md:177` — the one
"validated at M1 [that] rendered as fluent but hollow framework prose — an abstract
debate about how to weigh, with nothing to weigh." Its node composition matches that
note exactly:

| kind | count |
|------|-------|
| framework | 12 |
| uniqueness | 10 |
| weighing | 4 |
| impact | 2 (orphan) |
| ballot_directive | 1 |
| link | 0 |
| advocacy | 0 |

29 nodes, 31 edges (support 11, defensive_attack 12, comparison 8). No links, no
advocacy → no chains → zero components → the judge returns a presumption loss for AFF
while the prose reads as confident moral philosophy. This is the "confident moral
philosophy vs. presumption loss" artifact.

## It is NOT the perf-check round

The materiality perf check ran on `W3_B4_negwin_seed2_u400_ep002.json` (59 nodes /
116 edges, also N=0). That is a **different** round — coincidentally also a no-offense
presumption round, but larger and from a different bucket (W3_B4 vs B4post). The
flagship artifact is the 29-node B4post round above.

## compute_materiality output (epsilon = 0.0, zero-tolerance)

- **baseline:** winner = **NEG**, margin N = **0.000000** (presumption loss for AFF)
- **material: 0 / 60** elements. non-material: 60 / 60.
- **best path per side:** AFF = none, NEG = none (no offense chains either side).

### Divergence from expectation

The Step-1 prompt expected "only the presumption gate itself" to stay material. Actual
output: **nothing is material — not even the BallotDirective.** Deleting the BD (`n33`)
leaves winner = NEG, N = 0.0, unchanged. That is correct, not a bug: NEG wins by
presumption *regardless* of any single element, because there is no AFF offense to
disturb. With N pinned at 0 and no chains, no single deletion can flip the winner or
move the margin, so every element is immaterial under single-element deletion.

The visualization consequence is the strong version of the intended story: under the
translucency toggle this round goes **entirely translucent** — the whole "confident
moral philosophy" graph is materially inert scaffolding. That is arguably a *better*
flagship demonstration than "only the presumption gate lights up," but it is a
different claim, so it is flagged here for Yaz to confirm before UI is built on it.

_Method: `analysis.materiality.compute_materiality` — reruns the unmodified six-pass
judge on deep-copied single-element deletions. Read-only; nothing upstream touched._
