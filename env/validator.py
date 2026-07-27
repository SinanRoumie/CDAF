"""Episode-validity fences for the CDAF RL environment.

A self-play agent explores the whole space of well-typed argument graphs,
including corners no hand-authored oracle round covers. Where the judge has a
RULED semantics for a corner we CLOSE it with an oracle round; where it does NOT
(the branch is provisional / out of scope / malformed), the ENVIRONMENT must
refuse the graph so no training episode is ever scored against an unaudited
verdict. Those refusals live here, NOT in the judge -- the judge stays a pure
scoring function; the env owns episode validity.

Three fences (gap-audit STEP 3/4), each verdict-pure and detected where the fact
is knowable:

  A. MULTI-TERMINAL / MALFORMED COMPONENT (structural, pure ingest). A same-side
     Support component that contains an Impact but whose terminal-impact count is
     not exactly 1 hits the flat pre-per-path fallback (passes._build_chains else
     branch), which has none of the §3.3.1 per-path protections. Knowable from
     Pass-1 structure alone (adjacency + speech order), so it is checked at pure
     ingest. The component/terminal computation REUSES the judge's own primitives
     (`passes.build_context`, `passes._union_find`, `passes._terminals`) so it
     cannot drift from what the judge actually does.

  G. OFF-VOCAB SPEECH (structural, pure ingest). A node whose `speech` is not in
     SPEECH_ORDER is not inert in the judge: the judge's `_sidx` maps it to None,
     which silently short-circuits `node_extension_ok` to "extended" (privileging
     the node), dodges drop detection, and fakes a 1AC intro -- while the model's
     own `speech_index` maps the same string to len(SPEECH_ORDER) (treats it as
     last). There is no coherent inert semantics to fall back on, so an off-vocab
     speech is rejected as malformed at ingest.

  B. UNEQUAL-MAGNITUDE CONVERGENCE (judge trace marker). Two sign-conflicting
     paths converging on one shared impact with UNEQUAL magnitudes is declared out
     of scope for V1 (§3.3.1c); the judge's handling there is provisional. Whether
     a graph reaches it is only knowable AFTER passes 1-5 (it depends on resolved
     sign and magnitude), so it cannot be a pure structural predicate. The judge
     emits a WRITE-ONLY `CONVERGENCE_OUT_OF_SCOPE` marker when it reaches that
     branch (verdict unchanged -- see judge.trace.ConvergenceOutOfScope); this
     validator RUNS the pure judge and refuses any round whose trace carries the
     marker. We read the judge's own signal rather than re-deriving the condition
     in a second resolver, because a drifted second resolver is exactly the
     silent-divergence bug this pass exists to prevent.

`validate_round(rnd)` returns a `RoundValidity`; the RL env calls it at
episode-init and only admits graphs where `.ok` is True.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import List

from model import Impact, Support, SPEECH_ORDER
from judge import judge as run_judge
from judge import passes

CONVERGENCE_MARKER = "CONVERGENCE_OUT_OF_SCOPE"


@dataclass
class RoundValidity:
    """Result of validating one Round for episode admission. `ok` is True iff no
    fence tripped; `reasons` lists every rejection (keyed by fence tag) so a
    training harness can log/aggregate which corner an agent tried to reach."""
    ok: bool
    reasons: List[str] = field(default_factory=list)


def _offvocab_speech_nodes(rnd) -> List[str]:
    """FENCE G: node ids whose `speech` is not a canonical speech (∉ SPEECH_ORDER).
    Reads the raw round; no resolution needed."""
    vocab = set(SPEECH_ORDER)
    return [n.id for n in rnd.nodes if getattr(n, "speech", None) not in vocab]


def _multiterminal_components(rnd) -> List[List[str]]:
    """FENCE A: same-side Support components containing an Impact whose terminal-
    impact count (passes._terminals) is not exactly 1 -- the case that reaches the
    judge's flat multi-terminal fallback. Returns the offending components' member
    lists.

    REUSES the judge's own Pass-1 context and helpers so the enumeration is
    byte-identical to `passes._build_chains`; the only thing mirrored here is the
    same-side Support union loop (`_build_chains` lines ~624-637), kept in lockstep
    with that source of truth."""
    ctx = passes.build_context(rnd)                    # Pass 1 only: structure/adjacency
    ids = list(ctx.reachable)
    find, union, _parent = passes._union_find(ids)
    for e in ctx.edges:
        if isinstance(e, Support):
            a = ctx.nodes.get(e.source)
            b = ctx.nodes.get(e.target)
            if a and b and a.id in ctx.reachable and b.id in ctx.reachable and a.side == b.side:
                union(a.id, b.id)

    comps = defaultdict(list)
    for nid in ids:
        comps[find(nid)].append(nid)

    bad = []
    for members in comps.values():
        impacts = [m for m in members if isinstance(ctx.nodes[m], Impact)]
        if not impacts:
            continue                                   # no Impact -> not a scoring chain
        terminals = passes._terminals(ctx, members, impacts)   # judge's own terminal rule
        if len(terminals) != 1:
            bad.append(sorted(members))
    return bad


def trace_has_convergence_marker(trace) -> bool:
    """Pure predicate: does this trace carry the write-only CONVERGENCE_OUT_OF_SCOPE
    marker? Separated from the judge run so the fence mechanism is unit-testable
    independently of whether any V1 graph can reach the branch (in binary V1 accrual
    every live path magnitude is exactly 1.0, so the equal-magnitude WASH fires and
    the unequal branch is unreachable -- the fence is future-proofing for any regime
    that introduces fractional magnitudes)."""
    return any(getattr(r, "kind", None) == CONVERGENCE_MARKER for r in trace)


def _reaches_convergence_marker(rnd) -> bool:
    """FENCE B: run the pure judge and report whether it emitted the write-only
    CONVERGENCE_OUT_OF_SCOPE marker (the unequal-magnitude convergence branch,
    §3.3.1c). Reading the judge's own trace avoids a drifted second resolver."""
    _ballot, trace = run_judge(rnd)
    return trace_has_convergence_marker(trace)


def validate_round(rnd) -> RoundValidity:
    """Validate a Round for episode admission. Applies the three fences; returns a
    RoundValidity with every tripped fence recorded. Structural fences (A, G) run
    first (cheap, no resolution); the marker fence (B) runs the judge last."""
    reasons: List[str] = []

    offvocab = _offvocab_speech_nodes(rnd)
    if offvocab:
        reasons.append(
            f"G/off-vocab-speech: nodes {offvocab} carry a speech not in SPEECH_ORDER")

    multiterm = _multiterminal_components(rnd)
    if multiterm:
        reasons.append(
            f"A/multi-terminal: {len(multiterm)} same-side Support component(s) with "
            f"a terminal-impact count != 1: {multiterm}")

    if _reaches_convergence_marker(rnd):
        reasons.append(
            "B/unequal-magnitude-convergence: judge emitted CONVERGENCE_OUT_OF_SCOPE "
            "(§3.3.1c out of scope for V1)")

    return RoundValidity(ok=not reasons, reasons=reasons)


def is_valid(rnd) -> bool:
    """Convenience boolean for the RL loop's admission check."""
    return validate_round(rnd).ok
