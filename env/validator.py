"""Episode-validity fences for the CDAF RL environment.

A self-play agent explores the whole space of well-typed argument graphs,
including corners no hand-authored oracle round covers. Two DISTINCT concerns
live here, deliberately split (Phase-1 Ruling 1):

  1. STRUCTURAL ADMISSION (`validate_round`, Fences A + G). Well-formedness of a
     completed round: the graph the judge is handed must be one the judge has a
     ruled answer over. Under Phase 1 these corners are made UNREACHABLE by the
     legal-action generator (Fence A is a local, monotonic generator check; Fence
     G is satisfied by construction because the env stamps every node's speech
     from the current slot). So at the termination step `validate_round` is an
     ASSERTION, not a branch: if it ever fails, that is an environment BUG and the
     caller must raise loudly rather than return any reward value.

  2. SCOPE GUARD (`assert_scope_ruled`, Fence B). A SEPARATE concern with a
     separate call site: "does the judge have a RULED semantics for this round,"
     not "is this round well-formed." `CONVERGENCE_OUT_OF_SCOPE` fires on the
     unequal-magnitude convergence case (§3.3.1c), which is an UNRULED semantic
     question, not a malformed graph. In V1 it is PROVABLY UNREACHABLE -- binary
     accrual pins every live path magnitude at exactly 1.0, so the equal-magnitude
     wash (§3.3.1c) always fires and the unequal branch is never entered. The
     guard therefore asserts the marker NEVER appears; it must never silently
     return a reward value. Fence B becomes live only under a fractional-magnitude
     regime, which is itself a versioned environment update under judge-versioning
     discipline, so that switch forces a re-ruling of this guard anyway.

Fences A and G are pure structural predicates (adjacency + speech order), knowable
at ingest. Fence A REUSES the judge's own Pass-1 primitives (`passes.build_context`,
`passes._union_find`, `passes._terminals`) so it cannot drift from what the judge
actually does. Fence B reads a WRITE-ONLY marker off a trace the judge already
produced -- it never re-derives the condition in a second resolver (a drifted
second resolver is exactly the silent-divergence bug the split exists to prevent).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import List

from model import Impact, Support, SPEECH_ORDER
from judge import passes

CONVERGENCE_MARKER = "CONVERGENCE_OUT_OF_SCOPE"


@dataclass
class RoundValidity:
    """Result of the STRUCTURAL admission check (Fences A + G). `ok` is True iff no
    structural fence tripped; `reasons` lists every rejection (keyed by fence tag)
    so a training harness can log which corner an agent reached. Fence B is NOT a
    validity fence and does not appear here -- see `assert_scope_ruled`."""
    ok: bool
    reasons: List[str] = field(default_factory=list)


def _offvocab_speech_nodes(rnd) -> List[str]:
    """FENCE G: node ids whose `speech` is not a canonical speech (∉ SPEECH_ORDER).
    Reads the raw round; no resolution needed. In the env this is satisfied by
    construction (every node's speech is stamped from the current slot), so at
    termination this is a belt-and-suspenders assertion."""
    vocab = set(SPEECH_ORDER)
    return [n.id for n in rnd.nodes if getattr(n, "speech", None) not in vocab]


def _multiterminal_components(rnd) -> List[List[str]]:
    """FENCE A: same-side Support components containing an Impact whose terminal-
    impact count (passes._terminals) is not exactly 1 -- the case that reaches the
    judge's flat multi-terminal fallback. Returns the offending components' member
    lists.

    REUSES the judge's own Pass-1 context and helpers so the enumeration is
    byte-identical to `passes._build_chains`; the only thing mirrored here is the
    same-side Support union loop, kept in lockstep with that source of truth. The
    legal-action generator calls this on the PROSPECTIVE round after a candidate
    action to enforce the same predicate locally and monotonically (Ruling 2)."""
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


def validate_round(rnd) -> RoundValidity:
    """STRUCTURAL admission (Fences A + G). Returns a RoundValidity with every
    tripped structural fence recorded. This is the check the termination step runs
    as an ASSERTION: under Phase 1 the generator makes both corners unreachable, so
    a non-`ok` result at termination is an environment bug, and the env must raise.
    Fence B is NOT run here (it is a scope guard -- see `assert_scope_ruled`)."""
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

    return RoundValidity(ok=not reasons, reasons=reasons)


def is_valid(rnd) -> bool:
    """Convenience boolean for the structural admission check (Fences A + G)."""
    return validate_round(rnd).ok


# --- Fence B: scope guard (separate concern, separate call site) --------------

def trace_has_convergence_marker(trace) -> bool:
    """Pure predicate: does this trace carry the write-only CONVERGENCE_OUT_OF_SCOPE
    marker? Separated from any judge run so the mechanism is unit-testable
    independently of whether any V1 graph can reach the branch (in binary V1 accrual
    every live path magnitude is exactly 1.0, so the equal-magnitude WASH fires and
    the unequal branch is unreachable -- future-proofing for a fractional-magnitude
    regime)."""
    return any(getattr(r, "kind", None) == CONVERGENCE_MARKER for r in trace)


def assert_scope_ruled(trace) -> None:
    """FENCE B as a SCOPE GUARD (Ruling 1). Reads the judge's own terminal trace and
    asserts the write-only CONVERGENCE_OUT_OF_SCOPE marker is ABSENT. In V1 the
    unequal-magnitude convergence branch (§3.3.1c) is unreachable, so this NEVER
    fires; if it ever does, the round entered an UNRULED judge corner and the
    episode must NOT be scored -- we raise rather than return any reward value (any
    number is something a policy can learn to chase). Becomes live only under a
    fractional-magnitude regime, itself a versioned env update that re-rules this
    guard.

    Takes the trace the termination step ALREADY produced (the judge is run once);
    it does not re-run the judge."""
    assert not trace_has_convergence_marker(trace), (
        "SCOPE GUARD (Fence B) tripped: judge emitted CONVERGENCE_OUT_OF_SCOPE "
        "(§3.3.1c unequal-magnitude convergence). This branch is UNREACHABLE in V1 "
        "binary accrual, so reaching it is an environment/judge-version bug -- the "
        "episode is in an unruled corner and must not be scored.")
