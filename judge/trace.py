"""Decision-trace record types for the CDAF judge (§8).

One dataclass per record kind, sharing a thin `TraceRecord` base. These are pure
data shapes -- NO logic; the later passes (J2+) construct and emit them as the
trace, an ordered list of decision records.

Each record's instance fields are exactly the fields listed for it in the §8
table. The record's name and the pass that emits it are carried as class-level
tags (`kind`, `pass_no`) -- not per-instance fields -- mirroring how model/
carries `ntype` / `etype` as ClassVars.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, List, Optional, Union

# A chain/link sign is +1, -1, or the unresolved sentinel '?'.
Sign = Union[int, str]


@dataclass
class TraceRecord:
    """Thin common base for all trace records. `kind` is the on-trace tag and
    `pass_no` is the emitting pass (§8 table); both are class-level, supplied by
    each concrete subclass, and never per-instance fields."""
    kind: ClassVar[str] = ""
    pass_no: ClassVar[str] = ""


@dataclass
class Drop(TraceRecord):
    """A node whose response window passed with no opposing clash -- conceded."""
    node_id: str
    owner: str
    intro_speech: str
    window_speech: str

    kind: ClassVar[str] = "DROP"
    pass_no: ClassVar[str] = "2"


@dataclass
class Unresolved(TraceRecord):
    """A node introduced in the final speech of its side -- opponent never had
    standing; cannot establish offense."""
    node_id: str
    owner: str
    intro_speech: str

    kind: ClassVar[str] = "UNRESOLVED"
    pass_no: ClassVar[str] = "2"


@dataclass
class ExtensionFail(TraceRecord):
    """A chain whose spine was not carried through one of its side's speeches."""
    chain_id: str
    missing_speech: str
    spine_node_id: str

    kind: ClassVar[str] = "EXTENSION_FAIL"
    pass_no: ClassVar[str] = "2/6"


@dataclass
class Magnitude(TraceRecord):
    """A node's DF-QuAD accrual: base tau and surviving sigma, with the resolved
    attacker/supporter strengths that produced it."""
    node_id: str
    base_tau: float
    surviving_sigma: float
    attackers: list
    supporters: list

    kind: ClassVar[str] = "MAGNITUDE"
    pass_no: ClassVar[str] = "3"


@dataclass
class PolarityFlip(TraceRecord):
    """A contested link whose effective polarity was resolved -- by an
    established preference or by DF-QuAD against the 0.5 threshold."""
    link_id: str
    from_sign: Sign
    to_sign: Sign
    sigma: float
    via: str  # "preference" | "dfquad"

    kind: ClassVar[str] = "POLARITY_FLIP"
    pass_no: ClassVar[str] = "3"


@dataclass
class InertAttack(TraceRecord):
    """An attack with no matching factor to operate on -- contributes nothing."""
    edge_id: str
    reason: str

    kind: ClassVar[str] = "INERT_ATTACK"
    pass_no: ClassVar[str] = "3"


@dataclass
class ConvergenceOutOfScope(TraceRecord):
    """WRITE-ONLY fence marker (§3.3.1c). Emitted when `_aggregate_impact` reaches
    the UNEQUAL-magnitude sign-conflict convergence branch -- a V1-out-of-scope
    case whose verdict is only provisional (the strongest single path carries the
    impact). It records that the branch was reached; it is NEVER read by any
    downstream pass or gate, so the verdict is identical whether or not this record
    is emitted (an inert descriptive tag, like JUDGE_VERSION). The RL environment's
    episode-init validator reads it OFF THE TRACE to REFUSE the round; the judge
    itself does not act on it. `pos_mag`/`neg_mag` are the two conflicting paths'
    magnitudes (unequal here, which is exactly what puts it out of scope)."""
    impact_id: str
    pos_mag: float
    neg_mag: float

    kind: ClassVar[str] = "CONVERGENCE_OUT_OF_SCOPE"
    pass_no: ClassVar[str] = "5"


@dataclass
class Chain(TraceRecord):
    """A resolved argument chain: sign and magnitude products -> delta.

    The trailing fields are descriptive enrichment (beyond the §8 core) so the
    trace reconstructs a decision without recomputation -- read by judge/rfd.py,
    populated by the judge. `side` is the INTRODUCING side (load-bearing: the
    ballot routes delta into aff_sum/neg_sum by it). `owner` is the descriptive
    OWNING side -- the side the composed sign favors: the introducing side for a
    normal chain, its opponent for a turned chain (§3.5), and "" when the sign is
    unresolved. It is derived and non-load-bearing (it never routes delta), surfaced
    so the trace carries owning-side directly. `collapse_reason` is None when the
    chain establishes offense; otherwise one of "extension_fail" / "sign_flip" /
    "defensive_kill" / "unresolved_sign". `responsible` names the node most
    responsible."""
    chain_id: str
    sign: Sign
    mag: float
    delta: float
    side: str = ""
    owner: str = ""
    extended: bool = True
    in_scope: bool = True
    collapse_reason: Optional[str] = None
    responsible: Optional[str] = None

    kind: ClassVar[str] = "CHAIN"
    pass_no: ClassVar[str] = "3"


@dataclass
class FrameworkSelect(TraceRecord):
    """Which framework governs the round (§5.1), emitted EXACTLY ONCE per round --
    including when no frameworks were authored (a silent framework pass is how a
    decorative weigh hides, §8). `winning_framework_id` is the sole live framework
    or None (a wash). `live` is the final live set, `defeated` the frameworks a
    determinate weigh removed. `via` is one of weigh / sole_survivor (a winner) or
    no_frameworks / none_survived / all_defeated / multiple_live (all washes)."""
    winning_framework_id: Optional[str]
    live: list
    defeated: list
    via: str
    reason: str = ""

    kind: ClassVar[str] = "FRAMEWORK_SELECT"
    pass_no: ClassVar[str] = "5"


@dataclass
class FrameworkDefeat(TraceRecord):
    """A framework removed from the live set by a determinate framework weigh
    (§5.1). `preferred_id` is the framework the weigh kept; `weighing_id` decided
    it. Defeat removes from the live set and excludes no chain directly (§5.3)."""
    framework_id: str
    weighing_id: str
    preferred_id: str

    kind: ClassVar[str] = "FRAMEWORK_DEFEAT"
    pass_no: ClassVar[str] = "5"


@dataclass
class FrameworkGate(TraceRecord):
    """A chain's binary framework-gating result (§5.3), emitted once per chain.
    `framework_id` is the winning framework or None (a wash gates nothing).
    `impact_id` is the chain's terminal impact -- the one carrying the scored
    delta (§8, never first-in-iteration). `anchors` (descriptive) is the chain's
    framework anchors: frameworks reachable by a Support path that does NOT
    traverse a BallotDirective. `in_scope == (framework_id is None or framework_id
    in anchors)` by construction (a single BD-blocking walk feeds both)."""
    chain_id: str
    impact_id: Optional[str]
    framework_id: Optional[str]
    in_scope: bool
    anchors: list = field(default_factory=list)

    kind: ClassVar[str] = "FRAMEWORK_GATE"
    pass_no: ClassVar[str] = "5"


@dataclass
class Weigh(TraceRecord):
    """A weighing claim's outcome and the preference (if any) it established.
    `pair` is the compared node ids; `overrode` is True when the honored
    preference favored a smaller-raw-delta impact (descriptive enrichment).
    `favors_source` records whether the preference came from the Weighing's explicit
    `favors` pointer ("explicit") or the structural legacy default ("legacy_default"),
    or is None when no preference resolved -- Phase-2 debugging aid to distinguish
    agent-set preference from a derived default (§6.5)."""
    weighing_id: str
    outcome: str  # "resolved" | "symmetric"
    preferred_node: Optional[str]
    via: str
    pair: List[str] = field(default_factory=list)
    overrode: bool = False
    favors_source: Optional[str] = None  # "explicit" | "legacy_default" | None

    kind: ClassVar[str] = "WEIGH"
    pass_no: ClassVar[str] = "5"


@dataclass
class BdValidate(TraceRecord):
    """A BallotDirective's validation result at the ballot. `side` is the BD's
    claimed direction (descriptive enrichment)."""
    bd_id: str
    result: str
    reason: str
    side: str = ""

    kind: ClassVar[str] = "BD_VALIDATE"
    pass_no: ClassVar[str] = "6"


@dataclass
class Ballot(TraceRecord):
    """The final decision: net offense, gates passed, and the winner.

    Trailing fields are descriptive enrichment so the RFD renders without
    recomputation: `reason_class` labels the decision (AFF offense / NEG offense
    / presumption / framework lock-out / AFF structural failure); `aff_sum` and
    `neg_sum` are the per-side delta sums (N == aff_sum - neg_sum); `decomposition`
    is the per-chain breakdown [{chain_id, side, delta, contributed}]."""
    N: float
    gates_passed: List[str]
    winner: str
    reason_class: str = ""
    aff_sum: float = 0.0
    neg_sum: float = 0.0
    decomposition: List[dict] = field(default_factory=list)

    kind: ClassVar[str] = "BALLOT"
    pass_no: ClassVar[str] = "6"


# Registry: on-trace `kind` string -> record class. Mirrors model/'s
# NODE_CLASSES / EDGE_CLASSES convention.
RECORD_CLASSES = {
    cls.kind: cls
    for cls in (
        Drop, Unresolved, ExtensionFail, Magnitude, PolarityFlip, InertAttack,
        ConvergenceOutOfScope,
        Chain, FrameworkSelect, FrameworkDefeat, FrameworkGate, Weigh, BdValidate,
        Ballot,
    )
}
RECORD_KINDS = tuple(RECORD_CLASSES.keys())
