"""Pinned V1 constants for the CDAF judge.

Single source of truth for every tunable the judge consults. Pure constants --
imports nothing, so any pass can read these without side effects and the package
stays decoupled from everything app-side. See docs/judge_spec.md §1 (pinned
decisions) and §2.1 (speech order/side).
"""

# --- Judge / environment version ----------------------------------------------
# Bumped when the judge's decision SEMANTICS change (not just refactors), because
# a version bump invalidates prior oracle verdicts and any learned RL policy.
#   v1: original passes (weighing = impact-only, ballot-stage preference).
#   v2: recursive weighing clash-resolution (§6.5) -- weighing is the general
#       same-type clash-breaker and can decide link/turn polarity, resolved by a
#       single well-founded recursion (meta-weighing -> weighing -> magnitude).
#   v3: turn offense (§3.5) -- a polarity flip PRESERVES MAGNITUDE (magnitude
#       changes only through defensive attack, §3.2); a turned chain generates
#       offense for the composed sign's favored side at the preserved magnitude
#       IFF every offense-bearing node on it is live by the SIDE-AGNOSTIC UNION of
#       both sides' liveness stamps (§6). Composition (double-turn -> AFF at
#       inherited strength) falls out of sign product x magnitude invariant (§3.3).
JUDGE_VERSION = 3

# --- Ballot values (§0) --------------------------------------------------------
# The ballot is binary. PRESUMPTION (below) resolves to one of these.
AFF = "AFF"
NEG = "NEG"

# --- Pinned numeric decisions (§1) --------------------------------------------

# Base node strength: presumed-true. Uncontested nodes stand at full strength;
# attacks erode, supports restore. (tau = 1.0)
TAU = 1.0

# A contested link's surviving magnitude >= this keeps its polarity; below flips.
POLARITY_THRESHOLD = 0.5

# Near-zero net offense guard: abs(N) <= EPSILON drains to presumption, so float
# noise can't manufacture an AFF win. Small and configurable.
EPSILON = 1e-9

# Presumption: hardcoded and uncontestable. Every indeterminate result drains here.
PRESUMPTION = NEG

# --- Speech order and side (§2.1, judge-owned) --------------------------------
# The model stores speech/side as free strings; the judge owns the canonical
# vocabulary and maps them. Order by SPEECH_ORDER index, NEVER by element index
# -- on-disk order is byte-fidelity, not chronology. The neg block is the single
# string "2NC/1NR" (one speech).
SPEECH_ORDER = ["1AC", "1NC", "2AC", "2NC/1NR", "1AR", "2NR", "2AR"]

SPEECH_SIDE = {
    "1AC": AFF, "2AC": AFF, "1AR": AFF, "2AR": AFF,
    "1NC": NEG, "2NC/1NR": NEG, "2NR": NEG,
}
