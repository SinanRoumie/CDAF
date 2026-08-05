"""Pinned V1 constants for the CDAF judge.

Single source of truth for every NUMERIC tunable the judge consults. The numeric
constants are pure (no side effects), so any pass can read them and the package
stays decoupled from everything app-side. The one non-numeric fact here -- the
speech order/side vocabulary -- is a DOMAIN fact owned by `model`, so it is
imported (not re-declared) below: the judge depends on the model, never the
reverse. See docs/judge_spec.md §1 (pinned decisions) and §2.1 (speech order/side).
"""

# The canonical speech vocabulary is a domain fact (the flow of a policy round),
# owned by `model.speeches` and imported here so there is ONE copy. Re-exported at
# this name so `from judge.config import SPEECH_ORDER, SPEECH_SIDE` (judge.passes,
# §2.1) keeps working -- the judge's accessor for a model-owned fact, never a second
# hand-synced literal that can silently drift.
from model.speeches import SPEECH_ORDER, SPEECH_SIDE   # noqa: E402  (re-export)

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
#   v4: attacker-liveness gates accrual (§3.1, §6) -- an attack contributes to its
#       target's DF-QuAD ONLY while the attack itself is live (extended by its
#       maker, or the side-agnostic union where that applies). An attack its maker
#       abandoned LAPSES: it is removed from the target's attacker set and
#       contributes nothing -- it is NOT scored "conceded" merely because the
#       opposing side did not answer it. The mitigation path (answer the attacker)
#       is unchanged.
#   v5: a determinate weigh DEFEATS the dispreferred member, and a defeated
#       attacker does not attack the winner (§6.5, generalized). Folded into the
#       same attacker gate: an attacker also drops from its target's accrual if
#       resolve({attacker, target}) is determinate for the target. The link case
#       (defeated turn) was already this rule in the polarity channel; v5 adds the
#       uniqueness/framework/defensive case (defeated non-unique dropped from the
#       uniqueness's attacker set). One mechanism, no per-type special-casing.
#   v6: framework selection through resolve() + turn-eligibility (§5.1-§5.4, §3.4).
#       Framework selection is by LIVE-SET CARDINALITY, never element order: the
#       live set is the frameworks with sigma >= threshold AND maker-extension
#       (§5.4); a determinate framework weigh removes the dispreferred framework
#       from the live set via the SAME resolve() impacts use (§6.5, no magnitude
#       floor -- indeterminate yields no defeat); winning_framework is the sole
#       live framework or None (a wash: zero or many live -> no gating, §5.2). The
#       single exclusion condition is BD-blocking anchoring (§5.3); defeat excludes
#       no chain directly. Turn-eligibility (§3.4): an OffensiveAttack flips iff
#       BOTH endpoints are offense-bearing (Link/Impact); any Uniqueness/Advocacy/
#       Framework/Weighing/BD endpoint is inert. DefensiveAttack is NOT governed by
#       this (a non-unique on a link, a *->framework attack stay live). Emits
#       FRAMEWORK_SELECT (once/round) + FRAMEWORK_DEFEAT (per defeat) + a per-chain
#       FRAMEWORK_GATE (§8).
#   v7: per-path impact aggregation (§3.3.1). A same-side Support component with a
#       single terminal impact is scored by folding the DISTINCT root->impact spine
#       paths converging on that impact (passes._spine_paths/_path_stats/
#       _aggregate_impact), not the old flat series product over the union spine:
#       (a) per-path liveness -- the impact survives iff >=1 complete path is
#       extended and carries non-zero magnitude, so a dead redundant path removes
#       only itself (r29/r31/r33); (b) same-sign redundancy folds by MAX, emitted as
#       ONE chain object per impact so the ballot never double-counts (r35);
#       (c) equal-magnitude sign-conflict at the shared impact WASHES to UNRESOLVED
#       (r34). Unequal-magnitude convergence and the multi-terminal component remain
#       OUT OF SCOPE (provisional fallbacks, §3.3.1c).
#   v8: owner-side win condition + BD anchor-membership (§7). The AFF win-gates read
#       OWNER-side offense -- a chain counts for AFF when its composed sign FAVORS
#       AFF (owner == AFF), the introducing side for an ordinary chain and the
#       OPPONENT for a captured/turned chain -- so AFF can win on a NEG disad it
#       turned, mirroring NEG winning on a captured AFF chain (judge._favored_side /
#       aff_owned). A live AFF Advocacy must be reachable over Support from that
#       owner-side offense (judge._advocacy_reachable), the completeness/in-scope/
#       N>eps floors are unchanged, and presumption stays NEG-asymmetric (T1/T1b/
#       T2'/T3/T4). A turning link joins the chain it CAPTURED for BD anchoring only,
#       via `anchor_members`, gated on live capture (eff_pol[target] == -1); union-
#       find and aggregation still read `members` (passes._build_chains).
#   v9: CORRECTIVE STAMP of the already-landed uniform-uniqueness work (§12). Every
#       post-world node carries its own satellite Uniqueness attack surface (the
#       root-only uniqueness of prior versions is retired): wired_uniqueness index +
#       multi_parent (4aa81b4), non-unique convergence poisoning + de-link kick-out
#       (5bb9c15), satellite uniqueness as a chain MEMBER not a root (7cbbb6d), spine-
#       rooted-uniqueness deletion (b57761c), and the v9 fixture migration (c6904df).
#       These landed in judge/passes.py + oracle fixtures WITHOUT bumping this
#       constant; v9 stamps them (no code change here -- the semantics already shipped).
#   v10: weigh reads the explicit `favors` pointer (§6.5). CROSS-SIDE weighs take the
#       Weighing's `favors` member directly (legacy default = the lone own-side member,
#       byte-identical); SAME-SIDE weighs are DESCRIPTIVE-ONLY (inert in resolve, zero
#       tally effect) and only enrich the WEIGH trace via `favors_source`
#       (resolve.preferred_node / favors_source). Model: Weighing carries `favors`,
#       round-tripped by serialize. No pre-existing fixture verdict moves (all 20
#       corpus weighs are cross-side).
#   v11: divergent chains (multi-terminal Support components). `_build_chains` emits
#       ONE chain PER terminal impact via `_aggregate_impact` (single-terminal is the
#       1-iteration case, byte-identical), so a shared trunk diverging to N impacts
#       yields N branch chains whose per-path magnitudes SUM at the ballot -- the trunk
#       sigma multiplies into each (efficient + fragile). The prior flat multi-terminal
#       fallback (union sigma product, one chain) is removed. Env-side companions (not
#       judged): Fence A deleted (multi-terminal now legal), new `connect` action, and
#       the C7 weigh/Fence-A guard reverted. No pre-existing scored fixture is
#       multi-terminal, so all prior verdicts are byte-identical.
JUDGE_VERSION = 11

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

# --- Speech order and side (§2.1) ---------------------------------------------
# SPEECH_ORDER / SPEECH_SIDE are imported from `model.speeches` at the top of this
# module (single source of truth). Order by SPEECH_ORDER index, NEVER by element
# index -- on-disk order is byte-fidelity, not chronology. The neg block is the
# single string "2NC/1NR" (one speech). SPEECH_SIDE's values are the same "AFF"/
# "NEG" strings as the AFF/NEG constants above.
