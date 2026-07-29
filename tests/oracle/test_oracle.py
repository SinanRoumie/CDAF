"""Verdict-oracle harness (judge_spec §11).

Each §11 round is built PROGRAMMATICALLY as a model.Round with liveness set per
the spec (not drawn in the builder), one test per round, asserting the stated
winner plus the key trace fact that localizes the decision to one pass. Each
round is surgical -- it isolates one mechanism.

Notes on spec-vs-mechanics (surfaced for review, not silently asserted):
  * Rounds 2/8 win for NEG via an AFF chain that COLLAPSED, so reason_class is
    "AFF structural failure" -- correct per §7's own definition (an AFF chain
    existed but collapsed / failed a gate). §11's looser word "presumption" for
    some of these is the label to tighten later; the judge follows §7.
  * Round 3's mitigation resolves to full magnitude (1.0), not ~0.5: a conceded
    counter removes the defender entirely, and V1 has no fractional strengths.
    The winner (AFF) is unchanged.
  * TURN OFFENSE (v3, §3.5). Round 4 now verifies NEG winning ON turn offense:
    a polarity flip PRESERVES MAGNITUDE (§3.2), so the turned AFF chain carries
    its surviving magnitude into a NEG-favoring chain (sign -1, mag > eps) that
    generates real NEG offense (N < -eps) -- not a 0-0 presumption. Round 4b
    shows a turn into a DEAD impact (no side carried it) generating nothing, and
    round 4c shows a DOUBLE turn composing back to AFF at inherited strength
    (sign +1, mag inherited) -- a version that zeroes magnitude on a flip fails
    4c. Round 4d is turn win-path (a): AFF keeps the link/impact live while NEG
    carries the turn, so the chain is live by the UNION of both sides -- a
    favored-side-only liveness check wrongly rejects it. Round 5 (turn not
    extended) is unchanged: the inherited impact is carried to 2NR by NO side, so
    no offense -> NEG. All winners here match v2; round 4's reason_class changed
    from "AFF structural failure" to "NEG offense" (the verdict NEG is unchanged)
    -- the v3 upgrade this milestone delivers.
"""

import os
import random

from model import (
    Round, Advocacy, Uniqueness, Link, Impact, Framework, Weighing, BallotDirective,
    Support, DefensiveAttack, OffensiveAttack, Comparison, SPEECH_ORDER, SPEECH_SIDE,
    serialize,
)
from judge import judge, passes
from judge.config import AFF, NEG, EPSILON

CONTESTED, CONCEDED = "contested", "conceded"

ORACLE_DIR = os.path.dirname(__file__)


def _load(name):
    return serialize.load(os.path.join(ORACLE_DIR, name))


# --- tiny builders (ids are per-round, deterministic) -------------------------

class _B:
    def __init__(self):
        self.k = 0

    def _id(self):
        self.k += 1
        return f"n{self.k}"

    def _full(self, side, speech):
        i = SPEECH_ORDER.index(speech)
        return {s: CONCEDED for j, s in enumerate(SPEECH_ORDER)
                if SPEECH_SIDE[s] == side and j >= i}

    def n(self, cls, side, speech, live=None, label="x"):
        return cls(id=self._id(), label=label, side=side, speech=speech,
                   liveness=(live if live is not None else self._full(side, speech)))

    def sup(self, a, b):
        return Support(id=self._id(), source=a.id, target=b.id)

    def datk(self, a, b):
        return DefensiveAttack(id=self._id(), source=a.id, target=b.id)

    def oatk(self, a, b):
        return OffensiveAttack(id=self._id(), source=a.id, target=b.id)

    def cmp(self, weigh, member):
        # A Comparison points FROM the ranking weigh TO a ranked member (§6.5).
        return Comparison(id=self._id(), source=weigh.id, target=member.id)


def _ballot(trace):
    return [r for r in trace if r.kind == "BALLOT"][0]


def _chains(trace):
    return [r for r in trace if r.kind == "CHAIN"]


def _polarity_via(trace, link_id):
    pf = [r for r in trace if r.kind == "POLARITY_FLIP" and r.link_id == link_id]
    return pf[0].via if pf else None


# --- §11.1 Clean uncontested advantage -> AFF ---------------------------------

def test_r1_clean_uncontested_advantage_aff():
    """1AC advocacy->uniqueness->link->impact, all extended AFF 2AC/1AR/2AR;
    NEG drops everything. Chain mag 1.0, sign +. N = 1.0 > eps. -> AFF."""
    ballot, trace = judge(_load("r1.json"))
    assert ballot == AFF
    bl = _ballot(trace)
    assert bl.N > EPSILON and bl.reason_class == "AFF offense"
    ch = _chains(trace)[0]
    assert ch.extended and ch.sign == 1 and abs(ch.mag - 1.0) < 1e-9
    assert ch.owner == AFF                          # normal chain: owner = introducing side


# --- §11.2 Conceded terminal defense -> NEG -----------------------------------

def test_r2_conceded_terminal_defense_neg():
    """As (1) but NEG reads a defensive attack on the link in 1NC and AFF drops
    it (NEG carries it through the block + 2NR). Link sigma -> 0, chain mag -> 0,
    N ~ 0. -> NEG."""
    ballot, trace = judge(_load("r2.json"))
    assert ballot == NEG
    # §7 (redrawn): a complete, extended, still-AFF-favoring chain driven to zero
    # magnitude = "AFF structural failure" (offense built, then lost). §11.2's
    # "presumption" was the loose word; §7 is authoritative.
    assert _ballot(trace).reason_class == "AFF structural failure"
    ch = _chains(trace)[0]
    assert ch.mag < EPSILON                         # link killed -> chain collapsed


# --- §11.3 Answered defense -> mitigation -> AFF -------------------------------

def test_r3_answered_defense_mitigation_aff():
    """As (2) but AFF answers the defense in 2AC (a conceded counter), removing
    the defender; the link survives and the advantage stands. -> AFF (in V1 the
    magnitude restores fully rather than to ~0.5)."""
    ballot, trace = judge(_load("r3.json"))
    assert ballot == AFF
    assert _ballot(trace).reason_class == "AFF offense"
    ch = _chains(trace)[0]
    assert ch.mag > EPSILON                         # defense mitigated -> chain survives


# --- §11.4 Link turn, chain live -> NEG ON turn offense (v3, §3.5) ------------

def test_r4_link_turn_generates_neg_offense():
    """NEG turns the 1AC link in 1NC (an OffensiveAttack); AFF concedes and walks
    away, but NEG carries BOTH the turned link and the inherited impact through
    the block + 2NR (side-agnostic union liveness, §6).

    TURN OFFENSE (v3, §3.5). The flip PRESERVES MAGNITUDE (§3.2): the offensive
    attack is a sign-channel operation, so the AFF link keeps its surviving
    magnitude (1.0, no defensive attack on it) and carries it into a turned chain
    at NEG's sign. Because every offense-bearing node on that chain is live by the
    UNION of both sides' stamps, the turned chain GENERATES real NEG offense --
    sign -1, mag > eps, N < -eps -- rather than merely zeroing the AFF advantage.
    This is NEG winning ON turn offense, not a 0-0 presumption win."""
    ballot, trace = judge(_load("r4.json"))
    assert ballot == NEG
    # NEG wins ON turn offense: the turned chain carries NEG-favoring sign at real
    # magnitude, so N is genuinely negative (not a 0-0 presumption).
    bl = _ballot(trace)
    assert bl.reason_class == "NEG offense" and bl.N < -EPSILON
    ch = _chains(trace)[0]
    assert ch.sign == -1 and ch.mag > EPSILON      # magnitude PRESERVED across the flip
    assert ch.extended                             # union liveness kept the turned nodes live
    assert ch.side == AFF and ch.owner == NEG      # introducing side AFF; turn hands offense to NEG
    # the flip carried magnitude, so the AFF chain is not zeroed -- it contributes
    # NEG offense to the ballot decomposition:
    d = [x for x in bl.decomposition if x["chain_id"] == ch.chain_id][0]
    assert d["contributed"] and d["delta"] < -EPSILON


def test_r4b_turn_into_dead_impact_generates_nothing_neg():
    """As r4 but the inherited impact is DEAD -- no side carried it past its 1AC
    introduction (empty NEG liveness on the impact). The turn still neutralizes
    the AFF advantage (the link flips), but with the impact unextended the turned
    chain generates NOTHING: no NEG offense (N ~ 0), decided by presumption. This
    is the union-liveness gate (§3.5, §6): a turn into a node no one kept live
    produces no offense."""
    ballot, trace = judge(_load("r4b.json"))
    assert ballot == NEG
    # The turn neutralizes but generates NO NEG offense -- N is ~0, not negative:
    bl = _ballot(trace)
    assert bl.N > -EPSILON and bl.reason_class != "NEG offense"
    assert any(r.kind == "EXTENSION_FAIL" for r in trace)   # dead impact failed the union gate
    ch = _chains(trace)[0]
    assert not ch.extended
    assert not any(x["contributed"] for x in bl.decomposition)   # nothing credited


def test_r4c_double_turn_returns_to_aff_inherited_strength():
    """DOUBLE TURN: NEG reads BOTH a link turn and an impact turn (two
    OffensiveAttacks); AFF concedes each but re-extends the double-turned
    advantage (re-engagement, §6). Two flips COMPOSE by sign product (§3.3):
    (-1)(-1) = +1, back to AFF polarity. Because each flip PRESERVES MAGNITUDE
    (§3.2), the chain inherits full strength (mag 1.0) rather than zeroing -- so
    AFF wins at inherited strength on the double-cross. NO special-casing: this
    falls straight out of sign product x the magnitude invariant. A judge that
    zeroed magnitude on a flip would give mag 0 -> N 0 -> NEG here; the AFF win
    is the regression test for magnitude preservation through two flips."""
    ballot, trace = judge(_load("r4c.json"))
    assert ballot == AFF
    bl = _ballot(trace)
    assert bl.reason_class == "AFF offense" and bl.N > EPSILON
    ch = _chains(trace)[0]
    assert ch.sign == 1                            # two flips composed back to +1
    assert abs(ch.mag - 1.0) < 1e-9                # magnitude INHERITED through both flips
    assert ch.delta > EPSILON
    assert ch.owner == AFF                         # double turn: composed sign favors AFF again


def test_r4d_turn_live_by_union_win_path_a_neg():
    """Turn win-path (a) (§3.5, §6 union): AFF KEEPS the link and impact live
    through its OWN speeches (still contesting) while NEG carries the TURN. The
    link/impact records therefore hold AFF stamps, not NEG stamps -- yet the turned
    chain still counts, because every node is live by the UNION of both sides
    (kept alive by SOMEONE), not by the turning side alone. This is the case a
    'favored-side-only' liveness check wrongly rejects; the union check must accept
    it -> NEG on turn offense."""
    ballot, trace = judge(_load("r4d.json"))
    assert ballot == NEG
    ch = _chains(trace)[0]
    assert ch.sign == -1 and ch.mag > EPSILON and ch.extended   # union kept the turned chain live
    assert _ballot(trace).reason_class == "NEG offense" and _ballot(trace).N < -EPSILON


# --- §11.5 Link turn, NOT extended -> NEG -------------------------------------

def test_r5_link_turn_not_extended_neg():
    """As (4) but NEG fails to carry the inherited impact in 2NR (its liveness
    stops at the block). The chain still fails §6. -> NEG."""
    ballot, trace = judge(_load("r5.json"))
    assert ballot == NEG
    # §7 (redrawn) + Ruling 3: the AFF link was turned away and NEG did not carry
    # the turn, so NOBODY established offense and the chain failed extension (no
    # complete AFF chain stood). A flip is never structural failure -> presumption.
    assert _ballot(trace).reason_class == "presumption"
    assert any(r.kind == "EXTENSION_FAIL" for r in trace)


# --- §11.6 Framework lock-out -> NEG ------------------------------------------

def test_r6_framework_lockout_neg():
    """NEG wins a framework (unattacked + extended) with no support path to AFF's
    only impact, so that impact is out of scope. AFF has zero in-scope impacts.
    -> NEG (framework lock-out)."""
    ballot, trace = judge(_load("r6.json"))
    assert ballot == NEG
    assert _ballot(trace).reason_class == "framework lock-out"
    gates = {(r.impact_id, r.in_scope) for r in trace if r.kind == "FRAMEWORK_GATE"}
    assert ("im", False) in gates                  # AFF impact locked out


# --- §11.8 No-window new 2AR offense -> inert ---------------------------------

def test_r8_no_window_new_2ar_offense_inert():
    """AFF's terminal impact is introduced fresh in the 2AR (final speech); NEG
    never had standing and the attachment was not contested entering the 2AR, so
    it is UNRESOLVED (inert) -- establishes no offense. -> NEG."""
    ballot, trace = judge(_load("r8.json"))
    assert ballot == NEG
    # §7 (redrawn): the 2AR impact is UNRESOLVED (no window) -- offense never
    # legitimately existed, no complete AFF chain was ever established -> presumption
    # (not "AFF structural failure", which requires offense built then lost).
    assert _ballot(trace).reason_class == "presumption"
    assert any(r.kind == "UNRESOLVED" and r.node_id == "im" for r in trace)


# --- Recursive weighing (§6.5) -- clash resolution decides polarity -----------

def test_r9_turned_link_saved_by_determinate_weigh_aff():
    """A determinate won link-weigh decides the polarity clash for AFF: the link
    keeps its polarity via PREFERENCE (not sigma), the defeated turn drops from
    the chain magnitude, the chain survives, and AFF takes the ballot."""
    ballot, trace = judge(_load("r9.json"))
    assert ballot == AFF
    assert _ballot(trace).reason_class == "AFF offense"
    assert _polarity_via(trace, "lk") == "preference"     # weigh decided, not the 0.5 sigma rule
    # the WEIGH record shows the link-weigh resolved in favour of the AFF link:
    weigh = [r for r in trace if r.kind == "WEIGH" and r.outcome == "resolved"]
    assert weigh and weigh[0].preferred_node == "lk" and weigh[0].via == "preference"
    ch = _chains(trace)[0]
    assert ch.sign == 1 and ch.mag > EPSILON               # link kept polarity, magnitude survived


def test_r10_turned_link_indeterminate_falls_to_magnitude_neg():
    """Same round but NEG counter-weighs the SAME pair oppositely with no meta-
    weigh: the weighing layer has no lone survivor (§6.5 case b), so the clash
    falls to DF-QuAD magnitude -- the turn resolves as it did before the upgrade
    (link flips via the sigma threshold) and NEG takes the ballot."""
    ballot, trace = judge(_load("r10.json"))
    assert ballot == NEG
    # r10 investigation (case a): the symmetric weigh falls to magnitude and the
    # link FLIPS to NEG (sign -1, mag 1.0). But r10 authors only an AFF BD, so the
    # NEG-favoring turned offense is orphaned (BD_VALIDATE fails "offense favors NEG,
    # BD claims AFF") -- NOT a bug (r4 shows a NEG BD would validate it). Nobody
    # established scoring offense -> presumption. A flip is never structural failure.
    assert _ballot(trace).reason_class == "presumption"
    assert _polarity_via(trace, "lk") == "dfquad"         # fell back to the 0.5 sigma threshold
    assert all(r.outcome == "symmetric" for r in trace if r.kind == "WEIGH")


def test_r11_recursive_meta_weigh_breaks_tie_aff():
    """Depth-2: AFF and NEG weigh {link, turn} oppositely (would tie), but an AFF
    META-weigh ranks the two weighs and defeats NEG's. That leaves AFF's weigh the
    lone survivor at the base clash -> determinate -> the link is saved -> AFF.
    This is the case a fixed depth-2 check gets wrong; it must stay a recursion,
    so this is a permanent regression test that resolve is not depth-limited."""
    ballot, trace = judge(_load("r11.json"))
    assert ballot == AFF
    assert _ballot(trace).reason_class == "AFF offense"
    assert _polarity_via(trace, "lk") == "preference"     # meta broke the tie -> determinate


# --- Attacker-liveness gate (§3.1, §6 -- v4) ----------------------------------

def _uni_attackers(els, uni_id):
    """The uniqueness's DF-QuAD attacker-id set AT THE BALLOT -- after the liveness
    gate (pass 1) and the weigh-defeat gate (pass 4), which is where a defeated
    attacker is removed from the winner's accrual (§6.5, v5)."""
    ctx = passes.build_context(Round(elements=els, version=2))
    passes.pass2_drops(ctx)
    passes.pass3_accrual(ctx)
    passes.pass4_weighing_towers(ctx)
    return {a for a, _e in ctx.attackers_by_target.get(uni_id, [])}


def test_r12_dropped_nonunique_lapses_aff():
    """THE BUG FIX (§3.1, §6 -- v4). NEG reads a non-unique on the AFF uniqueness
    in 1NC and does NOT extend it (liveness {1NC} only). A dropped attack LAPSES:
    it is removed from the uniqueness's attacker set and contributes nothing -- it
    is NOT scored 'conceded' just because AFF didn't answer it. The uniqueness
    survives at sigma 1.0, the chain holds at mag 1.0, and AFF wins cleanly."""
    els = list(_load("r12.json").elements)
    assert "nonuniq" not in _uni_attackers(els, "uni")   # lapsed: NOT in the attacker set
    ballot, trace = judge(Round(elements=els, version=2))
    assert ballot == AFF
    bl = _ballot(trace)
    assert bl.reason_class == "AFF offense" and bl.N > EPSILON
    ch = _chains(trace)[0]
    assert ch.extended and abs(ch.mag - 1.0) < 1e-9        # uniqueness restored -> chain intact
    assert any(r.kind == "INERT_ATTACK" and "lapsed" in r.reason for r in trace)


def test_r13_extended_nonunique_still_contests_neg():
    """CONTRAST. Same round but NEG DOES extend the non-unique through 2NC/1NR and
    2NR (full NEG liveness). It stays LIVE, so it remains in the uniqueness's
    attacker set at full strength and drives the uniqueness to 0 -> chain mag 0 ->
    NEG. The gate removes only attacks the MAKER abandoned, never live ones."""
    els = list(_load("r13.json").elements)
    assert "nonuniq" in _uni_attackers(els, "uni")       # live: DOES contest
    ballot, trace = judge(Round(elements=els, version=2))
    assert ballot == NEG
    # §7 (redrawn): complete extended still-AFF-favoring chain, uniqueness driven
    # to zero magnitude -> "AFF structural failure" (offense built, then lost).
    assert _ballot(trace).reason_class == "AFF structural failure"
    ch = _chains(trace)[0]
    assert ch.mag < EPSILON                                # uniqueness zeroed -> chain collapsed


def test_r14_answered_nonunique_still_mitigates_aff():
    """MITIGATION STILL WORKS (oracle 3 path). The non-unique is extended (live)
    AND AFF answers it in 2AC (attacks the attacker, conceded). The gate leaves the
    live attack in the set; the leaves-first DF-QuAD then reduces it via AFF's
    counter (in V1 a conceded counter removes it entirely), so the uniqueness
    survives and AFF wins. The fix did not break the answer path."""
    els = list(_load("r14.json").elements)
    assert "nonuniq" in _uni_attackers(els, "uni")       # live (answered, not lapsed)
    ballot, trace = judge(Round(elements=els, version=2))
    assert ballot == AFF                                    # answered down -> uniqueness restored
    assert _ballot(trace).reason_class == "AFF offense"
    ch = _chains(trace)[0]
    assert ch.mag > EPSILON


# --- Won weigh defeats the attacker across types (§6.5 -- v5) ------------------

def test_r15_won_uniqueness_weigh_defeats_nonunique_aff():
    """THE BUG FIX (§6.5 -- v5). AFF WINS a determinate weigh over the uniqueness
    clash {n1, n7} (preferred = the AFF uniqueness). The defeated non-unique n7 does
    NOT attack the winner: it is removed from n1's DF-QuAD attacker set, n1 survives
    at sigma 1.0, the chain holds at mag 1.0, and AFF wins. Previously the weigh was
    decorative -- n7 still zeroed n1. Same rule as the link case (r9), now over
    uniquenesses."""
    els = list(_load("r15.json").elements)
    assert "nonuniq" not in _uni_attackers(els, "uni")   # defeated -> NOT in the attacker set
    ballot, trace = judge(Round(elements=els, version=2))
    assert ballot == AFF
    assert _ballot(trace).reason_class == "AFF offense"
    assert any(r.kind == "INERT_ATTACK" and r.reason == "defeated by weigh" for r in trace)
    ch = _chains(trace)[0]
    assert ch.extended and abs(ch.mag - 1.0) < 1e-9        # uniqueness restored -> chain intact
    won = [r for r in trace if r.kind == "WEIGH" and r.outcome == "resolved"]
    assert won and won[0].preferred_node == "uni"         # the weigh resolved for the uniqueness


def test_r16_lost_uniqueness_weigh_nonunique_still_attacks_neg():
    """CONTRAST / fallback. NEG WINS the weigh (preferred = the non-unique), so the
    clash does NOT resolve for the uniqueness: n7 is not defeated, stays in n1's
    attacker set, and drives n1 to 0 -> chain mag 0 -> NEG. A weigh only removes the
    attacker when the winner is the TARGET; a lost/indeterminate weigh leaves the
    attack contesting (the no-weigh fallback is r13)."""
    els = list(_load("r16.json").elements)
    assert "nonuniq" in _uni_attackers(els, "uni")       # not defeated -> still contests
    ballot, trace = judge(Round(elements=els, version=2))
    assert ballot == NEG
    # §7 (redrawn): complete extended still-AFF-favoring chain, uniqueness driven
    # to zero magnitude -> "AFF structural failure" (offense built, then lost).
    assert _ballot(trace).reason_class == "AFF structural failure"
    ch = _chains(trace)[0]
    assert ch.mag < EPSILON                                # uniqueness zeroed -> chain collapsed


# --- Framework wash: banked round (a), regression lock (§5.2, §5.4) -----------

def test_fw_wash_no_extend_regression_lock():
    """BANKED ROUND (a) -- two mirror framework chains (advocacy->link->impact,
    each impact -Support-> its own Framework -Support-> its own BD); the
    frameworks are introduced but NOT maker-extended past introduction; no weigh.

    v6 (updated deliberately from the pre-v6 lock -- verdict stable, PATH
    changed, per the milestone gate). The VERDICT is unchanged, (NEG,
    "presumption"), but it is now reached by the WASH path, not the accidental
    early-return the pre-v6 code took:
      * selection is by live-set CARDINALITY: len(live) == 0 because BOTH
        frameworks fail maker-extension (§5.4) -> winning_framework is None as a
        wash, not because a selection loop happened to find nothing;
      * two frameworks WERE authored (they exist, they just failed to extend),
        so FRAMEWORK_SELECT.via is "none_survived", NOT "no_frameworks" (which
        would mean none authored at all) -- the distinction this fixture exists
        to pin;
      * FRAMEWORK_GATE is now emitted once per chain with framework_id=None and
        in_scope=True (a deliberate no-op gate, §5.2/§5.3), both chains stay in
        scope, each contributes +1.0, N ~ 0 -> presumption as the ordinary
        backstop, never lock-out (a wash locks nobody out, §5.3).
    (Pre-v6 this asserted NO FRAMEWORK_GATE at all -- gating skipped by early
    return. That marker is intentionally replaced below.)
    """
    ballot, trace = judge(_load("fw_wash_no_extend.json"))
    assert ballot == NEG
    bl = _ballot(trace)
    assert bl.reason_class == "presumption"
    assert abs(bl.N) <= EPSILON                            # N ~ 0
    contributed = [d for d in bl.decomposition if d["contributed"]]
    assert len(contributed) == 2                           # BOTH chains contribute
    assert {d["side"] for d in contributed} == {AFF, NEG}
    # v6 wash path: winning_framework is None BY CARDINALITY, via "none_survived"
    # (two authored frameworks, neither maker-extended), and a per-chain no-op
    # gate is emitted (framework_id None, in_scope True).
    sel = [r for r in trace if r.kind == "FRAMEWORK_SELECT"]
    assert len(sel) == 1 and sel[0].winning_framework_id is None
    assert sel[0].via == "none_survived"
    gates = [r for r in trace if r.kind == "FRAMEWORK_GATE"]
    assert len(gates) == 2
    assert all(g.framework_id is None and g.in_scope for g in gates)
    # FLAG 2: the im->F->BD anchor path is pinned, not incidental -- each impact
    # reaches its OWN framework by a Support path that does not cross a BD.
    anchors = {g.chain_id: set(g.anchors) for g in gates}
    assert any("F_aff" in a for a in anchors.values())
    assert any("F_neg" in a for a in anchors.values())


# --- §11.21 Permutation determinism (RED pre-v6) ------------------------------

def test_r21_permutation_determinism():
    """Round 21 (§11): fw_weigh_lockout.json judged under N random permutations
    of the element list. Ballot AND reason_class must be identical across all of
    them -- the judge is a function of the round, not of on-disk order (§2.1,
    §5.1).

    RED pre-v6: `winning_framework` is chosen by iteration order (the first
    surviving+extended Framework in ctx.nodes.values()), so a permutation that
    puts F_neg's node before F_aff's flips the winning framework and the verdict
    (observed both ('AFF','AFF offense') and ('NEG','framework lock-out')). Goes
    green when selection routes through resolve() by live-set cardinality (§5.1).
    """
    base = list(_load("fw_weigh_lockout.json").elements)
    rng = random.Random(20240710)                          # seeded -> reproducible
    outcomes = set()
    for _ in range(40):
        perm = base[:]
        rng.shuffle(perm)
        ballot, trace = judge(Round(elements=perm, version=2))
        outcomes.add((ballot, _ballot(trace).reason_class))
    assert len(outcomes) == 1, f"order-dependent verdict: {sorted(outcomes)}"


# --- Framework-channel oracle rounds (§11.17-§11.25, v6) ----------------------

def _select(trace):
    return [r for r in trace if r.kind == "FRAMEWORK_SELECT"][0]


def _run_ctx(els):
    """Run all passes and return the ctx, for σ / extension / attacker inspection."""
    ctx = passes.build_context(Round(elements=els, version=2))
    passes.pass2_drops(ctx); passes.pass3_accrual(ctx)
    passes.pass4_weighing_towers(ctx); passes.pass5_clashes(ctx)
    passes.pass6_framework(ctx)
    return ctx


def test_r17_framework_weigh_lockout_neg():
    """§11.17: AFF impact anchored to F_aff, NEG mirror anchored to F_neg, NEG
    weighs {F_aff, F_neg} preferring F_neg. resolve() is determinate for F_neg ->
    F_aff defeated -> live = {F_neg} -> F_neg gates -> AFF's chain is not anchored
    to it -> no in-scope AFF impact. Pre-v6 this returned AFF because selection
    never called resolve().

    reason_class is "NEG offense", NOT "framework lock-out" (v6, decided): the
    NEG mirror chain is anchored to the winning F_neg, so it is IN SCOPE and
    carries N = -1.0. Lock-out is reserved for NEG winning FOR WANT of AFF offense
    (N ~ 0, e.g. r6); here NEG has its own in-scope offense, so it wins ON offense.
    §11.17's title 'lock-out' names the mechanism (F_neg shuts AFF out); the
    reason_class follows §7's narrowed rule and reads the offense that actually
    decided N. The framework machinery (weigh-defeat of F_aff, F_neg governing)
    still carries the round -- asserted on FRAMEWORK_SELECT below."""
    trace = judge(_load("fw_weigh_lockout.json"))[1]
    assert judge(_load("fw_weigh_lockout.json"))[0] == NEG
    assert _ballot(trace).reason_class == "NEG offense"
    sel = _select(trace)
    assert sel.via == "weigh" and sel.winning_framework_id == "F_neg"
    assert "F_aff" in sel.defeated
    # FLAG 2 anchor path pinned: NEG impact reaches F_neg without crossing a BD.
    gates = [g for g in trace if g.kind == "FRAMEWORK_DEFEAT"]
    assert gates and gates[0].framework_id == "F_aff" and gates[0].preferred_id == "F_neg"
    anchors = {frozenset(g.anchors) for g in trace if g.kind == "FRAMEWORK_GATE"}
    assert any("F_neg" in a for a in anchors) and any("F_aff" in a for a in anchors)


def test_r18_framework_wash_multiple_live_neg():
    """§11.18: as r17 but NO weighing node. Both frameworks live, len(live) == 2,
    winning_framework None -> wash (via multiple_live), no gating. Both chains in
    scope, both delta +1, N = 0. NEG (presumption) -- by the wash path, not
    lock-out and not any σ tiebreak. (σ eliminates below threshold, never ranks:
    the selector never compares σ between live frameworks -- `live` is a set,
    cardinality only -- so a σ difference among live frameworks cannot move the
    verdict; V1's binary accrual can't author a clean 0.7, but the invariance is
    structural, not fixture-dependent.)"""
    b = _B()
    adv = b.n(Advocacy, AFF, "1AC"); lka = b.n(Link, AFF, "1AC"); ima = b.n(Impact, AFF, "1AC")
    faff = b.n(Framework, AFF, "1AC"); bda = b.n(BallotDirective, AFF, "2AR")
    lkn = b.n(Link, NEG, "1NC"); imn = b.n(Impact, NEG, "1NC")
    fneg = b.n(Framework, NEG, "1NC"); bdn = b.n(BallotDirective, NEG, "2NR")
    ballot, trace = judge(Round(elements=[
        adv, lka, ima, faff, bda, lkn, imn, fneg, bdn,
        b.sup(adv, lka), b.sup(lka, ima), b.sup(ima, faff), b.sup(faff, bda),
        b.sup(lkn, imn), b.sup(imn, fneg), b.sup(fneg, bdn)], version=2))
    assert ballot == NEG
    bl = _ballot(trace)
    assert bl.reason_class == "presumption" and abs(bl.N) <= EPSILON
    sel = _select(trace)
    assert sel.via == "multiple_live" and sel.winning_framework_id is None
    assert len(sel.live) == 2


def test_r19_dual_anchored_impact_survives_defeat_aff():
    """§11.19: AFF impact anchored to BOTH F_util (AFF) and F_sv (NEG). NEG weighs
    F_sv > F_util (determinate). F_util defeated; live = {F_sv}; the AFF impact is
    still anchored to F_sv -> in scope. AFF (AFF offense). Rejecting a framework is
    not rejecting the argument that linked into it (§5.3: defeat is not
    exclusion)."""
    ballot, trace = judge(_load("r19.json"))
    assert ballot == AFF
    bl = _ballot(trace)
    assert bl.reason_class == "AFF offense" and bl.N > EPSILON
    sel = _select(trace)
    assert sel.via == "weigh" and sel.winning_framework_id == "fsv"
    assert "futil" in sel.defeated


def test_r20_kritik_offense_independent_of_kicked_framework_neg():
    """§11.20: framework kritik whose offense is independent of the criticized
    framework. AFF's advantage is anchored to F_util; AFF KICKS F_util (stops
    extending it after 2AC), so F_util fails MAKER-extension at FULL σ and leaves
    the live set. NEG's kritik offense (racism impact) is anchored to F_sv, never
    to F_util, so it is untouched: F_sv governs, the racism impact is in scope,
    delta = link × impact (the framework contributes no factor). NEG (NEG offense).

    Two spec assertions carry the round: σ(F_util) appears in NO chain product
    (frameworks are never spine reps, §3.6), and the verdict is INVARIANT to
    deleting F_util entirely -- the kritik's offense depends only on its own
    anchor F_sv.

    NOTE (V1 σ-regime, surfaced): §11.20's text also draws the kritik's
    DefensiveAttack onto F_util. In V1's binary accrual a LIVE conceded defensive
    attack drives σ to 0, which would unseat F_util by σ too (that is r23) and
    blur the "at full σ" isolation this round exists for. There is no clean
    intermediate σ (below full, above threshold, with live offense) in V1. So this
    round isolates the MAKER-EXTENSION kick at full σ and omits the σ-attack; the
    two-hat DefensiveAttack unseat is exercised at full extension by r23. Verdict
    and both structural claims are unaffected."""
    b = _B()
    adv = b.n(Advocacy, AFF, "1AC"); lka = b.n(Link, AFF, "1AC"); ima = b.n(Impact, AFF, "1AC")
    futil = b.n(Framework, AFF, "1AC", {"1AC": CONCEDED, "2AC": CONCEDED}, label="F_util")  # kicked
    bda = b.n(BallotDirective, AFF, "2AR")
    klink = b.n(Link, NEG, "1NC", label="util is racist (kritik link)")
    kim = b.n(Impact, NEG, "1NC", label="racism")
    fsv = b.n(Framework, NEG, "1NC", label="F_sv"); bdn = b.n(BallotDirective, NEG, "2NR")
    core = [adv, lka, ima, futil, bda, klink, kim, fsv, bdn,
            b.sup(adv, lka), b.sup(lka, ima), b.sup(ima, futil), b.sup(futil, bda),
            b.sup(klink, kim), b.sup(kim, fsv), b.sup(fsv, bdn)]
    ballot, trace = judge(Round(elements=core, version=2))
    assert ballot == NEG
    bl = _ballot(trace)
    assert bl.reason_class == "NEG offense" and bl.N < -EPSILON
    assert _select(trace).winning_framework_id == fsv.id
    # F_util unseated by maker-extension at FULL σ (not by σ).
    ctx = _run_ctx(core)
    assert ctx.sigma[futil.id] >= 0.5 and not passes.node_extension_ok(futil)[0]
    assert all(futil.id not in ch["spine_reps"] for ch in ctx.chains)   # σ in no chain product
    # INVARIANT to deleting F_util entirely (kritik offense depends only on F_sv):
    without = [e for e in core if getattr(e, "id", None) not in {futil.id}
               and getattr(e, "source", None) != futil.id
               and getattr(e, "target", None) != futil.id]
    b2, t2 = judge(Round(elements=without, version=2))
    assert b2 == NEG and _ballot(t2).reason_class == "NEG offense"


def test_r23_kritik_unseats_framework_by_defensive_attack_neg():
    """§11.23: as r20 but AFF KEEPS F_util extended throughout (full AFF
    liveness); NEG's two-hat kritik LINK drives σ(F_util) below threshold via a
    DefensiveAttack. F_util leaves live on the σ path (not maker-extension);
    live = {F_sv} -> F_sv gates; NEG's racism offense (anchored to F_sv) is in
    scope. NEG (NEG offense). The SAME link both attacked F_util and rooted the
    scoring chain (the two-hat spine rep). Contrast r20: r20 unseats by
    maker-extension at full σ; r23 unseats by σ at full extension -- same verdict,
    different mechanism."""
    b = _B()
    adv = b.n(Advocacy, AFF, "1AC"); lka = b.n(Link, AFF, "1AC"); ima = b.n(Impact, AFF, "1AC")
    futil = b.n(Framework, AFF, "1AC", label="F_util")            # FULL AFF liveness
    bda = b.n(BallotDirective, AFF, "2AR")
    klink = b.n(Link, NEG, "1NC", label="util is racist (two-hat)")
    kim = b.n(Impact, NEG, "1NC", label="racism")
    fsv = b.n(Framework, NEG, "1NC", label="F_sv"); bdn = b.n(BallotDirective, NEG, "2NR")
    els = [adv, lka, ima, futil, bda, klink, kim, fsv, bdn,
           b.sup(adv, lka), b.sup(lka, ima), b.sup(ima, futil), b.sup(futil, bda),
           b.sup(klink, kim), b.sup(kim, fsv), b.sup(fsv, bdn),
           b.datk(klink, futil)]                                 # HAT 2: unseats σ(F_util)
    ballot, trace = judge(Round(elements=els, version=2))
    assert ballot == NEG
    bl = _ballot(trace)
    assert bl.reason_class == "NEG offense" and bl.N < -EPSILON
    assert _select(trace).winning_framework_id == fsv.id
    ctx = _run_ctx(els)
    assert passes.node_extension_ok(futil)[0]                    # fully extended...
    assert ctx.sigma[futil.id] < 0.5                            # ...but unseated by σ
    # two-hat: the SAME link attacked F_util AND is a spine rep of the NEG chain.
    assert klink.id in {a for a, _e in ctx.attackers_by_target.get(futil.id, [])}
    assert any(klink.id in ch["spine_reps"] for ch in ctx.chains)


def test_r22_offense_at_framework_inert_aff():
    """§11.22: an OffensiveAttack targeting a Framework node. Turn-eligibility
    (§3.4): a framework bears no polarity, so the edge is inert -- emit
    INERT_ATTACK, σ unchanged, no POLARITY_FLIP. Same treatment as offense aimed
    at an Advocacy. F_aff governs and anchors the AFF chain -> AFF (AFF offense)."""
    els = list(_load("r22.json").elements)
    ballot, trace = judge(Round(elements=els, version=2))
    assert ballot == AFF
    assert _ballot(trace).reason_class == "AFF offense"
    assert any(r.kind == "INERT_ATTACK" and "3.4" in r.reason for r in trace)
    assert not any(r.kind == "POLARITY_FLIP" for r in trace)
    assert _run_ctx(els).sigma["fw"] >= 0.5                    # σ untouched by the inert offense


def test_r24_nonunique_on_link_dead_not_turned_neg():
    """§11.24: no separate uniqueness node; NEG reads a DefensiveAttack FROM a
    Uniqueness directly onto the AFF link (the non-unique), conceded and extended.
    DefensiveAttack is NOT governed by turn-eligibility (§3.4), so it stays live
    and drives the link's magnitude to 0 -- the link is DEAD, not turned: sign
    stays +1, magnitude -> 0, emit MAGNITUDE not POLARITY_FLIP. A uniqueness can
    mitigate a link but never manufactures offense from it. NEG.

    reason_class is 'AFF structural failure' (an AFF chain existed and collapsed,
    §7) -- §11.24's looser word 'presumption' is the label to tighten later, the
    same §11-vs-§7 gap already noted for r2/r8; the judge follows §7."""
    b = _B()
    adv = b.n(Advocacy, AFF, "1AC"); lk = b.n(Link, AFF, "1AC"); im = b.n(Impact, AFF, "1AC")
    bd = b.n(BallotDirective, AFF, "2AR")
    nonuniq = b.n(Uniqueness, NEG, "1NC", label="inevitable regardless of your link")
    els = [adv, lk, im, bd, nonuniq,
           b.sup(adv, lk), b.sup(lk, im), b.sup(im, bd),
           b.datk(nonuniq, lk)]                                 # non-unique defensively kills the link
    ballot, trace = judge(Round(elements=els, version=2))
    assert ballot == NEG
    assert _ballot(trace).reason_class == "AFF structural failure"
    assert not any(r.kind == "POLARITY_FLIP" and r.link_id == lk.id for r in trace)
    assert any(r.kind == "MAGNITUDE" and r.node_id == lk.id for r in trace)
    ch = _chains(trace)[0]
    assert ch.sign == 1 and ch.mag < EPSILON                    # dead (sign +1, mag 0), not turned


def test_r25_offense_at_uniqueness_inert_aff():
    """§11.25: an OffensiveAttack targeting a Uniqueness (either drawn direction).
    Turn-eligibility (§3.4): a uniqueness bears no offense, so the edge is inert --
    INERT_ATTACK, σ unchanged, no POLARITY_FLIP, the uniqueness is not turned.
    Confirms uniqueness is not turn-eligible. AFF (AFF offense)."""
    els = list(_load("r25.json").elements)
    ballot, trace = judge(Round(elements=els, version=2))
    assert ballot == AFF
    assert _ballot(trace).reason_class == "AFF offense"
    assert any(r.kind == "INERT_ATTACK" and "3.4" in r.reason for r in trace)
    assert not any(r.kind == "POLARITY_FLIP" for r in trace)
    assert _run_ctx(els).sigma["uni"] >= 0.5                   # σ untouched by the inert offense


# --- §11.26-28 Framework anchoring: impact-rooted, Advocacy/BD absorbing (§5.3) -

def _gate_for_impact(trace, impact_id):
    """The FRAMEWORK_GATE record for the chain whose terminal impact is impact_id."""
    return next(g for g in trace if g.kind == "FRAMEWORK_GATE" and g.impact_id == impact_id)


def test_r26_cross_side_direct_anchor_in_scope():
    """§11.26 / §5.3: a NEG impact that supports DIRECTLY into the AFF framework
    anchors to it -- "I win even under their framework". F_aff governs by being the
    sole survivor (NEG reads no framework of its own, so there is no framework
    debate). Cross-side direct anchoring is NOT side-scoped: the discriminator is
    the PATH (a direct impact->framework edge), not the side.

    This asserts on the ANCHOR, not the ballot: the round nets N = 0 (one shared
    framework, both chains in scope, +1 each) and that tie is incidental to what
    this round tests. The guard is that the impact-rooted walk still REACHES a
    framework across sides by a direct edge -- the §5.3 fix must not over-correct
    and sever legitimate cross-side direct anchoring."""
    _, trace = judge(_load("r26.json"))
    assert _select(trace).winning_framework_id == "faff" and _select(trace).via == "sole_survivor"
    neg_gate = _gate_for_impact(trace, "im_n")
    assert "faff" in neg_gate.anchors          # F_aff in anchors(im_n): cross-side direct anchor
    assert neg_gate.in_scope is True            # NEG chain in scope under the affirmative's framework


def test_r27_advocacy_fusion_does_not_anchor_neg():
    """§5.3 / AFFWINBYFW's isolated twin: a NEG disad whose uniqueness links off the
    SHARED advocacy (adv -> neg_uq) must NOT thereby anchor to the AFF framework.
    The disad's IMPACT was never supported into F_aff; only its premise reaches the
    advocacy, and the advocacy is absorbing (§5.3) -- the walk arrives there and
    halts, never crossing sideways into the AFF spine. NEG weighs F_neg > F_aff
    (determinate), so F_aff is defeated, F_neg governs, and only the NEG chain
    (anchored to F_neg) is in scope: (NEG, "NEG offense"), N < -eps.

    The disad's Support edge off the advocacy STAYS -- the §5.3 walk removes only
    its spurious anchoring, not the edge. Before the fix (whole-chain undirected
    walk) BOTH chains anchored {F_aff, F_neg} via the fusion and the round washed
    to presumption; this is the sentinel that flips red->green on the fix."""
    ballot, trace = judge(_load("r27.json"))
    assert ballot == NEG
    bl = _ballot(trace)
    assert bl.reason_class == "NEG offense" and bl.N < -EPSILON
    # the fusion no longer anchors: each impact reaches only its OWN framework
    assert "fneg" not in _gate_for_impact(trace, "im_a").anchors   # F_neg NOT in anchors(im_a)
    assert "faff" not in _gate_for_impact(trace, "im_n").anchors   # F_aff NOT in anchors(im_n)


def test_r28_aff_framework_win_aff():
    """§5.3 / AFF-mirror of fw_weigh_lockout: AFF and NEG in SEPARATE Support
    components (no cross-side edge), AFF weighs {F_aff, F_neg}. The weigh prefers
    F_aff by the own-side rule (an AFF weigh's preferred member is its own-side
    framework), so F_neg is defeated, F_aff governs, the NEG chain (anchored only
    to F_neg) is out of scope: (AFF, "AFF offense"), N = +1. This is the coverage
    the framework suite lacked -- AFF winning its own framework debate by weigh."""
    ballot, trace = judge(_load("r28.json"))
    assert ballot == AFF
    bl = _ballot(trace)
    assert bl.reason_class == "AFF offense" and bl.N > EPSILON
    sel = _select(trace)
    assert sel.via == "weigh" and sel.winning_framework_id == "faff" and "fneg" in sel.defeated
    assert _gate_for_impact(trace, "im_n").in_scope is False        # NEG chain out of scope


def test_r33_nonunique_on_live_link_poisons_shared_impact_neg():
    """§12.4.3 (v9, repurposed from the r30/delink probe): clean path A (u2->n2->n3)
    and redundant path B (u7->n7->n3), BOTH live and fully extended. A conceded NEG
    non-unique n8 zeroes Link B's uniqueness u7. A non-unique is STATE-level: because
    Link B is LIVE, it poisons the shared post-world state at impact n3 across ALL
    paths -- the clean sibling A does NOT rescue it (contrast r29's per-path delink).
    The complete, extended, sign-+1 AFF chain is driven to mag 0 -> AFF structural
    failure (§7). Contrast r31, where AFF concedes a delink on B and kicks out."""
    ballot, trace = judge(_load("r33.json"))
    assert ballot == NEG
    assert _ballot(trace).reason_class == "AFF structural failure"
    chs = _chains(trace)
    assert len(chs) == 1                                       # one component, one chain object
    assert chs[0].sign == 1 and chs[0].mag < EPSILON          # non-unique kill, not a flip
    ctx = _run_ctx(list(_load("r33.json").elements))
    assert ctx.sigma["u7"] < EPSILON                          # Link B's uniqueness dead
    assert ctx.sigma["n2"] >= 0.5 and ctx.sigma["n7"] >= 0.5  # both links individually intact


def test_r34_turned_redundant_link_washes_shared_impact_neg():
    """§ aggregation (convergence sign-conflict, EQUAL magnitude only): r34 (renamed
    from the r32.json probe). Identical to r33 EXCEPT e7 is an OffensiveAttackEdge --
    n8 TURNS n7 (Link<->Link, turn-eligible). The turned path shares its impact n3
    with the clean AFF path n4->n1->n2->n3. Ruling 4: a LIVE turned path sharing an
    impact with a clean AFF path WASHES that impact's AFF offense to 0,
    unconditionally, even though the clean sibling is fully extended. Equal-magnitude
    only (both paths mag 1.0; binary-wash and signed-net both yield 0) -- NOT
    generalized to partial turns.

    The turn scores for NEG only through a NEG BD anchored to it (ruling 2 / r10);
    there is none, so the turn banks 0, N washes to 0, and NEG wins by PRESUMPTION --
    NOT structural failure, since chain A is intact.

    EXPECTED TO FAIL mid-STEP-4: once per-path independence (4a) lands but before the
    convergence rule (4c), the clean path scores alone -> (AFF, "AFF offense"), +1.
    The convergence wash (4c) returns it to (NEG, "presumption"), N=0. (The CURRENT
    flattened engine reaches this verdict by accident -- n7's turn flips the whole
    fused chain to sign -1, driving aff_established false -- but for the wrong reason;
    the fix must reach "presumption" via the shared-impact wash. STEP-0 finding: the
    wash MUST leave no resolved sign-+1 extended AFF chain at n3, else _reason_class
    emits "AFF structural failure" as r33 does, not "presumption".)"""
    ballot, trace = judge(_load("r34.json"))
    assert ballot == NEG
    bl = _ballot(trace)
    assert bl.reason_class == "presumption" and abs(bl.N) <= EPSILON
    # the wash renders the shared impact NON-resolved-+1 (§3.3.1c / STEP-0 constraint):
    # sign '?', so aff_established is false and the label is "presumption", not
    # "AFF structural failure". This is what distinguishes the wash from the old
    # flatten accident (which flipped the whole fused chain to sign -1).
    chs = _chains(trace)
    assert len(chs) == 1 and chs[0].sign == "?"


def test_r29_shared_impact_one_path_defensive_killed_aff():
    """§3.3.1(a): two AFF chains to one shared impact n3, sharing advocacy n4.
    Chain A (n4-n8-n2-n3) is clean and fully extended. Chain B (n4-n9-n7-n3): its
    uniqueness n9 is clean/extended, its link n7 is fully extended BUT under the
    live unanswered de-link n11 -> sigma(n7)=0. Path B is dead at the LINK only (no
    non-unique). Per-path liveness: the impact survives because >=1 complete path
    (A) is fully extended and carries non-zero magnitude; the dead sibling path
    removes only itself. Survivor A carries: (AFF, "AFF offense"), N = +1.

    EXPECTED TO FAIL under the current flattened engine (the product over the union
    spine includes sigma(n7)=0 -> mag 0 -> NEG "AFF structural failure", the r33
    signature). Per-path aggregation (§3.3.1a) turns it green."""
    ballot, trace = judge(_load("r29.json"))
    assert ballot == AFF
    bl = _ballot(trace)
    assert bl.reason_class == "AFF offense" and abs(bl.N - 1.0) <= EPSILON


def test_r31_shared_impact_one_path_fully_kicked_aff():
    """§3.3.1(a): as r29 but chain B is kicked WHOLE -- both its link n7 and its
    uniqueness n9 are dropped (only-1AC) under live extended NEG pressure (de-link
    n11 -> n7, non-unique n10 -> n9). Path B fails extension outright (not merely
    zeroed). AFF kicked the dead chain; the clean sibling path A carries the shared
    impact: (AFF, "AFF offense"), N = +1.

    EXPECTED TO FAIL under the current flattened engine (same collapse as r29/r33).
    Per-path liveness (impact survives on >=1 complete extended path) turns it
    green -- a per-path extension failure on B does not poison A."""
    ballot, trace = judge(_load("r31.json"))
    assert ballot == AFF
    bl = _ballot(trace)
    assert bl.reason_class == "AFF offense" and abs(bl.N - 1.0) <= EPSILON


def test_r35_redundant_links_no_double_count_aff():
    """§3.3.1(b): the double-count guard. One impact n3, two clean fully-extended
    redundant links (n4-n1-n3 and n4-n2-n3), no attacks, one AFF BD. Same-sign
    redundancy aggregates by MAX over surviving paths, emitted as ONE chain object
    per impact-component -- the ballot must see the impact ONCE. Correct: N = +1.0
    (AFF, "AFF offense"). A per-path-object implementation would double-count to
    N = +2.0; asserting a single CHAIN record is the standing guard against that."""
    ballot, trace = judge(_load("r35.json"))
    assert ballot == AFF
    bl = _ballot(trace)
    assert bl.reason_class == "AFF offense" and abs(bl.N - 1.0) <= EPSILON
    assert len(_chains(trace)) == 1        # one chain per impact, not one per path (no double-count)


# --- Owner-side win condition (§7): AFF wins on captured turn offense ------------
# A turn hands the captured impact to the side its composed sign FAVORS (`owner`),
# which for a turned NEG disad is AFF. The win-gates read owner-side offense, not
# only AFF-introduced offense, so AFF can win on a disad it captured -- exactly as
# NEG wins on a captured AFF chain (r4). The N>eps floor and presumption asymmetry
# are the guards; T2'/T3/T4 pin them.

def test_T1_aff_wins_on_captured_disad_turn():
    """§7 owner-side + anchor-membership: AFF keeps the plan (advocacy n4 fully
    extended), drops its own advantage (n2/n3/n8 after 2AC), link-turns the NEG disad
    (n16 OffensiveAttacks n10), weighs the turn over the disad link (n17), and anchors
    the AFF BD to the TURNING LINK n16 (the natural authoring: n18 -> n16). n16 joins
    the captured chain only by its OffensiveAttack, so it is not a union-find member
    -- anchor_members (gated on the live capture eff_pol[n10] == -1) records it, and
    BD incidence reads that set. The captured chain is side NEG, owner AFF, in scope
    of the sole framework n12, N = +1: (AFF, "AFF offense"), N = +1.0.

    advocacy_present is satisfied by n4 reachable over Support from the captured
    chain (via the fusion edge e9: n4 -> n9), NOT by bare presence."""
    ballot, trace = judge(_load("AFFLinkturnsNeg.json"))
    assert ballot == AFF
    bl = _ballot(trace)
    assert bl.reason_class == "AFF offense" and abs(bl.N - 1.0) <= EPSILON


def test_T1b_aff_wins_with_bd_on_captured_impact():
    """§7 anchor-membership (both targets legal): identical to T1 but the AFF BD
    anchors the CAPTURED IMPACT n11 (n18 -> n11) instead of the turning link. Ruling 2
    is withdrawn -- a BD may anchor ANY node on the chain it directs the ballot toward,
    so both the turning link (T1) and the captured impact (here) are valid anchors and
    must resolve identically: (AFF, "AFF offense"), N = +1.0. Locks that neither anchor
    target regresses."""
    ballot, trace = judge(_load("AFFLinkturnsNeg_impactanchor.json"))
    assert ballot == AFF
    bl = _ballot(trace)
    assert bl.reason_class == "AFF offense" and abs(bl.N - 1.0) <= EPSILON


def test_T2prime_outweighed_turn_reverts_to_neg_offense():
    """§7 floor (turn defeated): as T1 but NEG wins a determinate weigh preferring
    the disad LINK n10 over the turn n16 (n17 is a NEG weigh). The turn is defeated
    -- n10 keeps its +1 polarity via preference -- so the disad reverts to NEG
    offense (side NEG, owner NEG, sign +1), the AFF turn captures nothing, and the
    NEG BD (n19 -> n11) scores it: N = -1, (NEG, "NEG offense").

    This is the guard that a turn which LOSES its polarity clash never enters
    owner_valid (favored side is NEG, not AFF). (The engine cannot render a turn's
    SIGN unresolved -- eff_pol is always +/-1; the sign==UNRESOLVED half of the
    _favored_side guard is exercised by r34's convergence wash instead.)"""
    ballot, trace = judge(_load("AFFturnOutweighed.json"))
    assert ballot == NEG
    bl = _ballot(trace)
    assert bl.reason_class == "NEG offense" and bl.N < -EPSILON


def test_T3_neg_mirror_captures_aff_impact():
    """§7 asymmetry (presumption): the mirror of T1 -- NEG captures an AFF impact by
    turning the AFF link (turn -> n lk_a), anchors a NEG BD to the captured impact,
    and weighs the turn. NEG wins by turned offense: (NEG, "NEG offense"), N = -1.
    NEG needs no owner-side gate of its own -- presumption is the tabula-rasa default
    and AFF carries the burden, so NEG wins by the ABSENCE of an AFF win. The
    owner-side change (AFF-only) leaves this untouched: whose offense counts is
    symmetric, who bears the burden is not."""
    ballot, trace = judge(_load("T3.json"))
    assert ballot == NEG
    bl = _ballot(trace)
    assert bl.reason_class == "NEG offense" and bl.N < -EPSILON


def test_T4_captured_turn_washed_to_zero_floor():
    """§7 N>eps floor: AFF captures the NEG disad (owner AFF, +1) but a clean,
    independent second NEG disad nets -1, so N washes to 0. The owner-side chain
    satisfies advocacy + complete + in-scope, yet the FLOOR (N > eps) fails, so NEG
    wins by presumption: (NEG, "presumption"), N = 0. This is the guard that the
    owner-side change does NOT become 'any captured turn wins' -- net offense is
    still required. Frameworkless so the two disads do not fuse through a shared
    framework; the fusion edge (adv -> uq_n) keeps the AFF advocacy reachable."""
    b = _B()
    adv = b.n(Advocacy, AFF, "1AC")
    uq_a = b.n(Uniqueness, AFF, "1AC", {"1AC": CONCEDED, "2AC": CONCEDED})
    lk_a = b.n(Link, AFF, "1AC", {"1AC": CONCEDED, "2AC": CONCEDED})
    im_a = b.n(Impact, AFF, "1AC", {"1AC": CONCEDED, "2AC": CONCEDED})
    uq_n = b.n(Uniqueness, NEG, "1NC"); lk_n = b.n(Link, NEG, "1NC")
    im_n = b.n(Impact, NEG, "1NC"); nbd = b.n(BallotDirective, NEG, "1NC")
    turn = b.n(Link, AFF, "2AC"); affbd = b.n(BallotDirective, AFF, "2AC")
    w = b.n(Weighing, AFF, "2AC")
    uq2 = b.n(Uniqueness, NEG, "1NC"); lk2 = b.n(Link, NEG, "1NC")
    im2 = b.n(Impact, NEG, "1NC"); nbd2 = b.n(BallotDirective, NEG, "1NC")
    ballot, trace = judge(Round(elements=[
        adv, uq_a, lk_a, im_a, uq_n, lk_n, im_n, nbd, turn, affbd, w, uq2, lk2, im2, nbd2,
        b.sup(adv, uq_a), b.sup(uq_a, lk_a), b.sup(lk_a, im_a),
        b.sup(adv, uq_n),                            # fusion: keeps the advocacy reachable
        b.sup(uq_n, lk_n), b.sup(lk_n, im_n), b.sup(nbd, im_n),
        b.oatk(turn, lk_n), b.cmp(w, turn), b.cmp(w, lk_n), b.sup(affbd, im_n),
        b.sup(uq2, lk2), b.sup(lk2, im2), b.sup(nbd2, im2)], version=2))  # clean 2nd NEG disad
    assert ballot == NEG
    bl = _ballot(trace)
    assert bl.reason_class == "presumption" and abs(bl.N) <= EPSILON


# --- Gap-audit CLOSEs (STEP 4): corners a self-play agent reaches that no prior
# --- round pinned. Each asserts the ruled verdict for a previously-unaudited path.

def test_C_impact_pair_weigh_drops_dispreferred_chain():
    """§6.5 SCOPE CLASH at the ballot (gap C). Two independently-BD-validated impact
    chains -- an AFF advantage and a NEG disad -- both reach the ballot; an AFF
    Weighing ranks the two TERMINAL IMPACTS {im_aff, im_neg}, determinate for im_aff
    (own-side preference). §6.5 (spec 'scope clashes exclude'): the DISPREFERRED
    impact's whole chain is EXCLUDED from the tally -- dropped entirely, no residual
    -- so the NEG chain contributes nothing and N reflects only the survivor.

    Without the weigh N = +1 - +1 = 0 -> NEG (presumption); the determinate impact-
    pair weigh drops the NEG chain, so N = +1 -> AFF. This is the ONLY path through
    judge._weighing_excluded, exercised by no prior round (every other weigh ranks
    links / uniquenesses / frameworks, never a terminal-impact pair at the ballot)."""
    b = _B()
    adv_a = b.n(Advocacy, AFF, "1AC"); uni_a = b.n(Uniqueness, AFF, "1AC")
    lk_a = b.n(Link, AFF, "1AC"); im_a = b.n(Impact, AFF, "1AC"); bd_a = b.n(BallotDirective, AFF, "2AR")
    uni_n = b.n(Uniqueness, NEG, "1NC"); lk_n = b.n(Link, NEG, "1NC")
    im_n = b.n(Impact, NEG, "1NC"); bd_n = b.n(BallotDirective, NEG, "2NR")
    w = b.n(Weighing, AFF, "2AC", label="AFF: our impact outweighs theirs")
    ballot, trace = judge(Round(elements=[
        adv_a, uni_a, lk_a, im_a, bd_a, uni_n, lk_n, im_n, bd_n, w,
        b.sup(adv_a, uni_a), b.sup(uni_a, lk_a), b.sup(lk_a, im_a), b.sup(im_a, bd_a),
        b.sup(uni_n, lk_n), b.sup(lk_n, im_n), b.sup(im_n, bd_n),
        b.cmp(w, im_a), b.cmp(w, im_n)], version=2))          # weigh ranks the two IMPACTS
    assert ballot == AFF
    bl = _ballot(trace)
    assert bl.reason_class == "AFF offense" and abs(bl.N - 1.0) <= EPSILON
    dec = {d["side"]: d for d in bl.decomposition}
    assert dec[AFF]["contributed"] is True                    # survivor scores
    assert dec[NEG]["contributed"] is False                   # dispreferred chain DROPPED entirely
    won = [r for r in trace if r.kind == "WEIGH" and r.outcome == "resolved"]
    assert won and won[0].preferred_node == im_a.id and set(won[0].pair) == {im_a.id, im_n.id}


def test_D_r32_nonunique_on_convergence_impact_kills_all_paths_neg():
    """§12.4.3 (v9, redrawn from the shared cut-vertex): two clean Links (n2, n7)
    converge on one shared Impact n3, each with its own satellite uniqueness; a live
    conceded NEG non-unique n8 zeroes the IMPACT's uniqueness u3. A non-unique on the
    convergence impact's own uniqueness is UNCONDITIONAL (there is no link to sever,
    so no kick-out): the shared state is non-unique, so both paths collapse. Neither
    link was individually touched (sigma(n2)=sigma(n7)=1.0), yet the impact dies.
    -> (NEG, 'AFF structural failure'), sign +1 (a non-unique kill, NOT a turn), mag 0."""
    ballot, trace = judge(_load("r32.json"))
    assert ballot == NEG
    assert _ballot(trace).reason_class == "AFF structural failure"
    chs = _chains(trace)
    assert len(chs) == 1                                       # one component, one chain object
    assert chs[0].sign == 1 and chs[0].mag < EPSILON          # non-unique kill, not a flip
    ctx = _run_ctx(list(_load("r32.json").elements))
    assert ctx.sigma["u3"] < EPSILON                          # impact uniqueness dead
    assert ctx.sigma["n2"] >= 0.5 and ctx.sigma["n7"] >= 0.5  # both links individually intact


def test_E_rebuttal_introduced_chain_not_extended_neg():
    """§6 'no new chains in rebuttals' chain-level guard (gap E). A whole NEG disad
    is introduced for the first time in the 2NR (a rebuttal speech), fully conceded.
    The component's earliest introduction is in REBUTTAL_SPEECHES, so it is forced
    extended=False and establishes no offense -- even though the 2NR impact has a
    later opposing speech (2AR) and would otherwise resolve. Without the guard the
    conceded disad nets N < 0 (NEG offense); the guard zeroes it -> N = 0 -> NEG by
    PRESUMPTION. Exercises the intro_speech-in-rebuttals branch (missing_speech ==
    the rebuttal itself)."""
    ballot, trace = judge(_load("E.json"))
    assert ballot == NEG
    bl = _ballot(trace)
    assert bl.reason_class == "presumption" and abs(bl.N) <= EPSILON
    ch = _chains(trace)[0]
    assert not ch.extended and not any(d["contributed"] for d in bl.decomposition)
    ef = [r for r in trace if r.kind == "EXTENSION_FAIL"]
    assert ef and ef[0].missing_speech == "2NR"               # the rebuttal-intro branch


def test_F_no_window_continuation_resolves_answered_aff():
    """§4 final-speech CONTINUATION refinement (gap F). AFF's terminal impact is
    introduced fresh in the 2AR (final speech, no response window), but it CONTINUES
    a clash that was CONTESTED entering the prior opposing speech (2NR): the link it
    attaches to carries a '2NR: contested' stamp. Unlike a fresh 2AR spike (r8, which
    goes UNRESOLVED), a legitimate continuation resolves 'answered' -- neither
    dropped nor unresolved -- so the chain stands and AFF wins on offense. This is
    the continues==True branch that r8 (continues==False) does not reach."""
    ballot, trace = judge(_load("F.json"))
    assert ballot == AFF
    assert _ballot(trace).reason_class == "AFF offense"
    assert not any(r.kind == "UNRESOLVED" and r.node_id == "im" for r in trace)  # not inert
    assert not any(r.kind == "DROP" and r.node_id == "im" for r in trace)        # not dropped
    ch = _chains(trace)[0]
    assert ch.sign == 1 and ch.extended


def test_H_severed_advocacy_fails_gate_neg():
    """§7 advocacy gate false branch (gap H). AFF builds a COMPLETE, IN-SCOPE,
    positive-N owner-side advantage (uniqueness->link->impact->BD, all conceded,
    N = +1) -- but the plan Advocacy is SEVERED: it sits in a disjoint Support
    component (linked only to a stray NEG BD), so it is NOT reachable over Support
    from the balloted offense. AFF must tie the plan to its offense; with no
    reachable advocacy the gate denies the win. Every OTHER gate passes -- the
    ballot's gates_passed carries complete_chain, in_scope_impact and N>eps but NOT
    'advocacy' -- so AFF loses to NEG on the advocacy gate alone. No prior round
    exercises this false branch (advocacy is reachable in every other AFF win)."""
    b = _B()
    uni_a = b.n(Uniqueness, AFF, "1AC"); lk_a = b.n(Link, AFF, "1AC")
    im_a = b.n(Impact, AFF, "1AC"); bd_a = b.n(BallotDirective, AFF, "2AR")
    adv = b.n(Advocacy, AFF, "1AC"); bd_n = b.n(BallotDirective, NEG, "2NR")
    ballot, trace = judge(Round(elements=[
        uni_a, lk_a, im_a, bd_a, adv, bd_n,
        b.sup(uni_a, lk_a), b.sup(lk_a, im_a), b.sup(im_a, bd_a),
        b.sup(adv, bd_n)], version=2))               # advocacy severed onto a disjoint component
    assert ballot == NEG
    bl = _ballot(trace)
    assert bl.reason_class == "AFF structural failure" and bl.N > EPSILON
    assert "advocacy" not in bl.gates_passed                  # the gate that failed
    assert {"complete_chain", "in_scope_impact", "N>eps"} <= set(bl.gates_passed)  # all others passed


def test_G1_backwards_drawn_attack_still_applies_neg():
    """§2.2 direction-agnostic attack (gap G, sentinel 1). r2's conceded defensive
    kill, but the DefensiveAttack edge is drawn BACKWARDS -- source = the earlier
    link (1AC), target = the later non-unique (1NC). Attacks are oriented by SPEECH
    RECENCY, not draw direction: the later-speech node is still the attacker, so the
    edge applies identically and kills the link -> NEG. It is NOT inert (the reversed
    draw is coherent); this pins the ib>ia swap branch."""
    ballot, trace = judge(_load("G1.json"))
    assert ballot == NEG
    ch = _chains(trace)[0]
    assert ch.mag < EPSILON                                   # applied despite the reversed draw
    assert not any(r.kind == "INERT_ATTACK" for r in trace)  # coherent, not inert


def test_G2_same_speech_clash_inert():
    """§2.2 (gap G, sentinel 2). Two nodes in the SAME speech joined by an attack:
    with no speech-recency ordering there is no attacker/target, so the edge is inert
    ('same-speech clash'). Attached beside a clean AFF advantage; the inert edge does
    nothing -> AFF, and the INERT_ATTACK is recorded."""
    ballot, trace = judge(_load("G2.json"))
    assert ballot == AFF
    assert any(r.kind == "INERT_ATTACK" and "same-speech" in r.reason for r in trace)


def test_G3_same_side_attack_inert():
    """§2.2 (gap G, sentinel 3). An attack between two SAME-SIDE nodes (AFF on AFF,
    different speeches) is incoherent -- a side does not attack itself -- so it is
    inert and changes no magnitude. The clean AFF advantage stands -> AFF, mag 1.0,
    with the INERT_ATTACK recorded."""
    ballot, trace = judge(_load("G3.json"))
    assert ballot == AFF
    ch = _chains(trace)[0]
    assert abs(ch.mag - 1.0) < 1e-9                           # inert -> magnitude untouched
    assert any(r.kind == "INERT_ATTACK" and "same-side" in r.reason for r in trace)
