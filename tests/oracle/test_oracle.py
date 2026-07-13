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
    b = _B()
    adv = b.n(Advocacy, AFF, "1AC"); uni = b.n(Uniqueness, AFF, "1AC")
    lk = b.n(Link, AFF, "1AC"); im = b.n(Impact, AFF, "1AC")
    bd = b.n(BallotDirective, AFF, "2AR")
    ballot, trace = judge(Round(elements=[
        adv, uni, lk, im, bd,
        b.sup(adv, uni), b.sup(uni, lk), b.sup(lk, im), b.sup(im, bd)], version=2))
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
    b = _B()
    adv = b.n(Advocacy, AFF, "1AC"); uni = b.n(Uniqueness, AFF, "1AC")
    lk = b.n(Link, AFF, "1AC"); im = b.n(Impact, AFF, "1AC")
    bd = b.n(BallotDirective, AFF, "2AR")
    d = b.n(Link, NEG, "1NC")                       # conceded defensive attack, NEG-extended
    ballot, trace = judge(Round(elements=[
        adv, uni, lk, im, bd, d,
        b.sup(adv, uni), b.sup(uni, lk), b.sup(lk, im), b.sup(im, bd),
        b.datk(d, lk)], version=2))
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
    b = _B()
    adv = b.n(Advocacy, AFF, "1AC"); uni = b.n(Uniqueness, AFF, "1AC")
    lk = b.n(Link, AFF, "1AC"); im = b.n(Impact, AFF, "1AC")
    bd = b.n(BallotDirective, AFF, "2AR")
    d = b.n(Link, NEG, "1NC")                       # NEG defense on the link
    c = b.n(Link, AFF, "2AC")                       # AFF answers the defense (conceded)
    ballot, trace = judge(Round(elements=[
        adv, uni, lk, im, bd, d, c,
        b.sup(adv, uni), b.sup(uni, lk), b.sup(lk, im), b.sup(im, bd),
        b.datk(d, lk), b.datk(c, d)], version=2))
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
    b = _B()
    adv = b.n(Advocacy, AFF, "1AC"); uni = b.n(Uniqueness, AFF, "1AC")
    lk = b.n(Link, AFF, "1AC", {"1AC": CONTESTED, "1NC": CONTESTED,
                                "2NC/1NR": CONTESTED, "2NR": CONTESTED})
    im = b.n(Impact, AFF, "1AC", {"1AC": CONCEDED, "1NC": CONCEDED,
                                  "2NC/1NR": CONCEDED, "2NR": CONCEDED})
    turn = b.n(Link, NEG, "1NC")                    # the offensive turn
    bd = b.n(BallotDirective, NEG, "2NR")
    ballot, trace = judge(Round(elements=[
        adv, uni, lk, im, turn, bd,
        b.sup(adv, uni), b.sup(uni, lk), b.sup(lk, im), b.sup(im, bd),
        b.oatk(turn, lk)], version=2))
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
    b = _B()
    adv = b.n(Advocacy, AFF, "1AC"); uni = b.n(Uniqueness, AFF, "1AC")
    lk = b.n(Link, AFF, "1AC", {"1AC": CONTESTED, "1NC": CONTESTED,
                                "2NC/1NR": CONTESTED, "2NR": CONTESTED})
    im = b.n(Impact, AFF, "1AC", {"1AC": CONCEDED})     # dead: no one carried it forward
    turn = b.n(Link, NEG, "1NC")
    bd = b.n(BallotDirective, NEG, "2NR")
    ballot, trace = judge(Round(elements=[
        adv, uni, lk, im, turn, bd,
        b.sup(adv, uni), b.sup(uni, lk), b.sup(lk, im), b.sup(im, bd),
        b.oatk(turn, lk)], version=2))
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
    b = _B()
    adv = b.n(Advocacy, AFF, "1AC"); uni = b.n(Uniqueness, AFF, "1AC")
    # AFF re-extends the turned link + impact through its speeches (§6); NEG's
    # 1NC/2NR contested stamps and AFF's 2AC/1AR/2AR conceded stamps coexist on
    # the same side-agnostic union record.
    live = {"1AC": CONTESTED, "1NC": CONTESTED, "2AC": CONCEDED,
            "1AR": CONCEDED, "2NR": CONTESTED, "2AR": CONCEDED}
    lk = b.n(Link, AFF, "1AC", dict(live)); im = b.n(Impact, AFF, "1AC", dict(live))
    turnL = b.n(Link, NEG, "1NC", label="link turn")
    turnI = b.n(Link, NEG, "1NC", label="impact turn")
    bd = b.n(BallotDirective, AFF, "2AR")
    ballot, trace = judge(Round(elements=[
        adv, uni, lk, im, turnL, turnI, bd,
        b.sup(adv, uni), b.sup(uni, lk), b.sup(lk, im), b.sup(im, bd),
        b.oatk(turnL, lk), b.oatk(turnI, im)], version=2))
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
    b = _B()
    aff_live = {"1AC": CONCEDED, "2AC": CONCEDED, "1AR": CONCEDED, "2AR": CONCEDED}
    adv = b.n(Advocacy, AFF, "1AC", dict(aff_live)); uni = b.n(Uniqueness, AFF, "1AC", dict(aff_live))
    lk = b.n(Link, AFF, "1AC", {"1AC": CONTESTED, "2AC": CONTESTED,
                                "1AR": CONTESTED, "2AR": CONTESTED})   # AFF keeps it live
    im = b.n(Impact, AFF, "1AC", dict(aff_live))                      # AFF keeps it live
    turn = b.n(Link, NEG, "1NC")                                     # NEG carries the turn
    bd = b.n(BallotDirective, NEG, "2NR")
    ballot, trace = judge(Round(elements=[
        adv, uni, lk, im, turn, bd,
        b.sup(adv, uni), b.sup(uni, lk), b.sup(lk, im), b.sup(im, bd),
        b.oatk(turn, lk)], version=2))
    assert ballot == NEG
    ch = _chains(trace)[0]
    assert ch.sign == -1 and ch.mag > EPSILON and ch.extended   # union kept the turned chain live
    assert _ballot(trace).reason_class == "NEG offense" and _ballot(trace).N < -EPSILON


# --- §11.5 Link turn, NOT extended -> NEG -------------------------------------

def test_r5_link_turn_not_extended_neg():
    """As (4) but NEG fails to carry the inherited impact in 2NR (its liveness
    stops at the block). The chain still fails §6. -> NEG."""
    b = _B()
    adv = b.n(Advocacy, AFF, "1AC"); uni = b.n(Uniqueness, AFF, "1AC")
    lk = b.n(Link, AFF, "1AC", {"1AC": CONTESTED, "1NC": CONTESTED,
                                "2NC/1NR": CONTESTED, "2NR": CONTESTED})
    im = b.n(Impact, AFF, "1AC", {"1AC": CONCEDED, "1NC": CONCEDED,
                                  "2NC/1NR": CONCEDED})     # gap at 2NR
    turn = b.n(Link, NEG, "1NC")
    bd = b.n(BallotDirective, NEG, "2NR")
    ballot, trace = judge(Round(elements=[
        adv, uni, lk, im, turn, bd,
        b.sup(adv, uni), b.sup(uni, lk), b.sup(lk, im), b.sup(im, bd),
        b.oatk(turn, lk)], version=2))
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
    b = _B()
    adv = b.n(Advocacy, AFF, "1AC"); uni = b.n(Uniqueness, AFF, "1AC")
    lk = b.n(Link, AFF, "1AC"); im = b.n(Impact, AFF, "1AC")
    bd = b.n(BallotDirective, AFF, "2AR")
    fw = b.n(Framework, NEG, "1NC")
    ballot, trace = judge(Round(elements=[
        adv, uni, lk, im, bd, fw,
        b.sup(adv, uni), b.sup(uni, lk), b.sup(lk, im), b.sup(im, bd),
        b.sup(fw, bd)], version=2))                 # fw reachable via BD, no path to the impact
    assert ballot == NEG
    assert _ballot(trace).reason_class == "framework lock-out"
    gates = {(r.impact_id, r.in_scope) for r in trace if r.kind == "FRAMEWORK_GATE"}
    assert (im.id, False) in gates                  # AFF impact locked out


# --- §11.8 No-window new 2AR offense -> inert ---------------------------------

def test_r8_no_window_new_2ar_offense_inert():
    """AFF's terminal impact is introduced fresh in the 2AR (final speech); NEG
    never had standing and the attachment was not contested entering the 2AR, so
    it is UNRESOLVED (inert) -- establishes no offense. -> NEG."""
    b = _B()
    adv = b.n(Advocacy, AFF, "1AC"); uni = b.n(Uniqueness, AFF, "1AC")
    lk = b.n(Link, AFF, "1AC")
    im = b.n(Impact, AFF, "2AR", {"2AR": CONCEDED})     # brand-new 2AR offense
    bd = b.n(BallotDirective, AFF, "2AR")
    ballot, trace = judge(Round(elements=[
        adv, uni, lk, im, bd,
        b.sup(adv, uni), b.sup(uni, lk), b.sup(lk, im), b.sup(im, bd)], version=2))
    assert ballot == NEG
    # §7 (redrawn): the 2AR impact is UNRESOLVED (no window) -- offense never
    # legitimately existed, no complete AFF chain was ever established -> presumption
    # (not "AFF structural failure", which requires offense built then lost).
    assert _ballot(trace).reason_class == "presumption"
    assert any(r.kind == "UNRESOLVED" and r.node_id == im.id for r in trace)


# --- Recursive weighing (§6.5) -- clash resolution decides polarity -----------

def _turned_link_round(b, counter_weigh=False, meta_weigh=False):
    """AFF advocacy->UQ->link->impact + BD (AFF wins alone); NEG turns the link
    (OffensiveAttack). AFF weighs {AFF link, the turn}. Optionally NEG counter-
    weighs the same pair oppositely, and optionally an AFF meta-weigh ranks the
    two weighs. Returns (elements, link_id)."""
    adv = b.n(Advocacy, AFF, "1AC"); uni = b.n(Uniqueness, AFF, "1AC")
    lk = b.n(Link, AFF, "1AC", label="AFF link"); im = b.n(Impact, AFF, "1AC")
    bd = b.n(BallotDirective, AFF, "2AR")
    turn = b.n(Link, NEG, "1NC", label="NEG turn")
    waff = b.n(Weighing, AFF, "2AC", label="AFF: our link beats the turn")
    els = [adv, uni, lk, im, bd, turn, waff,
           b.sup(adv, uni), b.sup(uni, lk), b.sup(lk, im), b.sup(im, bd),
           b.oatk(turn, lk),                         # NEG turns the AFF link
           b.cmp(waff, lk), b.cmp(waff, turn)]       # AFF weighs {link, turn}
    if counter_weigh:
        wneg = b.n(Weighing, NEG, "2NC/1NR", label="NEG: the turn beats their link")
        els += [wneg, b.cmp(wneg, lk), b.cmp(wneg, turn)]
        if meta_weigh:
            meta = b.n(Weighing, AFF, "1AR", label="AFF meta: our weigh controls")
            els += [meta, b.cmp(meta, waff), b.cmp(meta, wneg)]
    return els, lk.id


def test_r9_turned_link_saved_by_determinate_weigh_aff():
    """A determinate won link-weigh decides the polarity clash for AFF: the link
    keeps its polarity via PREFERENCE (not sigma), the defeated turn drops from
    the chain magnitude, the chain survives, and AFF takes the ballot."""
    b = _B()
    els, lk_id = _turned_link_round(b)               # AFF weighs; NEG does not counter
    ballot, trace = judge(Round(elements=els, version=2))
    assert ballot == AFF
    assert _ballot(trace).reason_class == "AFF offense"
    assert _polarity_via(trace, lk_id) == "preference"     # weigh decided, not the 0.5 sigma rule
    # the WEIGH record shows the link-weigh resolved in favour of the AFF link:
    weigh = [r for r in trace if r.kind == "WEIGH" and r.outcome == "resolved"]
    assert weigh and weigh[0].preferred_node == lk_id and weigh[0].via == "preference"
    ch = _chains(trace)[0]
    assert ch.sign == 1 and ch.mag > EPSILON               # link kept polarity, magnitude survived


def test_r10_turned_link_indeterminate_falls_to_magnitude_neg():
    """Same round but NEG counter-weighs the SAME pair oppositely with no meta-
    weigh: the weighing layer has no lone survivor (§6.5 case b), so the clash
    falls to DF-QuAD magnitude -- the turn resolves as it did before the upgrade
    (link flips via the sigma threshold) and NEG takes the ballot."""
    b = _B()
    els, lk_id = _turned_link_round(b, counter_weigh=True)
    ballot, trace = judge(Round(elements=els, version=2))
    assert ballot == NEG
    # r10 investigation (case a): the symmetric weigh falls to magnitude and the
    # link FLIPS to NEG (sign -1, mag 1.0). But r10 authors only an AFF BD, so the
    # NEG-favoring turned offense is orphaned (BD_VALIDATE fails "offense favors NEG,
    # BD claims AFF") -- NOT a bug (r4 shows a NEG BD would validate it). Nobody
    # established scoring offense -> presumption. A flip is never structural failure.
    assert _ballot(trace).reason_class == "presumption"
    assert _polarity_via(trace, lk_id) == "dfquad"         # fell back to the 0.5 sigma threshold
    assert all(r.outcome == "symmetric" for r in trace if r.kind == "WEIGH")


def test_r11_recursive_meta_weigh_breaks_tie_aff():
    """Depth-2: AFF and NEG weigh {link, turn} oppositely (would tie), but an AFF
    META-weigh ranks the two weighs and defeats NEG's. That leaves AFF's weigh the
    lone survivor at the base clash -> determinate -> the link is saved -> AFF.
    This is the case a fixed depth-2 check gets wrong; it must stay a recursion,
    so this is a permanent regression test that resolve is not depth-limited."""
    b = _B()
    els, lk_id = _turned_link_round(b, counter_weigh=True, meta_weigh=True)
    ballot, trace = judge(Round(elements=els, version=2))
    assert ballot == AFF
    assert _ballot(trace).reason_class == "AFF offense"
    assert _polarity_via(trace, lk_id) == "preference"     # meta broke the tie -> determinate


# --- Attacker-liveness gate (§3.1, §6 -- v4) ----------------------------------

def _uniqueness_attack_round(b, nonuniq_live, answer=False):
    """AFF advantage advocacy->uniqueness->link->impact + BD, all AFF-extended to
    2AR. NEG reads a non-unique (DefensiveAttack) on the uniqueness in 1NC with
    liveness `nonuniq_live`. If `answer`, AFF attacks the non-unique in 2AC (a
    conceded counter). Returns (elements, uniqueness_id, nonuniq_id)."""
    adv = b.n(Advocacy, AFF, "1AC"); uni = b.n(Uniqueness, AFF, "1AC")
    lk = b.n(Link, AFF, "1AC"); im = b.n(Impact, AFF, "1AC")
    bd = b.n(BallotDirective, AFF, "2AR")
    nonuniq = b.n(Uniqueness, NEG, "1NC", nonuniq_live, label="NEG non-unique")
    els = [adv, uni, lk, im, bd, nonuniq,
           b.sup(adv, uni), b.sup(uni, lk), b.sup(lk, im), b.sup(im, bd),
           b.datk(nonuniq, uni)]                      # NEG non-unique attacks the uniqueness
    if answer:
        counter = b.n(Uniqueness, AFF, "2AC", label="AFF answer to the non-unique")
        els += [counter, b.datk(counter, nonuniq)]    # AFF attacks the attacker (mitigation)
    return els, uni.id, nonuniq.id


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
    b = _B()
    els, uni_id, nonuniq_id = _uniqueness_attack_round(b, {"1NC": CONTESTED})
    assert nonuniq_id not in _uni_attackers(els, uni_id)   # lapsed: NOT in the attacker set
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
    b = _B()
    full_neg = {"1NC": CONTESTED, "2NC/1NR": CONCEDED, "2NR": CONCEDED}
    els, uni_id, nonuniq_id = _uniqueness_attack_round(b, full_neg)
    assert nonuniq_id in _uni_attackers(els, uni_id)       # live: DOES contest
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
    b = _B()
    full_neg = {"1NC": CONTESTED, "2NC/1NR": CONCEDED, "2NR": CONCEDED}
    els, uni_id, nonuniq_id = _uniqueness_attack_round(b, full_neg, answer=True)
    assert nonuniq_id in _uni_attackers(els, uni_id)       # live (answered, not lapsed)
    ballot, trace = judge(Round(elements=els, version=2))
    assert ballot == AFF                                    # answered down -> uniqueness restored
    assert _ballot(trace).reason_class == "AFF offense"
    ch = _chains(trace)[0]
    assert ch.mag > EPSILON


# --- Won weigh defeats the attacker across types (§6.5 -- v5) ------------------

def _uniqueness_weigh_round(b, weigh_side):
    """AFF advantage; NEG non-unique (extended, live) attacks the uniqueness;
    `weigh_side` weighs {uniqueness, non-unique}. Returns (els, uni_id, nonuniq_id).
    The weigh's preferred member is its own-side node -- AFF prefers the uniqueness,
    NEG prefers the non-unique (§6.5)."""
    adv = b.n(Advocacy, AFF, "1AC"); uni = b.n(Uniqueness, AFF, "1AC")
    lk = b.n(Link, AFF, "1AC"); im = b.n(Impact, AFF, "1AC")
    bd = b.n(BallotDirective, AFF, "2AR")
    nonuniq = b.n(Uniqueness, NEG, "1NC", label="NEG non-unique")   # full NEG liveness -> live
    speech = "2AC" if weigh_side == AFF else "2NC/1NR"
    w = b.n(Weighing, weigh_side, speech, label="weigh {uniqueness, non-unique}")
    els = [adv, uni, lk, im, bd, nonuniq, w,
           b.sup(adv, uni), b.sup(uni, lk), b.sup(lk, im), b.sup(im, bd),
           b.datk(nonuniq, uni),                        # non-unique attacks the uniqueness
           b.cmp(w, uni), b.cmp(w, nonuniq)]            # weigh ranks {uniqueness, non-unique}
    return els, uni.id, nonuniq.id


def test_r15_won_uniqueness_weigh_defeats_nonunique_aff():
    """THE BUG FIX (§6.5 -- v5). AFF WINS a determinate weigh over the uniqueness
    clash {n1, n7} (preferred = the AFF uniqueness). The defeated non-unique n7 does
    NOT attack the winner: it is removed from n1's DF-QuAD attacker set, n1 survives
    at sigma 1.0, the chain holds at mag 1.0, and AFF wins. Previously the weigh was
    decorative -- n7 still zeroed n1. Same rule as the link case (r9), now over
    uniquenesses."""
    b = _B()
    els, uni_id, nonuniq_id = _uniqueness_weigh_round(b, AFF)
    assert nonuniq_id not in _uni_attackers(els, uni_id)   # defeated -> NOT in the attacker set
    ballot, trace = judge(Round(elements=els, version=2))
    assert ballot == AFF
    assert _ballot(trace).reason_class == "AFF offense"
    assert any(r.kind == "INERT_ATTACK" and r.reason == "defeated by weigh" for r in trace)
    ch = _chains(trace)[0]
    assert ch.extended and abs(ch.mag - 1.0) < 1e-9        # uniqueness restored -> chain intact
    won = [r for r in trace if r.kind == "WEIGH" and r.outcome == "resolved"]
    assert won and won[0].preferred_node == uni_id         # the weigh resolved for the uniqueness


def test_r16_lost_uniqueness_weigh_nonunique_still_attacks_neg():
    """CONTRAST / fallback. NEG WINS the weigh (preferred = the non-unique), so the
    clash does NOT resolve for the uniqueness: n7 is not defeated, stays in n1's
    attacker set, and drives n1 to 0 -> chain mag 0 -> NEG. A weigh only removes the
    attacker when the winner is the TARGET; a lost/indeterminate weigh leaves the
    attack contesting (the no-weigh fallback is r13)."""
    b = _B()
    els, uni_id, nonuniq_id = _uniqueness_weigh_round(b, NEG)
    assert nonuniq_id in _uni_attackers(els, uni_id)       # not defeated -> still contests
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
    b = _B()
    adv = b.n(Advocacy, AFF, "1AC"); uni = b.n(Uniqueness, AFF, "1AC")
    lk = b.n(Link, AFF, "1AC"); im = b.n(Impact, AFF, "1AC"); bd = b.n(BallotDirective, AFF, "2AR")
    futil = b.n(Framework, AFF, "1AC", label="F_util")
    fsv = b.n(Framework, NEG, "1NC", label="F_sv")
    w = b.n(Weighing, NEG, "2NC/1NR", label="NEG: F_sv > F_util")
    ballot, trace = judge(Round(elements=[
        adv, uni, lk, im, bd, futil, fsv, w,
        b.sup(adv, uni), b.sup(uni, lk), b.sup(lk, im), b.sup(im, bd),
        b.sup(im, futil), b.sup(im, fsv),
        b.cmp(w, futil), b.cmp(w, fsv)], version=2))
    assert ballot == AFF
    bl = _ballot(trace)
    assert bl.reason_class == "AFF offense" and bl.N > EPSILON
    sel = _select(trace)
    assert sel.via == "weigh" and sel.winning_framework_id == fsv.id
    assert futil.id in sel.defeated


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
    b = _B()
    adv = b.n(Advocacy, AFF, "1AC"); uni = b.n(Uniqueness, AFF, "1AC")
    lk = b.n(Link, AFF, "1AC"); im = b.n(Impact, AFF, "1AC")
    fw = b.n(Framework, AFF, "1AC", label="F_aff"); bd = b.n(BallotDirective, AFF, "2AR")
    natk = b.n(Link, NEG, "1NC", label="offense aimed at the framework")
    els = [adv, uni, lk, im, fw, bd, natk,
           b.sup(adv, uni), b.sup(uni, lk), b.sup(lk, im), b.sup(im, fw), b.sup(fw, bd),
           b.oatk(natk, fw)]                                     # offense -> Framework: inert
    ballot, trace = judge(Round(elements=els, version=2))
    assert ballot == AFF
    assert _ballot(trace).reason_class == "AFF offense"
    assert any(r.kind == "INERT_ATTACK" and "3.4" in r.reason for r in trace)
    assert not any(r.kind == "POLARITY_FLIP" for r in trace)
    assert _run_ctx(els).sigma[fw.id] >= 0.5                    # σ untouched by the inert offense


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
    b = _B()
    adv = b.n(Advocacy, AFF, "1AC"); uni = b.n(Uniqueness, AFF, "1AC")
    lk = b.n(Link, AFF, "1AC"); im = b.n(Impact, AFF, "1AC"); bd = b.n(BallotDirective, AFF, "2AR")
    natk = b.n(Link, NEG, "1NC", label="offense aimed at the uniqueness")
    els = [adv, uni, lk, im, bd, natk,
           b.sup(adv, uni), b.sup(uni, lk), b.sup(lk, im), b.sup(im, bd),
           b.oatk(natk, uni)]                                   # offense -> Uniqueness: inert
    ballot, trace = judge(Round(elements=els, version=2))
    assert ballot == AFF
    assert _ballot(trace).reason_class == "AFF offense"
    assert any(r.kind == "INERT_ATTACK" and "3.4" in r.reason for r in trace)
    assert not any(r.kind == "POLARITY_FLIP" for r in trace)
    assert _run_ctx(els).sigma[uni.id] >= 0.5                   # σ untouched by the inert offense


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
    b = _B()
    adv = b.n(Advocacy, AFF, "1AC"); uq = b.n(Uniqueness, AFF, "1AC")
    lk = b.n(Link, AFF, "1AC"); im_a = b.n(Impact, AFF, "1AC")
    faff = b.n(Framework, AFF, "1AC", label="F_aff"); bda = b.n(BallotDirective, AFF, "2AR")
    nuq = b.n(Uniqueness, NEG, "1NC"); nlk = b.n(Link, NEG, "1NC"); im_n = b.n(Impact, NEG, "1NC")
    bdn = b.n(BallotDirective, NEG, "2NR")
    els = [adv, uq, lk, im_a, faff, bda, nuq, nlk, im_n, bdn,
           b.sup(adv, uq), b.sup(uq, lk), b.sup(lk, im_a), b.sup(im_a, faff), b.sup(faff, bda),
           b.sup(nuq, nlk), b.sup(nlk, im_n), b.sup(im_n, faff),   # cross-side DIRECT im_n -> F_aff
           b.sup(im_n, bdn)]
    _, trace = judge(Round(elements=els, version=2))
    assert _select(trace).winning_framework_id == faff.id and _select(trace).via == "sole_survivor"
    neg_gate = _gate_for_impact(trace, im_n.id)
    assert faff.id in neg_gate.anchors          # F_aff in anchors(im_n): cross-side direct anchor
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
    b = _B()
    adv = b.n(Advocacy, AFF, "1AC"); uq = b.n(Uniqueness, AFF, "1AC")
    lk = b.n(Link, AFF, "1AC"); im_a = b.n(Impact, AFF, "1AC")
    faff = b.n(Framework, AFF, "1AC", label="F_aff"); bda = b.n(BallotDirective, AFF, "2AR")
    nuq = b.n(Uniqueness, NEG, "1NC"); nlk = b.n(Link, NEG, "1NC"); im_n = b.n(Impact, NEG, "1NC")
    fneg = b.n(Framework, NEG, "1NC", label="F_neg"); bdn = b.n(BallotDirective, NEG, "2NR")
    w = b.n(Weighing, NEG, "2NC/1NR", label="F_neg > F_aff")
    els = [adv, uq, lk, im_a, faff, bda, nuq, nlk, im_n, fneg, bdn, w,
           b.sup(adv, uq), b.sup(uq, lk), b.sup(lk, im_a), b.sup(im_a, faff), b.sup(faff, bda),
           b.sup(adv, nuq),                                   # FUSION edge (AFFWINBYFW e9)
           b.sup(nuq, nlk), b.sup(nlk, im_n), b.sup(im_n, fneg), b.sup(fneg, bdn),
           b.cmp(w, faff), b.cmp(w, fneg)]
    ballot, trace = judge(Round(elements=els, version=2))
    assert ballot == NEG
    bl = _ballot(trace)
    assert bl.reason_class == "NEG offense" and bl.N < -EPSILON
    # the fusion no longer anchors: each impact reaches only its OWN framework
    assert fneg.id not in _gate_for_impact(trace, im_a.id).anchors   # F_neg NOT in anchors(im_a)
    assert faff.id not in _gate_for_impact(trace, im_n.id).anchors   # F_aff NOT in anchors(im_n)


def test_r28_aff_framework_win_aff():
    """§5.3 / AFF-mirror of fw_weigh_lockout: AFF and NEG in SEPARATE Support
    components (no cross-side edge), AFF weighs {F_aff, F_neg}. The weigh prefers
    F_aff by the own-side rule (an AFF weigh's preferred member is its own-side
    framework), so F_neg is defeated, F_aff governs, the NEG chain (anchored only
    to F_neg) is out of scope: (AFF, "AFF offense"), N = +1. This is the coverage
    the framework suite lacked -- AFF winning its own framework debate by weigh."""
    b = _B()
    adv = b.n(Advocacy, AFF, "1AC"); uq = b.n(Uniqueness, AFF, "1AC")
    lk = b.n(Link, AFF, "1AC"); im_a = b.n(Impact, AFF, "1AC")
    faff = b.n(Framework, AFF, "1AC", label="F_aff"); bda = b.n(BallotDirective, AFF, "2AR")
    nuq = b.n(Uniqueness, NEG, "1NC"); nlk = b.n(Link, NEG, "1NC"); im_n = b.n(Impact, NEG, "1NC")
    fneg = b.n(Framework, NEG, "1NC", label="F_neg"); bdn = b.n(BallotDirective, NEG, "2NR")
    w = b.n(Weighing, AFF, "2AC", label="F_aff > F_neg")
    els = [adv, uq, lk, im_a, faff, bda, nuq, nlk, im_n, fneg, bdn, w,
           b.sup(adv, uq), b.sup(uq, lk), b.sup(lk, im_a), b.sup(im_a, faff), b.sup(faff, bda),
           b.sup(nuq, nlk), b.sup(nlk, im_n), b.sup(im_n, fneg), b.sup(fneg, bdn),
           b.cmp(w, faff), b.cmp(w, fneg)]
    ballot, trace = judge(Round(elements=els, version=2))
    assert ballot == AFF
    bl = _ballot(trace)
    assert bl.reason_class == "AFF offense" and bl.N > EPSILON
    sel = _select(trace)
    assert sel.via == "weigh" and sel.winning_framework_id == faff.id and fneg.id in sel.defeated
    assert _gate_for_impact(trace, im_n.id).in_scope is False        # NEG chain out of scope
