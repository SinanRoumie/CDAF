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

from model import (
    Round, Advocacy, Uniqueness, Link, Impact, Framework, Weighing, BallotDirective,
    Support, DefensiveAttack, OffensiveAttack, Comparison, SPEECH_ORDER, SPEECH_SIDE,
)
from judge import judge
from judge.config import AFF, NEG, EPSILON

CONTESTED, CONCEDED = "contested", "conceded"


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
    assert _polarity_via(trace, lk_id) == "preference"     # meta broke the tie -> determinate
