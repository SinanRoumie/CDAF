"""Verdict-oracle harness (judge_spec §11).

Each §11 round is built PROGRAMMATICALLY as a model.Round with liveness set per
the spec (not drawn in the builder), one test per round, asserting the stated
winner plus the key trace fact that localizes the decision to one pass. Each
round is surgical -- it isolates one mechanism.

Notes on spec-vs-mechanics (surfaced for review, not silently asserted; see
PROGRESS.md for the deferred turn-offense milestone):
  * Rounds 2/4/5/8 win for NEG via an AFF chain that COLLAPSED, so reason_class
    is "AFF structural failure" -- correct per §7's own definition (an AFF chain
    existed but collapsed / failed a gate). §11's looser word "presumption" for
    some of these is the label to tighten later; the judge follows §7.
  * Round 3's mitigation resolves to full magnitude (1.0), not ~0.5: a conceded
    counter removes the defender entirely, and V1 has no fractional strengths.
    The winner (AFF) is unchanged.
  * Rounds 4 and 5 are NOT distinguished as offense-vs-presumption, and neither
    verifies NEG winning ON turn offense: the current turn mechanics (an offensive
    attack drives the target to sigma 0, and extension is read node-side, which
    AFF conceded) resolve both to NEG via the AFF chain's collapse. Winner NEG is
    correct for both. Awarding NEG offense FROM a turn is the deferred
    turn-offense milestone (PROGRESS.md), out of scope here.
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


# --- §11.4 Link turn collapses the AFF chain -> NEG ---------------------------

def test_r4_link_turn_collapses_aff_chain_neg():
    """NEG turns the 1AC link in 1NC (an OffensiveAttack); AFF concedes it.

    WHAT THIS ACTUALLY VERIFIES: NEG wins because the turn *collapses the AFF
    chain* -- the AFF link is driven to sigma 0 (mag -> 0) and, since AFF walked
    away, the chain fails its own-side extension (EXTENSION_FAIL). The verdict is
    NEG via AFF's structural failure.

    WHAT THIS DOES NOT VERIFY: NEG winning ON live turn offense. The judge cannot
    yet award NEG offense FROM a turn (turns zero the target and extension is read
    node-side, i.e. AFF's, who conceded). That is the deferred turn-offense
    milestone -- see PROGRESS.md. So this is not a green check for turn offense;
    it only checks that a conceded turn takes the AFF advantage off the table."""
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
    # NEG wins via the AFF chain's collapse, NOT via NEG turn offense:
    assert any(r.kind == "EXTENSION_FAIL" for r in trace)
    assert _ballot(trace).reason_class == "AFF structural failure"


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
