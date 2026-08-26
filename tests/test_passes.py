"""J3/E2 gate -- per-pass unit tests on small constructed Rounds.

Model C: extension is per-node LIVENESS, not an edge. A node built here carries a
liveness record; by default `node()` stamps FULL own-side liveness (introduction
through the side's final speech) so a clean chain is fully extended. Tests that
exercise a gap pass an explicit partial `live=`.
"""

from model import (
    Round, Advocacy, Uniqueness, Link, Impact, Framework, Weighing, BallotDirective,
    Support, DefensiveAttack, OffensiveAttack, Comparison,
    SPEECH_ORDER, SPEECH_SIDE, CONTESTED, CONCEDED,
)

from judge import passes, judge as judge_mod
from judge.config import AFF, NEG, EPSILON
from judge.judge import judge


# --- builders -----------------------------------------------------------------

_ID = [0]


def _nid():
    _ID[0] += 1
    return f"n{_ID[0]}"


def _full_liveness(side, speech):
    """Own-side speeches from `speech` onward, all conceded (fully extended)."""
    intro = SPEECH_ORDER.index(speech)
    return {s: CONCEDED for i, s in enumerate(SPEECH_ORDER)
            if SPEECH_SIDE[s] == side and i >= intro}


def node(cls, side, speech, label="x", live="full"):
    """Build a node. `live="full"` => extended through the side's final speech;
    pass a dict for a partial/gapped record, or None for the model default."""
    liveness = _full_liveness(side, speech) if live == "full" else live
    return cls(id=_nid(), label=label, side=side, speech=speech, liveness=liveness)


def support(a, b):      # spine connectivity
    return Support(id=_nid(), source=a.id, target=b.id)


def datk(a, b):
    return DefensiveAttack(id=_nid(), source=a.id, target=b.id)


def oatk(a, b):
    return OffensiveAttack(id=_nid(), source=a.id, target=b.id)


def aff_chain_extended():
    """A clean AFF advantage, every spine node extended 1AC->2AR (liveness),
    anchored by a BD. Returns (elements, dict of key nodes)."""
    adv = node(Advocacy, AFF, "1AC"); uni = node(Uniqueness, AFF, "1AC")
    link = node(Link, AFF, "1AC"); imp = node(Impact, AFF, "1AC")
    bd = node(BallotDirective, AFF, "2AR")
    els = [adv, uni, link, imp, bd,
           support(adv, link), support(uni, link), support(link, imp), support(imp, bd)]
    return els, {"adv": adv, "uni": uni, "link": link, "imp": imp, "bd": bd}


# --- Pass 1: discovery (undirected, from a BD) --------------------------------

def test_discovery_is_undirected_and_bd_anchored():
    a = node(Impact, AFF, "1AC")
    bd = node(BallotDirective, AFF, "2AR")
    orphan = node(Impact, NEG, "1NC")           # not connected to any BD
    # edge drawn impact->bd here; reverse should not matter
    rnd = Round(elements=[a, bd, orphan, support(a, bd)])
    ctx = passes.build_context(rnd)
    assert a.id in ctx.reachable and bd.id in ctx.reachable
    assert orphan.id not in ctx.reachable        # only BD-reachable subgraphs evaluated


def test_discovery_ignores_edge_direction():
    a = node(Impact, AFF, "1AC")
    bd = node(BallotDirective, AFF, "2AR")
    # draw the support edge bd->a (the "wrong" way); discovery must still reach a
    rnd = Round(elements=[a, bd, Support(id=_nid(), source=bd.id, target=a.id)])
    ctx = passes.build_context(rnd)
    assert a.id in ctx.reachable


# --- Pass 2: drop vs answered vs no-window ------------------------------------

def test_dropped_node_when_window_passes_unanswered():
    # AFF link in 1AC, NEG never attacks it -> dropped (conceded) in window 1NC
    adv = node(Advocacy, AFF, "1AC"); link = node(Link, AFF, "1AC")
    bd = node(BallotDirective, AFF, "2AR")
    rnd = Round(elements=[adv, link, bd, support(adv, link), support(link, bd)])
    ctx = passes.build_context(rnd); passes.pass2_drops(ctx)
    assert ctx.status[link.id] == "dropped"
    assert any(r.kind == "DROP" and r.node_id == link.id for r in ctx.trace)


def test_answered_node_when_clash_in_window():
    link = node(Link, AFF, "1AC")               # AFF, window = 1NC
    atk = node(Link, NEG, "1NC")                # NEG attacks in the window
    bd = node(BallotDirective, AFF, "2AR")
    # edge drawn atk->link AND we don't rely on direction
    rnd = Round(elements=[link, atk, bd, support(link, bd), datk(atk, link)])
    ctx = passes.build_context(rnd); passes.pass2_drops(ctx)
    assert ctx.status[link.id] == "answered"


def test_no_window_is_unresolved():
    # Impact introduced in 2AR (final AFF speech) -> opponent never had standing
    imp = node(Impact, AFF, "2AR")
    bd = node(BallotDirective, AFF, "2AR")
    rnd = Round(elements=[imp, bd, support(imp, bd)])
    ctx = passes.build_context(rnd); passes.pass2_drops(ctx)
    assert ctx.status[imp.id] == "unresolved"
    assert any(r.kind == "UNRESOLVED" and r.node_id == imp.id for r in ctx.trace)


# --- Pass 3: accrual + chain + binary extension -------------------------------

def test_clean_extended_chain_delta_is_plus_one():
    els, k = aff_chain_extended()
    rnd = Round(elements=els)
    ctx = passes.build_context(rnd)
    passes.pass2_drops(ctx); passes.pass_accrual(ctx); passes.pass5_chains(ctx)
    chains = [c for c in ctx.chains if c["side"] == AFF]
    assert len(chains) == 1
    ch = chains[0]
    assert ch["extended"] is True
    assert ch["sign"] == 1
    assert abs(ch["mag"] - 1.0) < 1e-9
    assert abs(ch["delta"] - 1.0) < 1e-9


def test_conceded_defense_collapses_chain_magnitude():
    # As clean chain, but NEG reads a conceded defensive attack on the 1AC link.
    els, k = aff_chain_extended()
    defender = node(Link, NEG, "1NC")           # attacks AFF link in its window, AFF drops it
    els += [defender, datk(defender, k["link"])]
    ctx = passes.build_context(Round(elements=els))
    passes.pass2_drops(ctx); passes.pass_accrual(ctx); passes.pass5_chains(ctx)
    ch = [c for c in ctx.chains if c["side"] == AFF][0]
    assert ch["mag"] < EPSILON                  # link sigma -> 0 collapses the product
    assert ctx.sigma[k["link"].id] == 0.0


def test_extension_failure_when_spine_not_carried():
    # AFF link's liveness stops at 2AC (missing 1AR/2AR) -> chain fails §6.
    adv = node(Advocacy, AFF, "1AC"); uni = node(Uniqueness, AFF, "1AC")
    link = node(Link, AFF, "1AC", live={"1AC": CONCEDED, "2AC": CONCEDED})  # gap at 1AR
    imp = node(Impact, AFF, "1AC")
    bd = node(BallotDirective, AFF, "2AR")
    els = [adv, uni, link, imp, bd,
           support(adv, link), support(uni, link), support(link, imp), support(imp, bd)]
    ctx = passes.build_context(Round(elements=els))
    passes.pass2_drops(ctx); passes.pass_accrual(ctx); passes.pass5_chains(ctx)
    ch = [c for c in ctx.chains if c["side"] == AFF][0]
    assert ch["extended"] is False
    assert any(r.kind == "EXTENSION_FAIL" and r.missing_speech == "1AR" for r in ctx.trace)


# --- Terminality follows node type, not introduction order (§2.2) -------------

def test_later_convergent_link_does_not_break_impact_terminality():
    """Regression: a LINK introduced LATER than the impact it supports must NOT
    disqualify that impact from being terminal -- a link is always a premise upstream of
    its impact (§2.2 type orientation), never downstream. Pre-fix, `_terminals` read the
    later link as 'chaining forward', so no node in the component was terminal and NO
    chain was emitted for a legitimately-scoring impact. A legal
    `introduce(link -> impact, support)` in a later speech reaches this state, so it was
    a live self-play/warm-start correctness hole."""
    adv = node(Advocacy, AFF, "1AC"); uni = node(Uniqueness, AFF, "1AC")
    lk_early = node(Link, AFF, "1AC")
    imp = node(Impact, AFF, "1AC")
    lk_late = node(Link, AFF, "2AC")                 # convergent premise link, LATER than the impact
    bd = node(BallotDirective, AFF, "1AC")
    els = [adv, uni, lk_early, imp, lk_late, bd,
           support(adv, lk_early), support(uni, imp),
           support(lk_early, imp), support(lk_late, imp),   # both links support the shared impact
           support(imp, bd)]
    ctx = passes.build_context(Round(elements=els))
    passes.pass2_drops(ctx); passes.pass_accrual(ctx); passes.pass5_chains(ctx)
    assert imp.id in passes._terminals(ctx, list(ctx.reachable), [imp.id])   # still terminal
    ch = [c for c in ctx.chains if c["side"] == AFF]
    assert len(ch) == 1 and ch[0]["extended"] and ch[0]["sign"] == 1        # chain emitted, scores
    assert imp.id in ch[0]["impacts"]


def test_later_impact_still_disqualifies_terminality_recency_retained():
    """Guard for the narrowing: only LINKS were removed from the forward check. A later
    IMPACT chained forward from an earlier one must STILL make the earlier impact
    non-terminal (the recency-governed impact->impact case is deliberately retained --
    node type alone cannot order two impacts)."""
    adv = node(Advocacy, AFF, "1AC"); lk = node(Link, AFF, "1AC")
    im1 = node(Impact, AFF, "1AC")
    im2 = node(Impact, AFF, "2AC")                   # later impact, downstream of im1
    bd = node(BallotDirective, AFF, "1AC")
    els = [adv, lk, im1, im2, bd,
           support(adv, lk), support(lk, im1), support(im1, im2), support(im2, bd)]
    ctx = passes.build_context(Round(elements=els))
    passes.pass2_drops(ctx); passes.pass_accrual(ctx); passes.pass5_chains(ctx)
    terminals = passes._terminals(ctx, list(ctx.reachable), [im1.id, im2.id])
    assert im2.id in terminals and im1.id not in terminals   # later impact is the terminal one


# --- Pass 5a: framework gate in/out -------------------------------------------

def test_framework_gate_excludes_unsupported_impact():
    # Winning framework anchors chain A (in scope) but not chain B (out). v6 gates
    # PER CHAIN by BD-blocking anchoring (§5.3), so A and B are SEPARATE chains
    # (own BDs): chain A's impact supports the framework directly (im->F, an
    # anchor); chain B's impact only co-supports its BD (not an anchor). Framework
    # introduced in 2NR (final NEG speech) so it is trivially maker-extended and
    # unattacked -> the sole live framework -> governs.
    fw = node(Framework, NEG, "2NR")
    impA = node(Impact, AFF, "1AC"); impB = node(Impact, AFF, "1AC")
    bdA = node(BallotDirective, AFF, "2AR"); bdB = node(BallotDirective, AFF, "2AR")
    els = [fw, impA, impB, bdA, bdB,
           support(impA, bdA), support(impB, bdB),
           support(fw, impA)]                   # only chain A is anchored to the framework
    ctx = passes.build_context(Round(elements=els))
    passes.pass2_drops(ctx); passes.pass_accrual(ctx); passes.pass5_chains(ctx); passes.pass6_framework(ctx)
    assert ctx.winning_framework is not None
    gates = {(r.impact_id, r.in_scope) for r in ctx.trace if r.kind == "FRAMEWORK_GATE"}
    assert (impA.id, True) in gates          # anchored to the winning framework
    assert (impB.id, False) in gates         # only co-supports its BD -> not anchored


# --- Pass 5b + 6: weighing preference overriding raw delta --------------------

def _two_impact_round_with_weighing(weigh_side):
    """A clean AFF impact and a clean NEG impact (both extended & anchored), plus
    a conceded weighing on `weigh_side` comparing the two."""
    els = []
    aff_imp = _strength_chain(els, AFF, add_defense=False)
    neg_imp = _strength_chain(els, NEG, add_defense=False)
    aff_bd = node(BallotDirective, AFF, "2AR"); neg_bd = node(BallotDirective, NEG, "2NR")
    els += [aff_bd, neg_bd, support(aff_imp, aff_bd), support(neg_imp, neg_bd)]
    w = node(Weighing, weigh_side, "2NR" if weigh_side == NEG else "2AR")
    els += [w, Comparison(id=_nid(), source=w.id, target=aff_imp.id),
            Comparison(id=_nid(), source=w.id, target=neg_imp.id)]
    return Round(elements=els)


def _strength_chain(els, side, add_defense):
    """Build a minimal fully-extended chain on `side` (liveness), anchored later
    by the caller. If `add_defense`, a conceded opposing defensive attacker in the
    link's window drives the link (and chain) to zero."""
    intro = "1AC" if side == AFF else "1NC"
    adv = node(Advocacy, side, intro); uni = node(Uniqueness, side, intro)
    link = node(Link, side, intro); imp = node(Impact, side, intro)
    els += [adv, uni, link, imp, support(adv, link), support(uni, link), support(link, imp)]
    if add_defense:
        opp = NEG if side == AFF else AFF
        win = passes.response_window(intro, side)
        els.append(node(Link, opp, win))
        els.append(datk(els[-1], link))
    return imp


def test_weighing_preference_overrides_raw_delta():
    # NEG wins a conceded weighing -> AFF (larger raw delta) excluded -> NEG.
    rnd = _two_impact_round_with_weighing(NEG)
    ballot, trace = judge(rnd)
    assert any(r.kind == "WEIGH" and r.outcome == "resolved" for r in trace)
    assert ballot == NEG


# --- Pass 6: BD validation + asymmetric net offense ---------------------------

def test_clean_aff_round_aff_wins():
    els, k = aff_chain_extended()
    ballot, trace = judge(Round(elements=els))
    assert ballot == AFF
    b = [r for r in trace if r.kind == "BALLOT"][0]
    assert b.N > EPSILON and b.winner == AFF
    assert "advocacy" in b.gates_passed and "N>eps" in b.gates_passed


def test_bd_validation_fails_when_no_offense():
    # AFF chain present but link conceded-defeated -> mag 0 -> BD fails -> NEG.
    els, k = aff_chain_extended()
    defender = node(Link, NEG, "1NC")
    els += [defender, datk(defender, k["link"])]
    ballot, trace = judge(Round(elements=els))
    assert ballot == NEG
    assert any(r.kind == "BD_VALIDATE" and r.result == "fail" for r in trace)


def test_empty_round_drains_to_neg_presumption():
    ballot, trace = judge(Round(elements=[]))
    assert ballot == NEG
    assert [r for r in trace if r.kind == "BALLOT"][0].winner == NEG


# --- Part 1 (J4a): polarity flip gated on OffensiveAttack (§3.2) --------------

def test_defensive_only_link_keeps_polarity_when_driven_to_zero():
    # A conceded DEFENSIVE attack drives the AFF link to sigma 0: dead, not turned.
    link = node(Link, AFF, "1AC")
    defender = node(Link, NEG, "1NC")            # defensive, conceded (AFF never answers)
    bd = node(BallotDirective, AFF, "2AR")
    rnd = Round(elements=[link, defender, bd, support(link, bd), datk(defender, link)])
    ctx = passes.build_context(rnd)
    passes.pass2_drops(ctx); passes.pass_accrual(ctx); passes.pass5_chains(ctx)
    assert link.id not in ctx.offense_on         # only defensively attacked
    assert ctx.sigma[link.id] == 0.0
    assert ctx.eff_pol[link.id] == 1             # keeps +1 (dead, not turned)
    assert any(r.kind == "MAGNITUDE" and r.node_id == link.id for r in ctx.trace)
    assert not any(r.kind == "POLARITY_FLIP" and r.link_id == link.id for r in ctx.trace)


def test_defensive_only_link_at_0_3_keeps_polarity_no_flip():
    # Defensively mitigated to sigma 0.3 (< 0.5): without the gate this would flip.
    link = node(Link, AFF, "1AC")
    defender = node(Link, NEG, "1NC")
    bd = node(BallotDirective, AFF, "2AR")
    rnd = Round(elements=[link, defender, bd, support(link, bd), datk(defender, link)])
    ctx = passes.build_context(rnd)
    passes.pass2_drops(ctx); passes._classify_attacks(ctx); passes._resolve_strengths(ctx)
    assert link.id not in ctx.offense_on
    ctx.sigma[link.id] = 0.3                      # simulate defensive mitigation to 0.3
    ctx.trace.clear(); ctx.eff_pol = {}
    passes._resolve_polarity(ctx)
    assert ctx.eff_pol[link.id] == 1             # < 0.5 but defensive-only -> keeps +1
    assert not any(r.kind == "POLARITY_FLIP" and r.link_id == link.id for r in ctx.trace)


def test_offensively_attacked_link_below_threshold_flips():
    # Symmetric case: an OffensiveAttack winning the link below 0.5 DOES flip to -1.
    link = node(Link, AFF, "1AC")
    offender = node(Link, NEG, "1NC")            # offensive, conceded -> link sigma 0
    bd = node(BallotDirective, AFF, "2AR")
    rnd = Round(elements=[link, offender, bd, support(link, bd), oatk(offender, link)])
    ctx = passes.build_context(rnd)
    passes.pass2_drops(ctx); passes.pass_accrual(ctx); passes.pass5_chains(ctx)
    assert link.id in ctx.offense_on             # target of an offensive attack
    assert ctx.sigma[link.id] == 0.0
    assert ctx.eff_pol[link.id] == -1            # below 0.5 -> flips
    flips = [r for r in ctx.trace if r.kind == "POLARITY_FLIP" and r.link_id == link.id]
    assert len(flips) == 1 and flips[0].to_sign == -1


def test_offensively_attacked_link_at_0_3_flips_at_fractional_sigma():
    link = node(Link, AFF, "1AC")
    offender = node(Link, NEG, "1NC")
    bd = node(BallotDirective, AFF, "2AR")
    rnd = Round(elements=[link, offender, bd, support(link, bd), oatk(offender, link)])
    ctx = passes.build_context(rnd)
    passes.pass2_drops(ctx); passes._classify_attacks(ctx); passes._resolve_strengths(ctx)
    assert link.id in ctx.offense_on
    ctx.sigma[link.id] = 0.3
    ctx.trace.clear(); ctx.eff_pol = {}
    passes._resolve_polarity(ctx)
    assert ctx.eff_pol[link.id] == -1            # offensive + below 0.5 -> flips
    assert any(r.kind == "POLARITY_FLIP" and r.link_id == link.id for r in ctx.trace)


# --- Part 2: RFD renderer (pure, causally inert) ------------------------------

from judge import rfd  # noqa: E402


def test_rfd_is_pure_and_changes_no_verdict():
    els, k = aff_chain_extended()
    rnd = Round(elements=els)
    ballot, trace = judge(rnd)
    trace_len_before = len(trace)
    text = rfd.render(ballot, trace)
    # rendering must not mutate the trace or re-decide
    assert len(trace) == trace_len_before
    ballot2, _ = judge(rnd)
    assert ballot2 == ballot
    assert isinstance(text, str) and text


def test_rfd_reports_winner_reason_and_decomposition_for_clean_aff():
    els, k = aff_chain_extended()
    ballot, trace = judge(Round(elements=els))
    text = rfd.render(ballot, trace)
    assert "AFF wins" in text
    assert "affirmative carries net offense" in text
    assert "Net offense N = +1.000" in text
    assert "contributes delta = +1.000" in text


def test_rfd_reports_collapse_reason_and_bd_failure():
    els, k = aff_chain_extended()
    defender = node(Link, NEG, "1NC")
    els += [defender, datk(defender, k["link"])]
    ballot, trace = judge(Round(elements=els))
    text = rfd.render(ballot, trace)
    assert "NEG wins" in text
    assert "killed by conceded defense" in text          # collapse reason named
    assert "ballot directive" in text and "failed" in text


def test_rfd_reports_weighing_override():
    rnd = _two_impact_round_with_weighing(NEG)
    ballot, trace = judge(rnd)
    text = rfd.render(ballot, trace)
    assert "Weighing" in text
    assert "preferred" in text


def test_rfd_turned_contributor_reads_as_one_statement():
    """A turned chain that generates opposing offense (§3.5) is ONE contribution;
    the RFD must render it as a single coherent line -- not as both a collapsed
    AFF argument AND a negative-delta contributor. Guards the display against
    re-suggesting a double-count that the ballot arithmetic does not make."""
    adv = node(Advocacy, AFF, "1AC"); uni = node(Uniqueness, AFF, "1AC")
    link = node(Link, AFF, "1AC",
                live={"1AC": CONTESTED, "1NC": CONTESTED, "2NC/1NR": CONTESTED, "2NR": CONTESTED})
    imp = node(Impact, AFF, "1AC",
               live={"1AC": CONCEDED, "1NC": CONCEDED, "2NC/1NR": CONCEDED, "2NR": CONCEDED})
    turn = node(Link, NEG, "1NC")
    bd = node(BallotDirective, NEG, "2NR")
    els = [adv, uni, link, imp, turn, bd,
           support(adv, link), support(uni, link), support(link, imp), support(imp, bd),
           oatk(turn, link)]
    ballot, trace = judge(Round(elements=els))
    assert ballot == NEG
    text = rfd.render(ballot, trace)
    chain_id = [r for r in trace if r.kind == "CHAIN"][0].chain_id
    # the turned chain is narrated ONCE, as a turn that transfers offense:
    assert text.count(chain_id) == 1
    assert "was turned: it no longer carries AFF offense and now carries NEG offense" in text
    # and it is NOT ALSO listed as a plain contributor or under collapsed arguments:
    assert "survives and contributes" not in text
    assert "Collapsed arguments" not in text


# --- REBUTTAL_SPEECHES derivation (§6) ----------------------------------------

def test_rebuttal_speeches_pinned_on_policy_ordering():
    """DURABLE PIN. On the pinned policy ordering the derived rebuttal set MUST be
    exactly {1AR, 2NR, 2AR} -- the value the oracle fixtures were adjudicated under.
    This lives in the suite (with the fixtures' policy labels) so a format edit that
    changes CONSTRUCTIVE_SPEECHES / SPEECH_ORDER and breaks the policy value fails
    LOUDLY here rather than silently letting a new rebuttal chain establish offense."""
    assert passes.REBUTTAL_SPEECHES == frozenset({"1AR", "2NR", "2AR"})


def test_rebuttal_speeches_are_derived_not_hardcoded():
    """The set is 'each side's speeches strictly after its last constructive', read off
    the model-owned label -- so it is disjoint from the constructives (incl. the neg
    block) and every member is a real speech, on whatever ordering is in force."""
    from model import CONSTRUCTIVE_SPEECHES
    assert passes.REBUTTAL_SPEECHES <= set(SPEECH_ORDER)
    assert passes.REBUTTAL_SPEECHES.isdisjoint(CONSTRUCTIVE_SPEECHES)
    assert "2AC" not in passes.REBUTTAL_SPEECHES        # a constructive add-on speech
    assert "2NC/1NR" not in passes.REBUTTAL_SPEECHES    # the neg block is constructive
