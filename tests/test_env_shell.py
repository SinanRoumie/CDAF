"""Phase-1 environment shell tests: state schema, structural legality (incl. the
ruled local Fence-A rule), the monotonic-settled-facts observation, and the
reset()/step() contract with a single judge call at termination.

These pin the shell's CONTRACT, not the judge's semantics (the judge is exercised by
the oracle suite). The load-bearing assertions: strategic-illegal-but-structural moves
are ADMITTED (not filtered), Fence A is enforced locally without deferred repair, the
observation carries no provisional/verdict signal, and termination runs the judge
exactly once and maps its ballot to a binary reward.
"""

from env import (
    CDAFEnvironment, RoundState,
    Introduce, Extend, Concede, Weigh, EndSpeech,
    check_legality, is_legal, NEW, SPEECH_BUDGET, TOTAL_BUDGET,
)
from judge.config import AFF, NEG


def _last(state):
    return list(state.nodes)[-1]


def _drive(env, actions):
    out = None
    for a in actions:
        ok, why = check_legality(env.state, a)
        assert ok, f"expected legal, got {why}: {a}"
        out = env.step(a)
    return out


# --- reset / sequence --------------------------------------------------------

def test_reset_empty_graph_at_1AC():
    env = CDAFEnvironment()
    obs = env.reset()
    assert obs["graph"]["nodes"] == [] and obs["graph"]["edges"] == []
    assert obs["sequence"]["slot"] == "1AC"
    assert obs["sequence"]["side"] == AFF
    assert obs["sequence"]["remaining_budget"] == SPEECH_BUDGET["1AC"]
    assert TOTAL_BUDGET == 52


def test_end_speech_advances_side_and_slot():
    env = CDAFEnvironment(); env.reset()
    obs, r, done, info = env.step(EndSpeech())
    assert not done and r == 0.0
    assert obs["sequence"]["slot"] == "1NC" and obs["sequence"]["side"] == NEG


def test_budget_exhaustion_auto_advances():
    env = CDAFEnvironment(); env.reset()
    for _ in range(SPEECH_BUDGET["1AC"]):
        obs, r, done, info = env.step(Introduce("x", "link", NEW))
    # after the budget-th move the speech auto-advances to 1NC
    assert obs["sequence"]["slot"] == "1NC"
    assert obs["sequence"]["moves_used"] == 0


# --- structural legality: well-formedness ------------------------------------

def test_illegal_params_and_targets_rejected():
    env = CDAFEnvironment(); env.reset()
    env.step(Introduce("a", "advocacy", NEW))
    nid = _last(env.state)
    # bad role
    assert not is_legal(env.state, Introduce("x", "not_a_role", NEW))
    # NEW must not carry an edge_type
    assert not is_legal(env.state, Introduce("x", "link", NEW, "support"))
    # attach to missing target
    assert not is_legal(env.state, Introduce("x", "link", "n999", "support"))
    # bad edge_type on attach
    assert not is_legal(env.state, Introduce("x", "link", nid, "bogus"))
    # extend/weigh on missing node
    assert not is_legal(env.state, Extend("n999"))
    assert not is_legal(env.state, Weigh("n999", nid, nid))
    # weigh self / favors mismatch
    assert not is_legal(env.state, Weigh(nid, nid, nid))
    assert not is_legal(env.state, Weigh(nid, "n999", "n42"))
    # legal: attach a valid relationship, including own-side targeting (shared node)
    assert is_legal(env.state, Introduce("x", "link", nid, "support"))
    assert is_legal(env.state, Introduce("x", "impact", nid, "offensive_attack"))


def test_illegal_action_raises_in_step():
    env = CDAFEnvironment(); env.reset()
    try:
        env.step(Extend("n999"))
        assert False, "expected ValueError on illegal action"
    except ValueError:
        pass


# --- structural legality: strategic moves are NOT filtered -------------------

def test_strategic_illegal_but_structural_is_admitted():
    """A brand-new chain introduced in a rebuttal, and an attack drawn in the 'wrong'
    window, are STRATEGICALLY weak (the judge scores them inert / unresolved) but
    STRUCTURALLY legal. The generator must admit them so the agent gets the learning
    signal (spec §Governing principle)."""
    env = CDAFEnvironment(); env.reset()
    env.step(Introduce("adv", "advocacy", NEW)); adv = _last(env.state)
    # jump to the final speech without answering anything: 1AC->...->2AR is 6 advances
    for _ in range(6):
        env.step(EndSpeech())
    assert env.state.current_slot == "2AR"
    # a fresh chain first introduced in the 2AR: structurally legal (judge rules it inert)
    assert is_legal(env.state, Introduce("late impact", "impact", NEW))
    assert is_legal(env.state, Introduce("late link", "link", adv, "offensive_attack"))


# --- Fence A: local, no deferred repair --------------------------------------

def test_fence_A_blocks_second_terminal_impact_but_allows_chaining():
    """A reachable same-side Support component may not gain a 2nd terminal impact.
    Introducing a sibling impact onto the same link is rejected; chaining the new
    impact FORWARD off the existing one (so the old impact is no longer terminal) is
    admitted -- construction-order artifact, not a debate move (Ruling 2)."""
    env = CDAFEnvironment(); env.reset()
    env.step(Introduce("link", "link", NEW)); lk = _last(env.state)
    env.step(Introduce("imp1", "impact", lk, "support")); im1 = _last(env.state)
    env.step(Introduce("vote", "ballot_directive", im1, "support"))   # makes it reachable
    # sibling terminal impact on the same link, same speech -> 2 terminals -> rejected
    ok, why = check_legality(env.state, Introduce("imp2", "impact", lk, "support"))
    assert not ok and "multi-terminal" in why
    # chaining a new impact FORWARD off im1 in a LATER speech de-terminalizes im1
    # (the judge's terminal rule needs a strictly-later-speech forward node) -> admitted
    env.step(EndSpeech())                    # -> 1NC
    env.step(EndSpeech())                    # -> 2AC (AFF again)
    assert is_legal(env.state, Introduce("imp2", "impact", im1, "support"))
    # a same-speech sibling is still blocked here too
    assert not is_legal(env.state, Introduce("imp3", "impact", lk, "support"))


# --- observation: monotonic settled facts only -------------------------------

def test_observation_excludes_provisional_signal():
    env = CDAFEnvironment(); env.reset()
    env.step(Introduce("adv", "advocacy", NEW))
    obs = env.step(Introduce("imp", "impact", NEW))[0]
    keys = set(obs)
    assert keys == {"graph", "closed_window_drops",
                    "permanent_extension_failures", "reachability", "sequence"}
    # no magnitude / verdict / eligibility leakage anywhere in the observation
    flat = repr(obs).lower()
    for banned in ("magnitude", "sigma", "verdict", "winner", "reward", "delta", "eligib"):
        assert banned not in flat, f"observation leaked provisional field: {banned}"


def test_closed_window_drop_is_settled_and_reachability():
    env = CDAFEnvironment(); env.reset()
    # 1AC: link + impact + BD (reachable chain); 2AC introduced impact orphan
    env.step(Introduce("lk", "link", NEW)); lk = _last(env.state)
    env.step(Introduce("im", "impact", lk, "support")); im = _last(env.state)
    env.step(Introduce("vote", "ballot_directive", im, "support"))
    # a non-impact node disconnected from any impact is orphaned (an impact would
    # trivially "route to an impact" -- itself -- so use a floating link)
    env.step(Introduce("floating", "link", NEW)); orphan = _last(env.state)
    # advance past 1NC (the link's response window) with no NEG clash
    env.step(EndSpeech())                    # -> 1NC
    env.step(EndSpeech())                    # -> 2AC ; now 1NC window has passed
    obs = env.reset() if False else __import__("env").observe(env.state)
    assert lk in obs["closed_window_drops"]  # window passed, no opposing clash -> settled
    assert obs["reachability"][lk] is True and obs["reachability"][im] is True
    assert obs["reachability"][orphan] is False


# --- termination: judge called once, binary reward ---------------------------

def test_empty_round_terminates_to_presumption_NEG():
    env = CDAFEnvironment(); env.reset()
    done = False; info = None; r = None
    for _ in range(len(SPEECH_BUDGET)):
        obs, r, done, info = env.step(EndSpeech())
    assert done is True
    assert info["winner"] == NEG            # no AFF offense -> presumption
    assert info["rewards"] == {AFF: 0.0, NEG: 1.0}
    assert r == 0.0                          # AFF-perspective reward


def test_liveness_status_is_derived_structurally_not_from_verb():
    """`to_round` stamps contested/conceded from graph structure (convert-faithful),
    never from extend-vs-concede. A cross-side attack contests the target's instance
    that was live when the opponent spoke; every other carried speech is conceded."""
    from model.nodes import CONTESTED, CONCEDED
    env = CDAFEnvironment(); env.reset()
    env.step(Introduce("lk", "link", NEW)); lk = _last(env.state)
    env.step(Introduce("im", "impact", lk, "support"))
    env.step(Introduce("vote", "ballot_directive", _last(env.state), "support"))
    env.step(EndSpeech())                              # -> 1NC (NEG)
    # NEG attacks the AFF link in 1NC (the link's response window)
    env.step(Introduce("no link", "link", lk, "offensive_attack"))
    env.step(EndSpeech())                              # -> 2AC
    env.step(Extend(lk))                               # AFF carries the link
    env.step(EndSpeech())
    for _ in range(4):                                 # 2NC/1NR, 1AR, 2NR, 2AR ... terminate
        if not env.state.terminated:
            env.step(EndSpeech())
    # materialize the link node and inspect its stamped liveness
    materialized = {n.id: n for n in env.state.to_round().nodes}
    live = materialized[lk].liveness
    assert live.get("1AC") == CONTESTED     # link was live & answered by the 1NC attack
    assert live.get("2AC") == CONCEDED      # carried later, unanswered that speech


def test_full_conceded_aff_chain_wins_aff():
    env = CDAFEnvironment(); env.reset()
    env.step(Introduce("plan", "advocacy", NEW)); adv = _last(env.state)
    env.step(Introduce("uq", "uniqueness", adv, "support")); uni = _last(env.state)
    env.step(Introduce("link", "link", uni, "support")); lk = _last(env.state)
    env.step(Introduce("impact", "impact", lk, "support")); im = _last(env.state)
    env.step(Introduce("vote aff", "ballot_directive", im, "support"))
    spine = (adv, uni, lk, im)
    env.step(EndSpeech())                    # 1AC done
    env.step(EndSpeech())                    # 1NC concedes
    for n in spine: env.step(Extend(n))      # 2AC
    env.step(EndSpeech())
    env.step(EndSpeech())                    # 2NC/1NR
    for n in spine: env.step(Extend(n))      # 1AR
    env.step(EndSpeech())
    env.step(EndSpeech())                    # 2NR
    for n in spine: env.step(Extend(n))      # 2AR
    obs, r, done, info = env.step(EndSpeech())
    assert done and info["winner"] == AFF and r == 1.0
