"""Extension carries a node FORWARD from a prior speech (Yaz ruling).

`Extend`/`Concede` is illegal on a node already carried in the current speech -- including
the speech it was introduced in (a node is stamped carried={introduction_speech} at birth).
So same-speech extends, and every 1AC extend, are impossible; only a node carried in a
STRICTLY PRIOR speech can be extended, and only once per speech.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import CDAFEnvironment, is_legal
from env.actions import Introduce, Extend, Concede, EndSpeech, NEW


def _fresh_1ac_with_advocacy():
    """Reset -> 1AC; introduce one advocacy; return (env, node_id)."""
    env = CDAFEnvironment(); env.reset()
    assert env.state.current_slot == "1AC"
    env.step(Introduce("", "advocacy", NEW, None))
    nid = next(iter(env.state.nodes))
    assert env.state.nodes[nid].carried == {"1AC"}          # stamped carried at birth
    return env, nid


def test_extend_illegal_in_introduction_speech():
    """A node cannot be extended/conceded in the same speech it was introduced (1AC)."""
    env, nid = _fresh_1ac_with_advocacy()
    assert env.state.already_carried_this_speech(nid)
    assert not is_legal(env.state, Extend(nid))
    assert not is_legal(env.state, Concede(nid))


def test_extend_legal_on_prior_speech_node_then_illegal_reextend():
    """From a later speech: the FIRST extend of a prior-speech node is legal (carries it
    forward); a second extend/concede in the SAME speech is an illegal no-op re-extend."""
    env, nid = _fresh_1ac_with_advocacy()
    env.step(EndSpeech())                                    # 1AC -> 1NC
    assert env.state.current_slot == "1NC"
    assert env.state.nodes[nid].carried == {"1AC"}          # not yet carried this speech
    # prior-speech node, first carriage this speech -> LEGAL
    assert is_legal(env.state, Extend(nid))
    assert is_legal(env.state, Concede(nid))
    env.step(Extend(nid))
    assert env.state.nodes[nid].carried == {"1AC", "1NC"}   # now carried this speech
    # same speech, already carried -> ILLEGAL re-extend (and re-concede)
    assert not is_legal(env.state, Extend(nid))
    assert not is_legal(env.state, Concede(nid))


def test_carry_forward_across_multiple_speeches():
    """A node stays extendable once per speech as the round advances -- legal the first time
    in each new speech, illegal a second time within it."""
    env, nid = _fresh_1ac_with_advocacy()
    for slot in ("1NC", "2AC", "2NC/1NR"):
        env.step(EndSpeech())
        assert env.state.current_slot == slot
        assert is_legal(env.state, Extend(nid))             # fresh speech -> legal
        env.step(Extend(nid))
        assert not is_legal(env.state, Extend(nid))         # re-extend same speech -> illegal
        assert slot in env.state.nodes[nid].carried
