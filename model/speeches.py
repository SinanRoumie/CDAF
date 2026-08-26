"""Fixed CDAF speech vocabulary -- domain schema, shared by model consumers.

The seven-speech order is a domain fact (the flow of a policy round), so it lives
with the model rather than being re-declared app-side. `"2NC/1NR"` (the neg
block) is a single combined speech. Liveness is always ordered by this sequence,
never by insertion order.
"""

SPEECH_ORDER = ["1AC", "1NC", "2AC", "2NC/1NR", "1AR", "2NR", "2AR"]

SPEECH_SIDE = {
    "1AC": "AFF", "1NC": "NEG", "2AC": "AFF", "2NC/1NR": "NEG",
    "1AR": "AFF", "2NR": "NEG", "2AR": "AFF",
}

# Which speeches are CONSTRUCTIVE (as opposed to rebuttal). This is a fact about the
# FORMAT, not about position -- SPEECH_ORDER cannot tell you a speech is constructive,
# so the schema must carry it. The neg block `"2NC/1NR"` is CONSTRUCTIVE: it contains
# the 2NC (its 1NR half is a rebuttal, but the block is one speech string and its
# constructive content is what matters for "a side's last constructive"). Consumers
# derive the rebuttal set from this label rather than hardcoding speech names, so the
# derivation transfers to any ordering instead of silently mis-firing on a relabel.
CONSTRUCTIVE_SPEECHES = frozenset({"1AC", "1NC", "2AC", "2NC/1NR"})

_INDEX = {s: i for i, s in enumerate(SPEECH_ORDER)}


def speech_index(speech: str) -> int:
    """Order index of a speech. Unknown speeches sort last (stable), so a
    malformed round still converts deterministically instead of raising."""
    return _INDEX.get(speech, len(SPEECH_ORDER))
