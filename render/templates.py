"""M0 templates (spec §8; RS23 budgets, RS26 extensions, RS27 markers, RS13c).

Node-type x edge-type -> a canned placeholder sentence with a claim slot. No
content, no LLM, deterministic. Every line carries its RS23 word budget; the
budget (not the placeholder text) is what `words per speech` sums, since M0
content is a stand-in.

Type-blind (RS5/RS6): the only inputs are node kind, edge kind, side, and the
judge-emitted invisibility reason. No "disad"/"kritik"/"theory" identifiers.
"""
from __future__ import annotations

from typing import Optional, Tuple

# RS23 per-event word budgets.
BUDGET_NODE = 50            # claim 15 / warrant 35
BUDGET_EDGE_SAME = 0        # both endpoints introduced this speech: syntax only
BUDGET_EDGE_XAPP = 50       # Support, >=1 endpoint pre-existing: cross-application
BUDGET_EDGE_REFUTE = 50     # DefensiveAttack: refutation
BUDGET_EDGE_ATTACK = 50     # OffensiveAttack: turn
BUDGET_EDGE_COMPARE = 0     # Comparison (weighing wiring): syntax only
BUDGET_EXTENSION = 15       # RS26: claim restatement only, no warrant

# Canned per-kind claim stems (RS19 claim slot). Warrant is a fixed 35-word slot.
#
# 'advocacy' is intentionally absent: its claim stem is per-node, not per-kind
# (RS19 correction note). A single per-kind string collapses two structurally
# distinct advocacies into one and hardcodes a policy plan. The advocacy stem is
# built by advocacy_stem() below, keyed on side (RS5b polarity) plus a per-side-
# per-speech parallel discriminator; linearize passes the result via stem_override.
_NODE_STEM = {
    "uniqueness": "the status quo holds",
    "link": "this causes the next step",
    "impact": "and that outcome matters",
    "framework": "evaluate the round this way",
    "weighing": "prefer this consideration",
    "ballot_directive": "vote here",
}
_WARRANT = "[warrant ×35w: reasoning-only placeholder, no card, no cite]"

# Parallel (NOT ranked) discriminators for the 2nd+ same-side advocacy in one
# speech. "first"/"second" would invite the model to read a later advocacy as
# derivative; advocacy nodes in a component are unordered (RS19 correction note).
_ADV_DISTINCT = {
    2: "a separate", 3: "a third, distinct", 4: "a fourth, distinct",
    5: "a fifth, distinct", 6: "a sixth, distinct", 7: "a seventh, distinct",
}


def advocacy_stem(side: str, ordinal: int) -> str:
    """RS19 correction / RS5b: the advocacy claim stem.

    Polarity from side alone — an AFF advocacy affirms the resolution, a NEG
    advocacy is a competing advocacy (counterplan or alternative), never "the plan".
    Type-neutral: no argument-type name, no policy-vs-value assumption. `ordinal`
    (1-based, scoped per side per speech) disambiguates multiple advocacies in one
    speech with a parallel, non-ranked phrase so two distinct advocacies never
    render as one; ordinal 1 carries no discriminator.
    """
    noun = "advocacy" if side == "AFF" else "competing advocacy"
    if ordinal <= 1:
        return "we advance our advocacy" if side == "AFF" else "we advance a competing advocacy"
    disc = _ADV_DISTINCT.get(ordinal, f"a {ordinal}th, distinct")
    return f"we advance {disc} {noun}"


def _tag(node) -> str:
    return f"{node.kind}:{node.id}"


def node_line(node, *, stem_override: Optional[str] = None) -> Tuple[str, int]:
    """RS23 new node: 50 words (claim 15 / warrant 35). `stem_override` carries a
    per-node claim stem (advocacy — RS19 correction note); absent it, the per-kind
    stem is used."""
    stem = stem_override if stem_override is not None else _NODE_STEM.get(node.kind, node.kind)
    claim = f"[claim: {_tag(node)} — {stem}]"
    return f"{claim} {_WARRANT}", BUDGET_NODE


def edge_same_line(edge, src, tgt) -> Tuple[str, int]:
    """RS23 edge in same speech as both endpoints: 0 words, syntax only."""
    return (f"[…which means… {edge.kind} {src.id}→{tgt.id}, same-speech, syntax only]",
            BUDGET_EDGE_SAME)


def edge_xapp_line(edge, src, tgt) -> Tuple[str, int]:
    """RS25 Support cross-application: 50 words. Endpoint content untouched."""
    return (f"[cross-apply: support {src.id}→{tgt.id} — extend our link, it applies here too] "
            f"{_WARRANT.replace('×35w', '×50w')}",
            BUDGET_EDGE_XAPP)


def edge_refute_line(edge, attacker, target) -> Tuple[str, int]:
    """RS25 DefensiveAttack: 50 words. Claim is the takeout; warrant why it fails."""
    return (f"[refute: {attacker.id} ─defensive→ {target.id} — takeout] "
            f"{_WARRANT.replace('×35w', '×50w')}",
            BUDGET_EDGE_REFUTE)


def edge_attack_line(edge, attacker, target) -> Tuple[str, int]:
    """OffensiveAttack (turn): 50 words. Not in RS23's table by name; rendered as
    a 50-word attack line so the move is visible."""
    return (f"[turn: {attacker.id} ─offensive→ {target.id}] "
            f"{_WARRANT.replace('×35w', '×50w')}",
            BUDGET_EDGE_ATTACK)


def edge_compare_line(edge, src, tgt) -> Tuple[str, int]:
    """Comparison edge (weighing wiring): 0 words, syntax only."""
    return (f"[weigh-link: compare {src.id}↔{tgt.id}, syntax only]", BUDGET_EDGE_COMPARE)


def extension_line(node, *, stem_override: Optional[str] = None) -> Tuple[str, int]:
    """RS26 extension: ~15 words, claim restated in full every time, no decay. An
    extended advocacy restates its own per-node stem (`stem_override`), so the
    discriminator that keeps it distinct at introduction persists on extension."""
    stem = stem_override if stem_override is not None else _NODE_STEM.get(node.kind, node.kind)
    return f"[extend {_tag(node)} — {stem}]", BUDGET_EXTENSION


def invis_marker(kind: str, reason: str) -> str:
    """RS27 marker for a judge-invisible edge. Consumes the judge-emitted reason
    verbatim; the renderer computes nothing (RS27b)."""
    return f"  ⟨judge-invisible: {kind} reason=\"{reason}\"⟩"


def incomplete_marker() -> str:
    """RS13c marker: component derived no register but the round has an Advocacy."""
    return "⟨incomplete: never connected to advocacy or framework⟩"
