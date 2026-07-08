"""Reason for Decision (RFD) -- a templated, plain-English decision summary.

Pure and causally inert: `render` reads the (ballot, trace) the judge already
returns and synthesizes the records it already emitted into prose. It computes
NOTHING new and changes NO verdict -- it reads the decided state, it never
decides. No model, no passes, no judge re-entry; standard library only.

The sentences are fixed templates (no LLM), the way a speech panel renders a
speech: one short paragraph per facet of the decision.
"""

from __future__ import annotations

from typing import List, Optional

from .config import AFF, NEG


def _opposing(side: str) -> str:
    return NEG if side == AFF else AFF


# Human phrasings for the structural collapse reasons emitted on CHAIN records.
_COLLAPSE_PHRASE = {
    "extension_fail": "failed extension (a spine node was not carried through every one of its side's speeches)",
    "sign_flip": "was turned (its link flipped polarity to the opponent)",
    "defensive_kill": "was killed by conceded defense (its magnitude was driven to zero)",
    "unresolved_sign": "could not establish a direction (sign unresolved)",
}

_REASON_PHRASE = {
    "AFF offense": "the affirmative carries net offense",
    "NEG offense": "the negative carries net offense",
    "presumption": "the round is indeterminate, so it drains to presumption",
    "framework lock-out": "the affirmative's impacts were locked out of the winning framework",
    "AFF structural failure": "the affirmative failed a structural requirement",
}


def _kind(trace: list, kind: str) -> list:
    return [r for r in trace if getattr(r, "kind", None) == kind]


def _last(trace: list, kind: str):
    found = _kind(trace, kind)
    return found[-1] if found else None


def _node(node_id: Optional[str]) -> str:
    return node_id if node_id else "an unnamed node"


def render(ballot: str, trace: list) -> str:
    """Return the RFD paragraph for a judged round. `ballot` and `trace` are
    exactly what judge.judge returns."""
    return "\n".join(render_lines(ballot, trace))


def render_lines(ballot: str, trace: list) -> List[str]:
    """The RFD as a list of plain-English sentences (one logical clause each)."""
    lines: List[str] = []
    b = _last(trace, "BALLOT")

    # 1. Verdict + reason class
    if b is None:
        lines.append(f"Decision: {ballot} wins.")
        return lines
    reason = _REASON_PHRASE.get(b.reason_class, b.reason_class or "the default presumption")
    lines.append(f"Decision: {b.winner} wins because {reason}.")

    # 2. Net offense and its decomposition. A TURNED contributor (its link flipped
    # polarity, §3.5) is one contribution seen two ways -- it stops carrying its
    # introducing side's offense and starts carrying the opponent's at the same
    # magnitude. Render it as ONE coherent statement here, and suppress its
    # duplicate appearance in the collapsed-arguments list below (it did not
    # collapse -- it generated offense for the other side).
    chain_by_id = {c.chain_id: c for c in _kind(trace, "CHAIN")}
    lines.append(
        f"Net offense N = {b.N:+.3f} (affirmative Sum-delta = {b.aff_sum:+.3f}, "
        f"negative Sum-delta = {b.neg_sum:+.3f})."
    )
    contributors = [d for d in b.decomposition if d["contributed"]]
    turned_ids = {d["chain_id"] for d in contributors
                  if getattr(chain_by_id.get(d["chain_id"]), "collapse_reason", None) == "sign_flip"}
    if contributors:
        for d in contributors:
            cid, intro = d["chain_id"], d["side"]
            if cid in turned_ids:
                # owning side is carried on the CHAIN record (descriptive); fall
                # back to deriving it if an older trace lacks the field.
                owner = getattr(chain_by_id.get(cid), "owner", "") or _opposing(intro)
                mag = abs(d["delta"])
                lines.append(
                    f"  - The {intro} argument {cid} was turned: it no longer carries {intro} "
                    f"offense and now carries {owner} offense at magnitude {mag:.3f} "
                    f"(delta = {d['delta']:+.3f})."
                )
            else:
                lines.append(
                    f"  - The {intro} argument {cid} survives and contributes "
                    f"delta = {d['delta']:+.3f}."
                )
    else:
        lines.append("  - No argument survived to contribute offense.")

    # 3. Every chain that collapsed, with the reason and responsible node. A turned
    # contributor (narrated above) is NOT a collapse and is excluded here, so it
    # reads as one statement rather than appearing as both a collapse and a credit.
    collapsed = [c for c in _kind(trace, "CHAIN")
                 if c.collapse_reason and c.chain_id not in turned_ids]
    if collapsed:
        lines.append("Collapsed arguments:")
        for c in collapsed:
            phrase = _COLLAPSE_PHRASE.get(c.collapse_reason, c.collapse_reason)
            tail = f", responsible: {_node(c.responsible)}" if c.responsible else ""
            lines.append(f"  - The {c.side} argument {c.chain_id} {phrase}{tail}.")

    # 4. Every weighing that fired
    weighs = _kind(trace, "WEIGH")
    if weighs:
        lines.append("Weighing:")
        for w in weighs:
            if w.outcome == "resolved":
                pair = " vs ".join(w.pair) if w.pair else "its compared pair"
                won_by = "concession" if w.via == "conceded" else "its sub-clash"
                override = (" and overrode raw delta (it preferred the smaller-delta impact)"
                            if w.overrode else " (consistent with raw delta)")
                lines.append(
                    f"  - Weighing {w.weighing_id} compared {pair}, preferred "
                    f"{_node(w.preferred_node)}, won by {won_by}{override}."
                )
            else:
                lines.append(
                    f"  - Weighing {w.weighing_id} did not resolve (symmetric); "
                    f"no preference was established."
                )

    # 5. Each ballot directive
    bds = _kind(trace, "BD_VALIDATE")
    if bds:
        lines.append("Ballot directives:")
        for v in bds:
            side = f"{v.side} " if v.side else ""
            verdict = "validated" if v.result == "pass" else "failed"
            lines.append(f"  - The {side}ballot directive {v.bd_id} {verdict}: {v.reason}.")

    return lines
