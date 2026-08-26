"""Magnitude channel and delta (§3.3), plus effective polarity (§3.2).

The MULTIPLICATION channel: propagation ALONG a chain. This module NEVER runs
DF-QuAD along the chain -- accrual at a node is dfquad.py, and the two are
different operations that never merge (§3, §0). Here we only multiply
already-resolved per-node strengths and signs.

    mag(a)  = prod_i sigma_i        # product of every node's surviving strength
    delta_a = sign(a) * mag(a)      # sign from qpn.sign_product over effective polarities

Under tau = 1.0 a clean uncontested chain holds at mag 1.0 regardless of length
(no length-fragility). A single near-zero node collapses the product -- one dead
link kills the chain, with no terminal-defense primitive required. A '?' sign
yields a '?' delta: no offense established, drains to presumption.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

from .config import POLARITY_THRESHOLD
from .qpn import UNRESOLVED, Sign


def magnitude(sigmas: Iterable[float]) -> float:
    """Running product of a chain's node strengths (uniqueness * link strengths
    * terminal severity). Empty chain = 1.0."""
    mag = 1.0
    for sigma in sigmas:
        mag *= sigma
    return mag


def delta(sign: Sign, mag: float):
    """delta = sign * mag. An UNRESOLVED ('?') sign yields an UNRESOLVED delta --
    the chain cannot establish offense (§3.3)."""
    if sign == UNRESOLVED:
        return UNRESOLVED
    return sign * mag


def effective_polarity(
    base_polarity: Sign,
    sigma: float,
    preference: Optional[Sign] = None,
) -> Tuple[Sign, str]:
    """Resolve a contested link's effective polarity (§3.2). Returns
    (eff_polarity, via) where `via` is "preference" or "dfquad".

    The offensive attack (a competing-polarity claim on symmetric bases) has
    already entered the link's DF-QuAD accrual (dfquad.accrue); `sigma` is the
    resulting surviving magnitude. Resolution order mirrors the "won preference
    overrides the raw number, absence falls back to the number" pattern:

      1. An established directional preference -- a won weighing/framework claim
         about which way the link cuts -- is honored first (via "preference").
      2. Otherwise fall back to DF-QuAD against the 0.5 threshold (via "dfquad"):
         surviving magnitude >= 0.5 keeps the base polarity, < 0.5 flips it. A
         genuinely-unresolved base ('?') stays '?'.
    """
    if preference is not None and preference != UNRESOLVED:
        return preference, "preference"
    if base_polarity == UNRESOLVED:
        return UNRESOLVED, "dfquad"
    if sigma >= POLARITY_THRESHOLD:
        return base_polarity, "dfquad"      # keeps polarity
    return -base_polarity, "dfquad"         # flips polarity
