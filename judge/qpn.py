"""Sign channel -- QPN propagation (§3.3).

The SIGN half of chain propagation, kept separate from the magnitude half
(chain.py) and from node accrual (dfquad.py). Along a serial chain the effective
polarities multiply; a single unresolved ('?') link makes the whole chain sign
unresolved:

    sign(a) = prod_i p_eff_i        # product of EFFECTIVE polarities; any '?' => '?'

A '?' sign means the chain cannot establish offense and drains to presumption.
Effective polarities are produced by chain.effective_polarity (§3.2); this module
only multiplies the resolved signs.
"""

from __future__ import annotations

from typing import Iterable, Union

# Sentinel for a genuinely-unresolved polarity / sign. A chain/link sign is
# +1, -1, or this sentinel.
UNRESOLVED = "?"

Sign = Union[int, str]


def sign_product(polarities: Iterable[Sign]) -> Sign:
    """QPN product of effective polarities. Returns +1 / -1, or UNRESOLVED if
    any input link is unresolved. An empty chain yields +1 (empty product)."""
    sign: Sign = 1
    for p in polarities:
        if p == UNRESOLVED:
            return UNRESOLVED
        sign *= p
    return sign
