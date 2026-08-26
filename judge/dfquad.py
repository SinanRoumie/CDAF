"""Node accrual -- DF-QuAD (§3.1).

ONE of the two strictly-separated channels. This module governs how strength
accrues AT A NODE from its already-resolved attackers and supporters. It never
multiplies along a chain -- that is the magnitude channel (chain.py). Keeping
accrual and propagation apart is the single most error-prone point in the judge
(§3, §0); they live in different files and never call into each other.

For a node with base `tau`, surviving attacker strengths {a_j} and supporter
strengths {s_k} (each already resolved, leaves first), DF-QuAD aggregates each
side multiplicatively and combines them through a piecewise influence function:

    no_attack  = prod_j (1 - a_j)     # 1.0 when unattacked, 0.0 when fully attacked
    no_support = prod_k (1 - s_k)     # 1.0 when unsupported, 0.0 when fully supported
    E = no_attack - no_support        # E in [-1, 1]; supports raise, attacks lower

    sigma = tau + (1 - tau) * E       if E >= 0   (support dominates -> raise toward 1)
    sigma = tau * (1 + E)             if E < 0    (attack dominates  -> lower toward 0)

With tau = 1.0 an unattacked node stays at 1.0 (supports cannot exceed the cap;
their role is to restore strength after attack). A fully conceded lone defender
(strength 1.0, no supporters) drives its target to 0 -- terminal. A defender
contested down to ~0.5 pulls the target only partway -- mitigation. The
terminal-vs-mitigatory distinction thus falls out of whether the defense
survived; it is never a declared category.

SPEC DISCREPANCY (deliberate): §3.1 prints the influence as
`prod_k(1 - s_k) - prod_j(1 - a_j)`, i.e. the two product terms transposed
relative to the form above. Taken literally that negates E and inverts the whole
model -- a full attacker would RAISE sigma and a full supporter would drive it to
0 -- contradicting the same line's "supports raise, attacks lower" comment, the
surrounding prose, and the J2 gate. We implement the stated behavior (standard
DF-QuAD): E = no_attack - no_support.
"""

from __future__ import annotations

from typing import Iterable

from .config import TAU


def influence(attackers: Iterable[float], supporters: Iterable[float]) -> float:
    """The signed influence E in [-1, 1]. >= 0 means support dominates (raise),
    < 0 means attack dominates (lower). Empty product = 1.0, so a node with
    neither attackers nor supporters has E = 0 and accrues to exactly tau."""
    no_attack = 1.0
    for a in attackers:
        no_attack *= (1.0 - a)
    no_support = 1.0
    for s in supporters:
        no_support *= (1.0 - s)
    return no_attack - no_support


def accrue(tau: float, attackers: Iterable[float], supporters: Iterable[float]) -> float:
    """Surviving strength sigma in [0, 1] for a node with base `tau`, given its
    surviving attacker/supporter strengths. The piecewise influence function of
    §3.1 -- this is the ONLY place node strength is computed."""
    E = influence(attackers, supporters)
    if E >= 0.0:
        return tau + (1.0 - tau) * E
    return tau * (1.0 + E)


def effective_strength(strength: float, inert: bool) -> float:
    """An inert attack/support contributes nothing (§3.4): it is treated as
    strength 0, whose (1 - 0) = 1 factor leaves the product -- and so sigma --
    unchanged. The judge does not reject an incoherent move (e.g. offense aimed
    at a pre-world uniqueness node, or a cross-channel attack with nothing to
    attenuate); it simply has no effect on any sigma."""
    return 0.0 if inert else strength
