"""J2 gate -- unit tests for the two channels (§3).

Reproduces the spec's stated behavior exactly:
  - tau=1.0 conceded lone defender -> target sigma 0 (terminal);
  - partial defense -> mitigation;
  - clean N-link chain holds at mag 1.0 for any N;
  - one zero node collapses the product;
  - polarity flips below the 0.5 threshold, holds at/above.

DF-QuAD (accrual) and multiplication (propagation) are exercised through
separate modules, never merged.
"""

import math

from judge import config, dfquad, qpn, chain


# --- DF-QuAD node accrual (§3.1) ----------------------------------------------

def test_unattacked_node_holds_at_tau():
    assert dfquad.accrue(1.0, [], []) == 1.0
    assert dfquad.accrue(0.5, [], []) == 0.5


def test_conceded_lone_defender_drives_target_to_zero():
    # tau=1.0, single conceded defender strength 1.0, no supporters -> 0 (terminal)
    assert dfquad.accrue(1.0, [1.0], []) == 0.0


def test_contested_defender_mitigates():
    # defender contested down to ~0.5 pulls the target only partway
    sigma = dfquad.accrue(1.0, [0.5], [])
    assert math.isclose(sigma, 0.5)
    assert 0.0 < sigma < 1.0


def test_defender_at_0_6_pulls_partway():
    # spec's worked figure: a defender surviving at 0.6 -> target at 0.4
    assert math.isclose(dfquad.accrue(1.0, [0.6], []), 0.4)


def test_support_restores_after_attack():
    attacked = dfquad.accrue(1.0, [0.5], [])
    restored = dfquad.accrue(1.0, [0.5], [0.5])
    assert restored > attacked          # support raises back toward the cap
    assert restored <= 1.0              # but never exceeds tau = 1.0


def test_support_only_cannot_exceed_cap_at_tau_one():
    assert dfquad.accrue(1.0, [], [1.0]) == 1.0


def test_inert_attack_contributes_nothing():
    base = dfquad.accrue(1.0, [], [])
    a = dfquad.effective_strength(1.0, inert=True)
    assert a == 0.0
    assert dfquad.accrue(1.0, [a], []) == base == 1.0


# --- QPN sign channel (§3.3) --------------------------------------------------

def test_sign_product_multiplies_polarities():
    assert qpn.sign_product([1, 1, 1]) == 1
    assert qpn.sign_product([1, -1, 1]) == -1
    assert qpn.sign_product([-1, -1]) == 1


def test_sign_product_empty_is_positive():
    assert qpn.sign_product([]) == 1


def test_sign_product_unresolved_propagates():
    assert qpn.sign_product([1, qpn.UNRESOLVED, -1]) == qpn.UNRESOLVED
    assert qpn.sign_product([qpn.UNRESOLVED]) == qpn.UNRESOLVED


# --- magnitude channel + delta (§3.3) -----------------------------------------

def test_clean_chain_holds_at_one_regardless_of_length():
    for n in (1, 2, 3, 7, 20):
        assert chain.magnitude([1.0] * n) == 1.0


def test_one_near_zero_node_collapses_chain():
    assert math.isclose(chain.magnitude([1.0, 1e-9, 1.0]), 1e-9)
    assert chain.magnitude([1.0, 0.0, 1.0]) == 0.0


def test_delta_is_sign_times_mag():
    assert chain.delta(1, 1.0) == 1.0
    assert chain.delta(-1, 0.5) == -0.5


def test_delta_unresolved_sign_is_unresolved():
    assert chain.delta(qpn.UNRESOLVED, 1.0) == qpn.UNRESOLVED


def test_clean_chain_delta_is_plus_one():
    # oracle round 1 building block: sign + , mag 1.0 -> delta 1.0
    sign = qpn.sign_product([1, 1, 1])
    mag = chain.magnitude([1.0, 1.0, 1.0])
    assert chain.delta(sign, mag) == 1.0


# --- effective polarity (§3.2) ------------------------------------------------

def test_polarity_holds_at_or_above_threshold():
    assert config.POLARITY_THRESHOLD == 0.5
    assert chain.effective_polarity(1, 0.5) == (1, "dfquad")     # exactly at threshold keeps
    assert chain.effective_polarity(1, 0.9) == (1, "dfquad")
    assert chain.effective_polarity(-1, 0.75) == (-1, "dfquad")


def test_polarity_flips_below_threshold():
    assert chain.effective_polarity(1, 0.49) == (-1, "dfquad")
    assert chain.effective_polarity(1, 0.0) == (-1, "dfquad")
    assert chain.effective_polarity(-1, 0.1) == (1, "dfquad")


def test_preference_overrides_dfquad():
    # an established directional preference is honored over the raw magnitude
    assert chain.effective_polarity(1, 0.1, preference=1) == (1, "preference")
    assert chain.effective_polarity(1, 0.9, preference=-1) == (-1, "preference")


def test_unresolved_base_stays_unresolved():
    assert chain.effective_polarity(qpn.UNRESOLVED, 0.9) == (qpn.UNRESOLVED, "dfquad")


# --- the two channels never merge (§3 / §0) -----------------------------------

def test_contested_link_flow_uses_both_channels_separately():
    # 1) DF-QuAD resolves the contested link's surviving magnitude (accrual)...
    sigma = dfquad.accrue(1.0, [0.6], [])           # attacker at 0.6 -> sigma 0.4
    assert math.isclose(sigma, 0.4)
    # 2) ...then the magnitude is read against 0.5 for polarity (propagation side)
    eff, via = chain.effective_polarity(1, sigma)
    assert (eff, via) == (-1, "dfquad")             # 0.4 < 0.5 -> flips
