"""CDAF judge -- the deterministic reward function over a model.Round.

Pure Python: imports only model/ and the standard library, never anything
app-side (no Dash / dash-cytoscape). Given the same Round it returns the same
result every time. See docs/judge_spec.md.

J1 ships the foundations: pinned constants (config) and the decision-trace
record shapes (trace). J2 adds the two strictly-separated channels: node accrual
(dfquad) and chain propagation -- sign (qpn) and magnitude/delta (chain). J3 adds
the six passes (passes) and the judge entry point (judge.judge).
"""

from . import config, trace, dfquad, qpn, chain, resolve, passes, rfd
from .judge import judge
from .config import JUDGE_VERSION

__all__ = ["config", "trace", "dfquad", "qpn", "chain", "resolve", "passes",
           "rfd", "judge", "JUDGE_VERSION"]
