"""Environment-side machinery for the CDAF RL build.

The `judge` package is a PURE scoring function: given a well-typed Round it emits
a verdict + trace and never rejects a graph. Episode VALIDITY is a separate
concern and lives here, in the environment, so the judge stays pure and the RL
loop owns which graphs a training agent is allowed to generate episodes over.

`validator` holds the ingest/episode-init fences that refuse graphs which would
reach code paths the judge has not been audited on (out-of-scope / malformed
corners a self-play agent would otherwise exploit).
"""

from .validator import RoundValidity, validate_round, is_valid

__all__ = ["RoundValidity", "validate_round", "is_valid"]
