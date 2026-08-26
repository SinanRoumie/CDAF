"""Warm-start data generation: invert static oracle fixtures into legal
`(observation, action)` sequences for imitation warm-start.

Implements `docs/warm_start_data_spec.md` exactly (rulings A-H). This package is the
conversion PIPELINE only -- fixture-graph in, `(observation, action)` sequence out. It
does not define the supervised objective / imitation loss (a later milestone), and it
does not import `8node1ac.json` (a separate task).

Public surface:
  convert(round, name)      -> ConversionResult   (the inversion + self-validation)
  oracle_verdict(round)     -> (winner, reason_class)
  ConversionResult, ConversionError
"""

from .convert import (
    convert, oracle_verdict, ConversionResult, ConversionError,
)

__all__ = [
    "convert", "oracle_verdict", "ConversionResult", "ConversionError",
]
