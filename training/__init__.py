"""CDAF RL training loop (Phase 5, `docs/rl_training_spec.md`).

The last piece before self-play can begin. Encoder (`policy/encoder.py`), factored
action heads + critic (`policy/heads.py`), legal-action masking (`policy/masking.py`),
and warm-start data conversion (`warmstart/convert.py`) are all built and validated;
this package is the loop that consumes them:

  config       the two-section (tuning / semantics) run config. SEMANTIC parameters
               are UNSET sentinels that fail loudly -- the human rules on them before
               any real run (rl_training_spec §Adjustment protocol / §Configuration).
  imitation    supervised warm-start (behavior cloning) against the factored heads,
               consuming warmstart/convert.py's (observation, action) pairs.
  rollout      self-play episode collection against the real env, using the existing
               legal-action masking throughout; GAE advantage/return estimation.
  ppo          the clipped PPO surrogate (per-stage log-probs/entropy summed across
               the factored heads), value loss, entropy bonus.
  pool         the fixed-interval self-play checkpoint pool (Approach A) + opponent
               sampling; checkpoint save/load.
  loop         the orchestrator that wires warm-start -> PPO self-play. REFUSES to
               start while any semantic hyperparameter is unset.

THIS MILESTONE IS "build and validate the loop", NOT "start a real run." Nothing here
kicks off training on its own; `training/smoke.py` runs a trivially-capped pass purely
to prove the wiring does not crash.
"""
