"""Self-play checkpoint pool -- Approach A (`rl_training_spec.md` §Self-play, RL plan
notes): opponent selection by a FIXED-INTERVAL checkpoint pool.

Snapshot the learner every N updates into a bounded pool; sample opponents from it.
Pure self-play is rejected (debate has rock-paper-scissors structure -- a policy training
only against its current self can cycle at a 50% local win rate indistinguishable from
equilibrium); the pool forces the frontier to stay competent against everything it
previously beat.

WHAT IS SEMANTIC HERE (left for the human, spec §Adjustment protocol):
  * pool COMPOSITION -- the cap and the recent/anchor split (spec starting value: cap 20
    = most recent 15 + 5 early anchors). Changes WHAT the policy must stay strong against.
  * SAMPLING RATIO -- self vs pool (spec starting value: 50/50). Changes how much of the
    gradient comes from the moving frontier vs the fixed history.
Both are read from `SemanticsConfig` via `require(...)`, so an unset value fails loudly.

WHAT IS MECHANICAL (in `TuningConfig`): the snapshot CADENCE (`snapshot_interval`) -- WHEN
to snapshot, not what the pool is or how it is drawn from.

PROPOSED update/sampling policy (surfaced for confirmation, since it has semantic weight):
  * Update rule (bounded pool = recent R + anchor A): always keep the A earliest snapshots
    as anchors; among the rest keep the R most recent; drop the oldest non-anchor when
    over cap. Rationale: anchors stop the pool diluting toward only-recent checkpoints
    (which can co-adapt and cycle together), while the recent window keeps opponents near
    the current frontier.
  * Opponent sampling: with prob `self_play_ratio` play the CURRENT policy (self); else
    draw a pool member UNIFORMLY. Uniform (not prioritized/loss-weighted) on purpose --
    prioritized sampling is a refinement that adds per-opponent win-rate tracking and can
    overfit to a single hard opponent (spec). Anchors and recents share the uniform draw.
These are proposals; the numbers (ratio, cap, R, A) stay UNSET pending a ruling.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from policy import ActorCritic
from .config import SemanticsConfig
from .checkpoint import save_checkpoint, load_checkpoint, EncoderSpec


@dataclass
class PoolEntry:
    update: int             # the update index at which this snapshot was taken
    path: str               # checkpoint file on disk
    is_anchor: bool = False


@dataclass
class CheckpointPool:
    """A bounded pool of policy snapshots on disk. Composition (cap / recent / anchors)
    and the self-play sampling ratio are SEMANTIC and read from `semantics` at use time,
    so this object holds no default for them -- an unset value raises when consulted.

    `directory` is where snapshot files are written. `entries` is the live pool, oldest
    first."""
    directory: str
    semantics: SemanticsConfig
    entries: List[PoolEntry] = field(default_factory=list)
    _snap_counter: int = 0
    # In-memory cache of DESERIALIZED opponents, keyed by checkpoint path. Opponents are
    # read-only (eval, no grad), so one loaded policy is safely reused across every episode
    # that draws it -- instead of a fresh 4 MB `torch.load` per episode. This removes both
    # the per-episode disk cost and the exposure to transient `torch.load` failures that
    # crashed validation-v1 at update 10. Evicted entries are dropped from the cache too.
    _cache: dict = field(default_factory=dict)

    def __post_init__(self):
        os.makedirs(self.directory, exist_ok=True)

    # --- snapshotting ---------------------------------------------------------
    def snapshot(self, ac: ActorCritic, update: int, *, encoder_spec: EncoderSpec = None) -> PoolEntry:
        """Write `ac` into the pool as of `update`, then enforce the composition bound.
        The first `pool_anchors` snapshots ever taken are marked anchors (the 'early
        anchors' of the spec's composition)."""
        anchors = int(self.semantics.require("pool_anchors"))
        path = os.path.join(self.directory, f"pool_update{update:06d}.pt")
        save_checkpoint(path, ac, encoder_spec=encoder_spec,
                        meta={"update": update, "pool_snapshot_index": self._snap_counter})
        entry = PoolEntry(update=update, path=path, is_anchor=(self._snap_counter < anchors))
        self._snap_counter += 1
        self.entries.append(entry)
        self._enforce_cap()
        return entry

    def _enforce_cap(self) -> None:
        """Keep at most `pool_cap` entries: all anchors, plus the `pool_recent` most-recent
        non-anchors; drop the oldest non-anchor beyond that. Deletes evicted files."""
        cap = int(self.semantics.require("pool_cap"))
        recent = int(self.semantics.require("pool_recent"))
        anchors = [e for e in self.entries if e.is_anchor]
        non_anchors = [e for e in self.entries if not e.is_anchor]
        keep_recent = non_anchors[-recent:] if recent > 0 else []
        keep = set(id(e) for e in anchors) | set(id(e) for e in keep_recent)
        # Enforce the hard cap too: if anchors + recent still exceed cap, drop oldest
        # non-anchors first (anchors are load-bearing and kept).
        kept = [e for e in self.entries if id(e) in keep]
        if len(kept) > cap:
            kept_non_anchor = [e for e in kept if not e.is_anchor]
            overflow = len(kept) - cap
            drop_extra = set(id(e) for e in kept_non_anchor[:overflow])
            kept = [e for e in kept if id(e) not in drop_extra]
        evicted = [e for e in self.entries if e not in kept]
        for e in evicted:
            if os.path.exists(e.path):
                os.remove(e.path)
            self._cache.pop(e.path, None)          # drop any cached deserialized opponent
        self.entries = kept

    # --- opponent sampling ----------------------------------------------------
    def sample_opponent_is_self(self, rng) -> bool:
        """True -> play the current policy (self); False -> draw from the pool. Uses the
        SEMANTIC `self_play_ratio`. If the pool is empty (early training) always returns
        True: there is nothing else to play yet."""
        ratio = float(self.semantics.require("self_play_ratio"))
        if not self.entries:
            return True
        return rng.random() < ratio

    def sample_opponent_entry(self, rng) -> Optional[PoolEntry]:
        """Uniformly draw one pool member, or None if the pool is empty."""
        if not self.entries:
            return None
        return self.entries[rng.randrange(len(self.entries))]

    def load_opponent(self, entry: PoolEntry, map_location="cpu") -> ActorCritic:
        """Return the deserialized opponent for `entry`, loading from disk ONCE and caching
        it in memory for reuse across episodes (opponents are read-only)."""
        cached = self._cache.get(entry.path)
        if cached is not None:
            return cached
        ac, _meta = load_checkpoint(entry.path, map_location=map_location)
        ac.eval()
        self._cache[entry.path] = ac
        return ac

    def sample_opponent(self, current: ActorCritic, rng, map_location="cpu") -> ActorCritic:
        """Resolve one opponent policy per the sampling policy: `current` for a self game,
        else a uniformly-drawn pool member loaded from disk."""
        if self.sample_opponent_is_self(rng):
            return current
        entry = self.sample_opponent_entry(rng)
        return current if entry is None else self.load_opponent(entry, map_location)

    def ladder(self) -> List[PoolEntry]:
        """All pool checkpoints, for the ladder-evaluation protocol (spec §Evaluation:
        every checkpoint played against every other detects cycling directly)."""
        return list(self.entries)
