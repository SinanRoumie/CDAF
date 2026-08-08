"""Checkpoint save/load for the CDAF actor-critic.

A checkpoint is the full reproducible state of one policy snapshot: the ActorCritic's
`state_dict` (which INCLUDES the shared encoder -- one instance, one set of weights),
the encoder's constructor hyperparameters (so a checkpoint can be re-instantiated
without the original code path), an optional optimizer state, and a free-form `meta`
blob (update index, config hash, ...). Torch's `save`/`load` handle the tensors.

The encoder hyperparameters are stored because the pointer/critic heads' widths are
derived from `encoder.d_model` / `encoder.graph_dim`; loading must reconstruct the SAME
architecture before `load_state_dict`, and re-reading them off a live object is not
possible from a cold checkpoint file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from policy import GraphEncoder, ActorCritic


@dataclass
class EncoderSpec:
    """The GraphEncoder constructor arguments needed to rebuild an identical architecture
    from a cold checkpoint. Kept explicit (not pickled objects) so a checkpoint stays
    loadable across refactors of the encoder's internals."""
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    d_ff: Optional[int] = None
    role_dim: int = 16
    owner_dim: int = 4
    speech_dim: int = 8
    slot_dim: int = 16
    d_global: int = 64

    def build(self) -> GraphEncoder:
        return GraphEncoder(**self.__dict__)


def new_actor_critic(spec: EncoderSpec = None, seed: int = None) -> ActorCritic:
    """Fresh actor-critic on a fresh (shared) encoder. `seed` makes initialization
    reproducible."""
    if seed is not None:
        torch.manual_seed(seed)
    spec = spec or EncoderSpec()
    return ActorCritic(spec.build())


def save_checkpoint(path: str, ac: ActorCritic, *, encoder_spec: EncoderSpec = None,
                    optimizer: torch.optim.Optimizer = None, meta: dict = None) -> None:
    """Serialize one policy snapshot to `path`. `encoder_spec` defaults to the encoder's
    own live hyperparameters so a checkpoint is self-describing."""
    enc = ac.encoder
    if encoder_spec is None:
        encoder_spec = EncoderSpec(
            d_model=enc.d_model, n_heads=enc.layers[0].attn.n_heads,
            n_layers=len(enc.layers), d_ff=enc.layers[0].ff[0].out_features,
            role_dim=enc.role_emb.embedding_dim, owner_dim=enc.owner_emb.embedding_dim,
            speech_dim=enc.speech_emb.embedding_dim, slot_dim=enc.slot_emb.embedding_dim,
            d_global=enc.d_global)
    torch.save({
        "encoder_spec": encoder_spec.__dict__,
        "actor_critic_state": ac.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "meta": meta or {},
    }, path)


def load_checkpoint(path: str, *, optimizer: torch.optim.Optimizer = None,
                    map_location="cpu"):
    """Rebuild an ActorCritic (and optionally restore an optimizer) from `path`.
    Returns (actor_critic, meta)."""
    blob = torch.load(path, map_location=map_location, weights_only=False)
    spec = EncoderSpec(**blob["encoder_spec"])
    ac = ActorCritic(spec.build())
    ac.load_state_dict(blob["actor_critic_state"])
    if optimizer is not None and blob.get("optimizer_state") is not None:
        optimizer.load_state_dict(blob["optimizer_state"])
    return ac, blob.get("meta", {})
