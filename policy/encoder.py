"""The CDAF graph encoder (Phase 4, `encoder_spec.md`).

Graph-in, embeddings-out. Maps one environment observation to:
  * per-node embeddings   (N, d_model)  -- the pointer/target head reads these later
  * a pooled graph vector  (2*d_model + d_global)  -- the critic reads this later

This module builds the ENCODER ONLY. Action heads and the critic head are a later
milestone and are not here; the SAME encoder instance is shared between actor and
critic (encoder_spec §Parameter sharing), so it is constructed once and its output
fed to two heads.

Architecture (encoder_spec §Architecture):
  * FULL self-attention across all nodes -- no message-passing hop limit, so chain
    length and graph size never truncate context.
  * Structure enters ONLY as a LEARNED ADDITIVE BIAS on the attention score, keyed
    by the directed relation between each ordered pair (per head). No reverse-edge
    relations: attention is already all-pairs; the relation's asymmetry is preserved
    by keying the bias on the ordered pair (`policy.features._relation_matrix`).
  * CONTENT-BLIND: input is the content-free observation via `policy.features`; the
    encoder never receives text.
  * Pooling is sum || mean || global (encoder_spec §Pooling).

Design choices NOT pinned by the spec (spec §Open items #1 leaves layer count/width
open) -- made here as implementation judgment and documented on `GraphEncoder`:
d_model=128, 4 heads, 3 pre-norm attention layers, FFN width 2*d_model, and the small
categorical embedding widths below. No dropout (deterministic; RL adds its own
exploration noise). These are the sweep knobs, isolated in the constructor.

Batching: a single graph per forward() (variable N). PPO-time batching over many
graphs of differing sizes (pad+mask or ragged) belongs with the training loop and is
deliberately out of this milestone's scope.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import (
    GraphFeatures, extract_features,
    ROLE_VOCAB, RELATION_VOCAB, CONT_DIM, GLOBAL_CONT_DIM,
    DEGREE_EDGE_TYPES,
)
from model import SPEECH_ORDER

_N_ROLES = len(ROLE_VOCAB)
_N_RELATIONS = len(RELATION_VOCAB)
_N_SLOT_EMB = len(SPEECH_ORDER) + 1     # +1 for the terminal slot (features._TERMINAL_SLOT)


@dataclass
class EncoderOutput:
    """Encoder result for one graph.
      node_embeddings: (N, d_model) -- per-node, permutation-EQUIVARIANT.
      graph_embedding: (2*d_model + d_global,) -- pooled, permutation-INVARIANT.
      node_ids: the id per row of node_embeddings (observation order)."""
    node_embeddings: torch.Tensor
    graph_embedding: torch.Tensor
    node_ids: list


class _RelationBiasedAttention(nn.Module):
    """Multi-head self-attention over N nodes with a learned additive bias per
    (relation, head) added to every attention score. Single graph (no batch dim)."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must divide evenly among heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        # One learned scalar bias per (relation, head). Looked up by the (N, N)
        # relation matrix -> (N, N, n_heads) and added to the pre-softmax scores.
        self.relation_bias = nn.Embedding(_N_RELATIONS, n_heads)

    def forward(self, x: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
        # x: (N, d_model); rel: (N, N) long.
        n = x.shape[0]
        qkv = self.qkv(x).view(n, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=1)                 # each (N, n_heads, d_head)
        q = q.transpose(0, 1)                       # (n_heads, N, d_head)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        bias = self.relation_bias(rel).permute(2, 0, 1)   # (n_heads, N, N)
        attn = F.softmax(scores + bias, dim=-1)
        ctx = torch.matmul(attn, v)                 # (n_heads, N, d_head)
        ctx = ctx.transpose(0, 1).reshape(n, self.n_heads * self.d_head)
        return self.out(ctx)


class _EncoderLayer(nn.Module):
    """Pre-norm transformer block: relation-biased self-attention + FFN, both residual."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = _RelationBiasedAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rel)
        x = x + self.ff(self.norm2(x))
        return x


class GraphEncoder(nn.Module):
    """Shared actor/critic encoder. Call `encode(observation)` for the high-level
    path (feature extraction + forward), or `forward(features)` with a precomputed
    `GraphFeatures`.

    Hyperparameters (encoder_spec §Open items #1: no principled a-priori value --
    these are the sweep knobs):
      d_model   node/hidden width (default 128)
      n_heads   attention heads (default 4)
      n_layers  stacked attention blocks (default 3)
      d_ff      FFN inner width (default 2*d_model)
      role/owner/speech/slot embedding widths -- small categorical embeddings.
    """

    def __init__(self, d_model: int = 128, n_heads: int = 4, n_layers: int = 3,
                 d_ff: int | None = None, role_dim: int = 16, owner_dim: int = 4,
                 speech_dim: int = 8, slot_dim: int = 16, d_global: int = 64):
        super().__init__()
        d_ff = d_ff or 2 * d_model
        self.d_model = d_model
        self.d_global = d_global
        # Pooled graph vector = sum || mean || global (encoder_spec §Pooling).
        self.graph_dim = 2 * d_model + d_global

        # --- node input embeddings (all categorical features) ---
        self.role_emb = nn.Embedding(_N_ROLES, role_dim)
        self.owner_emb = nn.Embedding(2, owner_dim)             # 0 self / 1 opponent
        self.speech_emb = nn.Embedding(len(SPEECH_ORDER), speech_dim)
        node_in = role_dim + owner_dim + speech_dim + CONT_DIM
        self.node_in = nn.Linear(node_in, d_model)

        self.layers = nn.ModuleList(
            _EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers))
        self.final_norm = nn.LayerNorm(d_model)

        # --- global context (encoder_spec §Global context): slot + remaining budget,
        # held separately and concatenated at pooling (never replicated per node). ---
        self.slot_emb = nn.Embedding(_N_SLOT_EMB, slot_dim)
        self.global_in = nn.Linear(slot_dim + GLOBAL_CONT_DIM, d_global)

    # --- convenience path ------------------------------------------------------
    def encode(self, observation: dict) -> EncoderOutput:
        """Extract content-blind features from an env observation and encode."""
        return self.forward(extract_features(observation))

    # --- core ------------------------------------------------------------------
    def forward(self, features: GraphFeatures) -> EncoderOutput:
        dev = self.node_in.weight.device
        n = features.n_nodes

        global_vec = self._global(features, dev)                # (d_global,)

        if n == 0:
            # Empty graph (e.g. the reset() observation): no nodes to attend over.
            # sum/mean pool to zeros; the graph vector is still well-defined via the
            # global context alone.
            node_emb = torch.zeros((0, self.d_model), device=dev)
            pooled = torch.zeros(2 * self.d_model, device=dev)
            graph = torch.cat([pooled, global_vec], dim=0)
            return EncoderOutput(node_emb, graph, list(features.node_ids))

        role = torch.from_numpy(features.role_idx).to(dev)
        owner = torch.from_numpy(features.owner_idx).to(dev)
        speech = torch.from_numpy(features.speech_idx).to(dev)
        cont = torch.from_numpy(features.cont).to(dev)
        rel = torch.from_numpy(features.rel).to(dev)

        x = torch.cat([
            self.role_emb(role), self.owner_emb(owner), self.speech_emb(speech), cont,
        ], dim=-1)
        x = self.node_in(x)

        for layer in self.layers:
            x = layer(x, rel)
        node_emb = self.final_norm(x)                           # (N, d_model)

        pooled = torch.cat([node_emb.sum(dim=0), node_emb.mean(dim=0)], dim=0)
        graph = torch.cat([pooled, global_vec], dim=0)
        return EncoderOutput(node_emb, graph, list(features.node_ids))

    def _global(self, features: GraphFeatures, dev) -> torch.Tensor:
        slot = torch.tensor(features.slot_idx, dtype=torch.long, device=dev)
        gcont = torch.from_numpy(features.global_cont).to(dev)
        return self.global_in(torch.cat([self.slot_emb(slot), gcont], dim=0))
