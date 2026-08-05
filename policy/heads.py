"""Factored action heads + critic head, consuming the SHARED `GraphEncoder` output
(encoder_spec §Interface to the action heads, §Parameter sharing).

One `GraphEncoder` forward pass produces per-node embeddings and a pooled graph
vector; `ActorCritic` feeds BOTH the actor heads and the critic from that single pass
-- the encoder is never duplicated or forked.

Action selection is FACTORED and MASKED (action_schema_spec vocabulary):

    action type  ->  node argument(s) via a POINTER over node embeddings  ->  the
    remaining structural parameter (role / edge_type / favors)

Each stage's distribution is masked by `policy.masking.LegalActionMask` (which
delegates every legality decision to the env's generator) BEFORE sampling, so illegal
choices receive exactly zero probability -- masking, not post-hoc filtering.

  * Target selection is a pointer net (additive-attention scoring against the encoder's
    per-node embeddings), never a fixed-size index space -- so it handles any node
    count and generalizes across rounds where the same structure carries different ids.
  * `introduce`'s target may be NEW (no node exists yet): the introduce-target pointer
    scores over [node embeddings || one learned NEW sentinel], so NEW is a pointer
    outcome, not a separate index space.
  * The policy is CONTENT-BLIND: it emits no `content`/`justification`. Assembled
    actions carry an empty-string placeholder the judge never reads.

The critic head reads ONLY the pooled graph embedding (sum||mean||global), never the
per-node embeddings -- `value()` takes just the pooled vector.

SCOPE: heads + masking only. No PPO loss, no rollout/training loop. `sample_action`
returns a legal env action (and per-stage diagnostics); log-prob/entropy accounting
for the surrogate loss belongs to the training milestone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from env import NEW
from env.actions import Introduce, Extend, Concede, Weigh, Connect, EndSpeech
from .encoder import GraphEncoder, EncoderOutput
from .masking import (
    LegalActionMask, ACTION_TYPES, ACTION_ROLE_ORDER, EDGE_TYPE_ORDER,
)

_NEG_INF = float("-inf")


def masked_log_probs(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """log-softmax over `logits` with masked entries forced to -inf (prob 0). `mask` is
    a bool tensor, True == legal. Requires at least one legal entry."""
    filled = logits.masked_fill(~mask, _NEG_INF)
    return torch.log_softmax(filled, dim=-1)


def masked_probs(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Probabilities with illegal entries at exactly 0.0."""
    return masked_log_probs(logits, mask).exp()


def _sample_index(logits: torch.Tensor, mask_np, generator: Optional[torch.Generator]) -> int:
    """Sample one index from a masked categorical. Returns a Python int."""
    mask = torch.as_tensor(mask_np, dtype=torch.bool, device=logits.device)
    probs = masked_probs(logits, mask)
    if generator is not None:
        return int(torch.multinomial(probs, 1, generator=generator).item())
    return int(torch.multinomial(probs, 1).item())


class PointerHead(nn.Module):
    """Additive-attention pointer: score each item embedding against a query vector.
    Returns one logit per item (no fixed output dimension -- works for any item count)."""

    def __init__(self, d_item: int, d_query: int, d_hidden: Optional[int] = None):
        super().__init__()
        d_hidden = d_hidden or d_item
        self.item_proj = nn.Linear(d_item, d_hidden, bias=False)
        self.query_proj = nn.Linear(d_query, d_hidden)
        self.score = nn.Linear(d_hidden, 1, bias=False)

    def forward(self, items: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        # items: (M, d_item); query: (d_query,) -> logits (M,)
        h = torch.tanh(self.item_proj(items) + self.query_proj(query))
        return self.score(h).squeeze(-1)


@dataclass
class SampledAction:
    """A sampled action plus per-stage diagnostics for inspection/tests. `action` is a
    concrete env action object, guaranteed legal by construction under the masks."""
    action: object
    action_type: str
    stages: dict            # stage name -> {"probs": Tensor, "mask": ndarray, "choice": int}


class ActorCritic(nn.Module):
    """Shared-encoder actor + critic. Holds ONE `GraphEncoder`; `evaluate()` runs it
    once and both the value head and the action heads read that single output."""

    # index of each action type in ACTION_TYPES, for readability
    _T = {name: i for i, name in enumerate(ACTION_TYPES)}

    def __init__(self, encoder: GraphEncoder, d_query: int = 128, hidden: int = 128):
        super().__init__()
        self.encoder = encoder                      # SHARED -- not duplicated/forked
        d = encoder.d_model
        g = encoder.graph_dim
        self.d_query = d_query

        # Critic: scalar value from the POOLED graph embedding only.
        self.value_head = nn.Sequential(
            nn.Linear(g, hidden), nn.GELU(), nn.Linear(hidden, 1))

        # Actor -- categorical heads over fixed vocabularies.
        self.type_head = nn.Sequential(nn.Linear(g, hidden), nn.GELU(),
                                       nn.Linear(hidden, len(ACTION_TYPES)))
        self.role_head = nn.Sequential(nn.Linear(g, hidden), nn.GELU(),
                                       nn.Linear(hidden, len(ACTION_ROLE_ORDER)))
        # edge_type heads condition on the involved node embedding(s) too.
        self.introduce_edge_head = nn.Sequential(
            nn.Linear(g + d, hidden), nn.GELU(), nn.Linear(hidden, len(EDGE_TYPE_ORDER)))
        self.connect_edge_head = nn.Sequential(
            nn.Linear(g + 2 * d, hidden), nn.GELU(), nn.Linear(hidden, len(EDGE_TYPE_ORDER)))

        # Pointer + one learned query projection per node-selection slot.
        self.pointer = PointerHead(d, d_query)
        self._slots = ("introduce_target", "extend", "concede",
                       "weigh_a", "weigh_b", "connect_source", "connect_target", "favors")
        self.query_proj = nn.ModuleDict({s: nn.Linear(g, d_query) for s in self._slots})
        # Learned NEW-node sentinel: the extra pointer item for `introduce`'s target.
        self.new_token = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.new_token, std=0.02)

    # --- shared forward -------------------------------------------------------
    def evaluate(self, observation: dict) -> EncoderOutput:
        """Run the shared encoder ONCE. Both heads read the returned output."""
        return self.encoder.encode(observation)

    # --- critic ---------------------------------------------------------------
    def value(self, graph_embedding: torch.Tensor) -> torch.Tensor:
        """Scalar state value from the pooled graph embedding ALONE (per-node embeddings
        are deliberately not an input -- encoder_spec §Pooling / critic)."""
        return self.value_head(graph_embedding).squeeze(-1)

    # --- actor ----------------------------------------------------------------
    def _q(self, slot: str, g: torch.Tensor) -> torch.Tensor:
        return self.query_proj[slot](g)

    def sample_action(self, enc_out: EncoderOutput, state,
                      generator: Optional[torch.Generator] = None) -> SampledAction:
        """Factored, masked sampling of one legal action from a single encoder output.
        `state` is the live `RoundState`, used ONLY to build masks via the env generator
        (`LegalActionMask`). Returns a `SampledAction` whose `.action` is a concrete,
        structurally-legal env action."""
        node_emb = enc_out.node_embeddings          # (N, d)
        g = enc_out.graph_embedding                  # (graph_dim,)
        node_ids = enc_out.node_ids
        mask = LegalActionMask(state)
        assert node_ids == mask.node_ids, "encoder node order must match mask node order"

        stages: dict = {}

        def record(name, logits, mask_np):
            choice = _sample_index(logits, mask_np, generator)
            stages[name] = {"probs": masked_probs(logits, torch.as_tensor(
                mask_np, dtype=torch.bool, device=logits.device)),
                "mask": mask_np, "choice": choice}
            return choice

        # (1) action type
        t = record("type", self.type_head(g), mask.type_mask())
        type_name = ACTION_TYPES[t]

        if type_name == "end_speech":
            return SampledAction(EndSpeech(), type_name, stages)

        if type_name in ("extend", "concede"):
            slot = type_name
            tmask = getattr(mask, f"{slot}_target_mask")()
            i = record(f"{slot}_target", self.pointer(node_emb, self._q(slot, g)), tmask)
            action = Extend(node_ids[i]) if type_name == "extend" else Concede(node_ids[i])
            return SampledAction(action, type_name, stages)

        if type_name == "introduce":
            return self._sample_introduce(node_emb, g, node_ids, mask, stages, record)

        if type_name == "weigh":
            return self._sample_weigh(node_emb, g, node_ids, mask, stages, record)

        # connect
        return self._sample_connect(node_emb, g, node_ids, mask, stages, record)

    # --- per-type assembly ----------------------------------------------------
    def _sample_introduce(self, node_emb, g, node_ids, mask, stages, record) -> SampledAction:
        # target pointer over [nodes || NEW sentinel]
        items = torch.cat([node_emb, self.new_token.unsqueeze(0)], dim=0)   # (N+1, d)
        tmask = mask.introduce_target_mask()
        i = record("introduce_target", self.pointer(items, self._q("introduce_target", g)), tmask)
        is_new = (i == len(node_ids))
        target = NEW if is_new else node_ids[i]

        r = record("role", self.role_head(g), mask.introduce_role_mask(target))
        role = ACTION_ROLE_ORDER[r]

        if is_new:
            return SampledAction(Introduce("", role, NEW, None), "introduce", stages)

        ctx = torch.cat([g, node_emb[i]], dim=0)
        e = record("introduce_edge", self.introduce_edge_head(ctx), mask.introduce_edge_mask(target))
        return SampledAction(Introduce("", role, target, EDGE_TYPE_ORDER[e]), "introduce", stages)

    def _sample_weigh(self, node_emb, g, node_ids, mask, stages, record) -> SampledAction:
        ia = record("weigh_a", self.pointer(node_emb, self._q("weigh_a", g)), mask.weigh_a_mask())
        a_id = node_ids[ia]
        ib = record("weigh_b", self.pointer(node_emb, self._q("weigh_b", g)),
                    mask.weigh_b_mask(a_id))
        b_id = node_ids[ib]
        # favors is a pointer at one of the two chosen node embeddings.
        pair = torch.stack([node_emb[ia], node_emb[ib]], dim=0)             # (2, d)
        f = record("favors", self.pointer(pair, self._q("favors", g)), mask.favors_mask(a_id, b_id))
        favors_id = a_id if f == 0 else b_id
        return SampledAction(Weigh(a_id, b_id, favors_id, ""), "weigh", stages)

    def _sample_connect(self, node_emb, g, node_ids, mask, stages, record) -> SampledAction:
        isrc = record("connect_source", self.pointer(node_emb, self._q("connect_source", g)),
                      mask.connect_source_mask())
        s_id = node_ids[isrc]
        itgt = record("connect_target", self.pointer(node_emb, self._q("connect_target", g)),
                      mask.connect_target_mask(s_id))
        t_id = node_ids[itgt]
        ctx = torch.cat([g, node_emb[isrc], node_emb[itgt]], dim=0)
        e = record("connect_edge", self.connect_edge_head(ctx), mask.connect_edge_mask(s_id, t_id))
        return SampledAction(Connect(s_id, t_id, EDGE_TYPE_ORDER[e]), "connect", stages)
