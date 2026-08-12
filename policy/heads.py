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

SCOPE: heads + masking. `sample_action` returns a legal env action (and per-stage
diagnostics). The training milestone's log-prob/entropy accounting is now also here
-- `sample_with_log_prob` (rollout) and `evaluate_action` (teacher-forced imitation +
PPO ratio) reuse the SAME `_walk` `sample_action` uses, so the factorization has one
source of truth. The PPO/BC losses themselves live in `training/`, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from env import NEW
from env.actions import Introduce, Extend, Concede, Weigh, Connect, EndSpeech
from .encoder import GraphEncoder, EncoderOutput
from .curriculum import CurriculumLegalActionMask
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


def _stage_log_prob_entropy(logits: torch.Tensor, mask_np, choice: int):
    """One factored stage's (log-prob of `choice`, entropy) under the legal-action mask.
    Entropy sums ONLY over legal entries -- masked entries hold exactly 0 probability and
    -inf log-prob, whose 0 * -inf product is NaN, so they are excluded rather than added
    as 0 (mathematically identical, numerically safe)."""
    m = torch.as_tensor(mask_np, dtype=torch.bool, device=logits.device)
    logp = masked_log_probs(logits, m)          # (K,), masked entries -inf
    p = logp.exp()                               # masked entries exactly 0
    entropy = -(p[m] * logp[m]).sum()
    return logp[choice], entropy


def _decode_action(action, node_ids, n_new_index: int) -> dict:
    """Invert a concrete env action into the per-stage index choices the factored heads
    would have to make to PRODUCE it -- the teacher-forcing target for imitation/PPO.
    Mirrors `_walk`'s stage order exactly (single source of truth for the factorization).
    `n_new_index` is the introduce-target index of the NEW sentinel (== len(node_ids))."""
    _T = {name: i for i, name in enumerate(ACTION_TYPES)}
    if isinstance(action, EndSpeech):
        return {"type": _T["end_speech"]}
    if isinstance(action, Extend):
        return {"type": _T["extend"], "extend_target": node_ids.index(action.node_id)}
    if isinstance(action, Concede):
        return {"type": _T["concede"], "concede_target": node_ids.index(action.node_id)}
    if isinstance(action, Introduce):
        d = {"type": _T["introduce"],
             "introduce_target": (n_new_index if action.target == NEW
                                  else node_ids.index(action.target)),
             "role": ACTION_ROLE_ORDER.index(action.role)}
        if action.target != NEW:
            d["introduce_edge"] = EDGE_TYPE_ORDER.index(action.edge_type)
        return d
    if isinstance(action, Weigh):
        return {"type": _T["weigh"],
                "weigh_a": node_ids.index(action.node_a),
                "weigh_b": node_ids.index(action.node_b),
                "favors": 0 if action.favors == action.node_a else 1}
    if isinstance(action, Connect):
        return {"type": _T["connect"],
                "connect_source": node_ids.index(action.source_id),
                "connect_target": node_ids.index(action.target_id),
                "connect_edge": EDGE_TYPE_ORDER.index(action.edge_type)}
    raise TypeError(f"cannot decode action of type {type(action).__name__}")


class ActorCritic(nn.Module):
    """Shared-encoder actor + critic. Holds ONE `GraphEncoder`; `evaluate()` runs it
    once and both the value head and the action heads read that single output."""

    # index of each action type in ACTION_TYPES, for readability
    _T = {name: i for i, name in enumerate(ACTION_TYPES)}

    def __init__(self, encoder: GraphEncoder, d_query: int = 128, hidden: int = 128,
                 curriculum: bool = False):
        super().__init__()
        self.encoder = encoder                      # SHARED -- not duplicated/forked
        d = encoder.d_model
        g = encoder.graph_dim
        self.d_query = d_query
        # Opening unlock-curriculum: a policy-layer scaffold that AND-composes with the env
        # legal-action mask during the AFF 1AC (rl_training_spec §Opening curriculum). Off by
        # default; enable for warm-start / early PPO. Never widens legality, never touches
        # the judge.
        self.curriculum = curriculum

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

    def _walk(self, enc_out: EncoderOutput, state, choose):
        """The ONE factored-decision walker: emit each stage's (logits, legal mask) in the
        canonical order and let `choose(name, logits, mask_np) -> index` pick each stage's
        index. Sampling and teacher-forced evaluation differ ONLY in `choose`, so the stage
        order and conditioning can never drift between them.

        Returns (action, action_type, records) where records[name] = (logits, mask_np,
        choice). Conditional stages (weigh's second node, connect's edge_type, ...) are
        built GIVEN the already-chosen prior index, so a legal choice at every stage
        assembles a structurally-legal complete action."""
        node_emb = enc_out.node_embeddings          # (N, d)
        g = enc_out.graph_embedding                  # (graph_dim,)
        node_ids = enc_out.node_ids
        mask = CurriculumLegalActionMask(state) if self.curriculum else LegalActionMask(state)
        assert node_ids == mask.node_ids, "encoder node order must match mask node order"

        records: dict = {}

        def stage(name, logits, mask_np):
            idx = choose(name, logits, mask_np)
            records[name] = (logits, mask_np, idx)
            return idx

        t = stage("type", self.type_head(g), mask.type_mask())
        type_name = ACTION_TYPES[t]

        if type_name == "end_speech":
            return EndSpeech(), type_name, records

        if type_name in ("extend", "concede"):
            slot = type_name
            tmask = getattr(mask, f"{slot}_target_mask")()
            i = stage(f"{slot}_target", self.pointer(node_emb, self._q(slot, g)), tmask)
            action = Extend(node_ids[i]) if type_name == "extend" else Concede(node_ids[i])
            return action, type_name, records

        if type_name == "introduce":
            items = torch.cat([node_emb, self.new_token.unsqueeze(0)], dim=0)   # (N+1, d)
            i = stage("introduce_target",
                      self.pointer(items, self._q("introduce_target", g)),
                      mask.introduce_target_mask())
            is_new = (i == len(node_ids))
            target = NEW if is_new else node_ids[i]
            r = stage("role", self.role_head(g), mask.introduce_role_mask(target))
            role = ACTION_ROLE_ORDER[r]
            if is_new:
                return Introduce("", role, NEW, None), type_name, records
            ctx = torch.cat([g, node_emb[i]], dim=0)
            e = stage("introduce_edge", self.introduce_edge_head(ctx),
                      mask.introduce_edge_mask(target, role))
            return Introduce("", role, target, EDGE_TYPE_ORDER[e]), type_name, records

        if type_name == "weigh":
            ia = stage("weigh_a", self.pointer(node_emb, self._q("weigh_a", g)),
                       mask.weigh_a_mask())
            a_id = node_ids[ia]
            ib = stage("weigh_b", self.pointer(node_emb, self._q("weigh_b", g)),
                       mask.weigh_b_mask(a_id))
            b_id = node_ids[ib]
            pair = torch.stack([node_emb[ia], node_emb[ib]], dim=0)             # (2, d)
            f = stage("favors", self.pointer(pair, self._q("favors", g)),
                      mask.favors_mask(a_id, b_id))
            favors_id = a_id if f == 0 else b_id
            return Weigh(a_id, b_id, favors_id, ""), type_name, records

        # connect
        isrc = stage("connect_source", self.pointer(node_emb, self._q("connect_source", g)),
                     mask.connect_source_mask())
        s_id = node_ids[isrc]
        itgt = stage("connect_target", self.pointer(node_emb, self._q("connect_target", g)),
                     mask.connect_target_mask(s_id))
        t_id = node_ids[itgt]
        ctx = torch.cat([g, node_emb[isrc], node_emb[itgt]], dim=0)
        e = stage("connect_edge", self.connect_edge_head(ctx),
                  mask.connect_edge_mask(s_id, t_id))
        return Connect(s_id, t_id, EDGE_TYPE_ORDER[e]), type_name, records

    @staticmethod
    def _stages_dict(records) -> dict:
        """Convert `_walk`'s raw records into the public per-stage diagnostics
        (`{"probs", "mask", "choice"}`) that `SampledAction` and tests consume."""
        return {name: {"probs": masked_probs(logits, torch.as_tensor(
            mask_np, dtype=torch.bool, device=logits.device)),
            "mask": mask_np, "choice": choice}
            for name, (logits, mask_np, choice) in records.items()}

    @staticmethod
    def _joint_log_prob_entropy(records):
        """Sum the per-stage log-probs and entropies over exactly the stages that fired
        for this action -- the factored joint log-prob and total entropy (rl_training_spec
        §PPO: 'per-stage log-probs/entropy summed across the factored heads'). Variable
        length: `end_speech` fires one stage, an attaching `introduce` fires three."""
        logp = None
        entropy = None
        for _name, (logits, mask_np, choice) in records.items():
            lp, ent = _stage_log_prob_entropy(logits, mask_np, choice)
            logp = lp if logp is None else logp + lp
            entropy = ent if entropy is None else entropy + ent
        return logp, entropy

    def sample_action(self, enc_out: EncoderOutput, state,
                      generator: Optional[torch.Generator] = None) -> SampledAction:
        """Factored, masked sampling of one legal action from a single encoder output.
        `state` is the live `RoundState`, used ONLY to build masks via the env generator
        (`LegalActionMask`). Returns a `SampledAction` whose `.action` is a concrete,
        structurally-legal env action."""
        def choose(_name, logits, mask_np):
            return _sample_index(logits, mask_np, generator)

        action, type_name, records = self._walk(enc_out, state, choose)
        return SampledAction(action, type_name, self._stages_dict(records))

    def sample_with_log_prob(self, enc_out: EncoderOutput, state,
                             generator: Optional[torch.Generator] = None):
        """Rollout entry point: sample a legal action AND return its factored joint
        log-prob and total entropy (both differentiable w.r.t. the current parameters if
        `enc_out` was produced with grad enabled). Returns
        (SampledAction, log_prob, entropy)."""
        def choose(_name, logits, mask_np):
            return _sample_index(logits, mask_np, generator)

        action, type_name, records = self._walk(enc_out, state, choose)
        logp, entropy = self._joint_log_prob_entropy(records)
        return SampledAction(action, type_name, self._stages_dict(records)), logp, entropy

    def evaluate_action(self, enc_out: EncoderOutput, state, action):
        """Teacher-forcing entry point (imitation + PPO ratio): the factored joint
        log-prob and total entropy of a GIVEN concrete `action` under the current
        parameters, decomposed through the SAME stages `_walk` uses to sample. The
        reconstructed action is asserted structurally identical to `action`, so a decode
        or mask drift fails loudly rather than silently scoring a different move.

        A demonstrated action that is unreachable under the factored masks (its choice at
        some stage is masked out) yields a `-inf` log-prob; callers/tests check finiteness
        to surface such a fixture rather than train on it."""
        node_ids = enc_out.node_ids
        decoded = _decode_action(action, node_ids, n_new_index=len(node_ids))

        def choose(name, _logits, _mask_np):
            return decoded[name]

        rebuilt, _type_name, records = self._walk(enc_out, state, choose)
        assert _same_structure(rebuilt, action), (
            f"teacher-forced rebuild {rebuilt} != demonstrated {action}")
        return self._joint_log_prob_entropy(records)


def _same_structure(a, b) -> bool:
    """Structural equality of two env actions, IGNORING content/justification (the policy
    is content-blind, so a rebuilt action carries an empty placeholder). Used to guard the
    teacher-forcing decode against drift."""
    if type(a) is not type(b):
        return False
    if isinstance(a, Introduce):
        return (a.role, a.target, a.edge_type) == (b.role, b.target, b.edge_type)
    if isinstance(a, (Extend, Concede)):
        return a.node_id == b.node_id
    if isinstance(a, Weigh):
        return (a.node_a, a.node_b, a.favors) == (b.node_a, b.node_b, b.favors)
    if isinstance(a, Connect):
        return (a.source_id, a.target_id, a.edge_type) == (b.source_id, b.target_id, b.edge_type)
    return isinstance(a, EndSpeech)
