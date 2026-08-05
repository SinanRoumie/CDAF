"""Content-blind feature extraction: environment `observation` -> plain-numpy
tensors the encoder embeds (`encoder_spec.md` §Node features / §Structural bias /
§Global context).

This layer is deliberately framework-free (NumPy only) and PURE: it reads a fixed,
enumerated set of keys off the observation dict and never touches node content. The
observation itself carries no content field (`env/observation.py`), and this module
adds a second guarantee -- it only ever indexes the specific keys listed below -- so
corrupting or injecting a `content` field anywhere in the observation cannot change
its output (pinned by `tests/test_encoder.py::test_content_blind`).

DELIBERATE DEFERRALS for this milestone (ruled with the human, recorded here so a
reader does not mistake them for oversights):

  * EDGE RECENCY (encoder_spec §Edge recency) -- the attention bias is keyed by
    relation type ONLY, not (relation x recency bucket). Edges track no
    introduction speech anywhere in the env state/observation today, so the bucket
    is not reconstructable without an env change (out of the "encoder only" scope;
    encoder_spec itself lists bucket boundaries as Open Item #2). `RELATION_VOCAB`
    and the bias table can gain a recency dimension later without touching this
    layer's node features.

  * COMPARISON `favors` SPLIT (encoder_spec §Structural bias, compares-favored /
    compares-unfavored) -- comparison edges collapse to ONE directed relation pair
    (`compares` / `compared_by`). The observation does not expose the weigh `favors`
    pointer, so favored vs unfavored is not distinguishable from it.

Everything else in encoder_spec §Node features is present or derivable from today's
observation and is encoded here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from model import SPEECH_ORDER, speech_index
from judge.config import PRESUMPTION
from env.actions import SPEECH_BUDGET

# --- pinned vocabularies (stable index order == checkpoint compatibility) ------

# Every role that can appear on a graph node in the observation. This is the
# `introduce` role vocabulary (action_schema_spec) PLUS `weighing`: weigh nodes are
# real graph nodes with role "weighing" (env/state.py), so the embedding table must
# cover them even though `weighing` is not an `introduce` role. (encoder_spec's node
# -feature role list omits it; that omission is resolved here -- it does not change
# what the policy sees, since weighing nodes are already in the observation.)
ROLE_VOCAB = (
    "uniqueness", "link", "impact", "advocacy",
    "framework", "ballot_directive", "weighing",
)
_ROLE_INDEX = {r: i for i, r in enumerate(ROLE_VOCAB)}

# Edge types counted for the degree features (in/out, split by type).
DEGREE_EDGE_TYPES = ("support", "defensive_attack", "offensive_attack", "comparison")

# Directed relation vocabulary for the additive attention bias. Index of the ORDERED
# pair (row i attends to col j): the relation i bears TOWARD j along the authored edge
# between them. "supports/supported_by" are NOT reverse edges -- they are the two ways
# one authored undirected edge presents to the two ordered pairs under all-pairs
# attention (encoder_spec §Structural bias). Comparison is collapsed (see module doc).
RELATION_VOCAB = (
    "no_relation",                  # 0
    "supports",                     # 1  authored support edge i -> j
    "supported_by",                 # 2  authored support edge j -> i
    "defensively_attacks",          # 3
    "defensively_attacked_by",      # 4
    "offensively_attacks",          # 5
    "offensively_attacked_by",      # 6
    "compares",                     # 7  authored comparison edge i -> j (i is the weigh)
    "compared_by",                  # 8  authored comparison edge j -> i
)
_REL_INDEX = {r: i for i, r in enumerate(RELATION_VOCAB)}
# edge_type -> (forward relation index, reverse relation index)
_EDGE_TO_RELATION = {
    "support": (_REL_INDEX["supports"], _REL_INDEX["supported_by"]),
    "defensive_attack": (_REL_INDEX["defensively_attacks"], _REL_INDEX["defensively_attacked_by"]),
    "offensive_attack": (_REL_INDEX["offensively_attacks"], _REL_INDEX["offensively_attacked_by"]),
    "comparison": (_REL_INDEX["compares"], _REL_INDEX["compared_by"]),
}

_N_SLOTS = len(SPEECH_ORDER)            # 7 speech slots
_TERMINAL_SLOT = _N_SLOTS               # extra slot index for the terminal observation
_MAX_BUDGET = max(SPEECH_BUDGET.values())

# Continuous per-node feature layout (documented order; CONT_DIM must match).
#   [0]      presumption flag        (owner is the presumption-favored side)
#   [1]      speeches-ago (norm)     (current_slot_index - intro_index) / N_SLOTS
#   [2:9]    liveness stamps         7-dim binary over SPEECH_ORDER (carried?)
#   [9:12]   settled facts           dropped, permanent-extension-fail, reachable
#   [12:20]  degree features         in/out for each of 4 edge types (log1p)
#   [20]     sigma                   per-node DF-QuAD strength (already in [0,1])
#   [21:25]  eff-pol one-hot         [none, +1, -1, unresolved]
CONT_DIM = 1 + 1 + _N_SLOTS + 3 + 2 * len(DEGREE_EDGE_TYPES) + 1 + 4
GLOBAL_CONT_DIM = 2                     # [remaining_budget/max, moves_used/max]


@dataclass
class GraphFeatures:
    """Framework-free feature bundle for one observation. Arrays are node-aligned in
    observation order; `node_ids[k]` is the id of row k. `rel[i, j]` is the directed
    relation index of the ordered pair (i attends to j)."""
    node_ids: List[str]
    role_idx: np.ndarray        # (N,) int64   -> role embedding
    owner_idx: np.ndarray       # (N,) int64   -> relative-owner embedding (0 self / 1 opp)
    speech_idx: np.ndarray      # (N,) int64   -> introduction-speech embedding
    cont: np.ndarray            # (N, CONT_DIM) float32
    rel: np.ndarray             # (N, N) int64
    slot_idx: int               # current speech slot (or _TERMINAL_SLOT) -> global embedding
    global_cont: np.ndarray     # (GLOBAL_CONT_DIM,) float32

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)


def _eff_pol_onehot(value) -> List[float]:
    """[is_none, is_+1, is_-1, is_unresolved]. `value` is None (non-offense node),
    +1, -1, or the UNRESOLVED sentinel ('?') -- matched structurally so no import of
    the sentinel is needed."""
    if value is None:
        return [1.0, 0.0, 0.0, 0.0]
    if value == 1:
        return [0.0, 1.0, 0.0, 0.0]
    if value == -1:
        return [0.0, 0.0, 1.0, 0.0]
    return [0.0, 0.0, 0.0, 1.0]         # UNRESOLVED ('?')


def extract_features(obs: Dict) -> GraphFeatures:
    """Build `GraphFeatures` from an environment observation. Reads ONLY the keys
    enumerated below; no content/text field is ever accessed (content-blind)."""
    graph = obs["graph"]
    nodes = graph["nodes"]
    edges = graph["edges"]
    seq = obs["sequence"]
    accrual = obs.get("accrual", {})
    sigma = accrual.get("sigma", {})
    eff_pol = accrual.get("eff_pol", {})
    dropped = set(obs.get("closed_window_drops", ()))
    ext_fail = set(obs.get("permanent_extension_failures", ()))
    reach = obs.get("reachability", {})

    # Perspective for RELATIVE owner (encoder_spec §Why relative owner): self ==
    # the side to move. On the terminal observation `side` is None; fall back to AFF
    # so the feature is defined (the encoder is consulted at decision time, where a
    # side is always present, so this only affects a terminal-state encode).
    perspective = seq.get("side") or "AFF"

    cur_idx = seq["slot_index"]         # current slot index; == _N_SLOTS once terminated
    id_to_pos = {n["id"]: k for k, n in enumerate(nodes)}
    n = len(nodes)

    role_idx = np.zeros(n, dtype=np.int64)
    owner_idx = np.zeros(n, dtype=np.int64)
    speech_idx = np.zeros(n, dtype=np.int64)
    cont = np.zeros((n, CONT_DIM), dtype=np.float32)

    # Degree counts, split by edge type and direction (out = authored source, in =
    # authored target). Direction here is bookkeeping for a feature, not a semantic
    # claim -- the judge reads edges undirected.
    out_deg = {nid: {t: 0 for t in DEGREE_EDGE_TYPES} for nid in id_to_pos}
    in_deg = {nid: {t: 0 for t in DEGREE_EDGE_TYPES} for nid in id_to_pos}
    for e in edges:
        t = e["edge_type"]
        if t not in out_deg[e["source"]]:      # unknown type: ignore for degree feats
            continue
        out_deg[e["source"]][t] += 1
        in_deg[e["target"]][t] += 1

    for nid, node in ((node["id"], node) for node in nodes):
        k = id_to_pos[nid]
        role_idx[k] = _ROLE_INDEX[node["role"]]
        owner_idx[k] = 0 if node["owner"] == perspective else 1
        intro_i = speech_index(node["introduction_speech"])
        speech_idx[k] = intro_i

        row = cont[k]
        row[0] = 1.0 if node["owner"] == PRESUMPTION else 0.0
        row[1] = (cur_idx - intro_i) / _N_SLOTS
        carried = set(node.get("carried_speeches", ()))
        for si, slot in enumerate(SPEECH_ORDER):
            row[2 + si] = 1.0 if slot in carried else 0.0
        row[9] = 1.0 if nid in dropped else 0.0
        row[10] = 1.0 if nid in ext_fail else 0.0
        row[11] = 1.0 if reach.get(nid, False) else 0.0
        base = 12
        for ti, t in enumerate(DEGREE_EDGE_TYPES):
            row[base + ti] = np.log1p(in_deg[nid][t])
            row[base + len(DEGREE_EDGE_TYPES) + ti] = np.log1p(out_deg[nid][t])
        s = sigma.get(nid)
        row[20] = float(s) if s is not None else 0.0
        row[21:25] = _eff_pol_onehot(eff_pol.get(nid))

    rel = _relation_matrix(edges, id_to_pos, n)

    slot = _TERMINAL_SLOT if seq.get("terminated") else cur_idx
    global_cont = np.array([
        seq["remaining_budget"] / _MAX_BUDGET,
        seq["moves_used"] / _MAX_BUDGET,
    ], dtype=np.float32)

    return GraphFeatures(
        node_ids=[node["id"] for node in nodes],
        role_idx=role_idx, owner_idx=owner_idx, speech_idx=speech_idx,
        cont=cont, rel=rel, slot_idx=int(slot), global_cont=global_cont,
    )


def _relation_matrix(edges, id_to_pos, n) -> np.ndarray:
    """(N, N) directed relation index. `rel[i, j]` is the relation node i bears toward
    node j: the forward relation if an authored edge runs i -> j, the reverse relation
    if it runs j -> i, else no_relation. First matching edge wins if a pair has several
    (deterministic in observation edge order); the diagonal stays no_relation."""
    rel = np.zeros((n, n), dtype=np.int64)
    for e in edges:
        fr = _EDGE_TO_RELATION.get(e["edge_type"])
        if fr is None:
            continue
        i = id_to_pos.get(e["source"])
        j = id_to_pos.get(e["target"])
        if i is None or j is None or i == j:
            continue
        if rel[i, j] == 0:
            rel[i, j] = fr[0]           # i -> j : forward
        if rel[j, i] == 0:
            rel[j, i] = fr[1]           # j <- i : reverse
    return rel
