"""CDAF policy network (Phase 4).

Currently exposes the graph ENCODER only -- graph-in, per-node + pooled
embeddings out (`encoder_spec.md`). Action heads and the critic head consume
this encoder's output and are built in a later milestone; nothing here builds
them, and nothing here touches the training loop.

The encoder is CONTENT-BLIND by construction: it reads only the environment
`observation` (`env.observation.observe`), which carries no node content at all,
and `policy.features` reads a fixed, enumerated set of keys from it -- never any
content/text field. See `encoder_spec.md` §"the policy is content-blind".
"""

from .features import GraphFeatures, extract_features, RELATION_VOCAB, ROLE_VOCAB
from .encoder import GraphEncoder, EncoderOutput
from .heads import (
    ActorCritic, PointerHead, SampledAction,
    masked_log_probs, masked_probs,
)
from .masking import (
    LegalActionMask, ACTION_TYPES, ACTION_ROLE_ORDER, EDGE_TYPE_ORDER,
)

__all__ = [
    # encoder
    "GraphEncoder",
    "EncoderOutput",
    "GraphFeatures",
    "extract_features",
    "RELATION_VOCAB",
    "ROLE_VOCAB",
    # heads / actor-critic
    "ActorCritic",
    "PointerHead",
    "SampledAction",
    "masked_log_probs",
    "masked_probs",
    # masking
    "LegalActionMask",
    "ACTION_TYPES",
    "ACTION_ROLE_ORDER",
    "EDGE_TYPE_ORDER",
]
