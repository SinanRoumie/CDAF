"""M1 renderer — one LLM call per speech (spec §8 M1).

Builds on the M0 skeleton: the skeleton IS the prompt (`linearize.speech_skeletons`).
One call per speech in forward order. Speech N's prompt includes the rendered
prose of speeches 1..N-1 (RS8 — frozen prior content) and the graph only as of
end of speech N (RS7 — no future speeches, no judge state, no verdict). Type-blind
(RS5/RS6): no argument-type labels. Pure analytics, no citations (RS34/RS35).

Frozen model at temperature 0 (RS2), recorded in every artifact (RS32). Cache key
includes the resolution (RS30, extended). LLM calls live only in render/ (M1).

This module is import-safe without an API key: `make_client()` raises a clear
error only when you actually try to call. `render_round(..., dry_run=True)`
builds and returns the prompts without any network call.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

from model.round import Round
from .adapter import analyze
from .linearize import speech_skeletons, SpeechBlock

# --- frozen render parameters (RS2/RS32) -------------------------------------
MODEL_ID = "claude-sonnet-4-6"
TEMPERATURE = 0
MAX_TOKENS = 2000                     # generous per-speech ceiling; budgets are targets
RENDER_MODE = "M1"
PROMPT_TEMPLATE_VERSION = "m1-v3"     # bump on ANY prompt change (RS30 guard); v2 adds RS5b;
                                      # v3: per-node advocacy stem + rule-6 distinct-ids (RS19 note)

# --- resolutions (the shared motion; NOT an argument-type label) -------------
RESOLUTIONS: Dict[str, str] = {
    "R1": "The United States federal government should establish national health "
          "insurance in the United States.",
    "R2": "The United States federal government should enact a moratorium on "
          "hyperscale data center construction.",
    "R3": "The development of Artificial General Intelligence is immoral.",
}


# --- citation validator (RS35) -----------------------------------------------
# Author-year and numeric-cite patterns. Fail loudly on a hit (do not strip):
# a hit means the prompt's no-citation instruction leaked.
_CITE_PATTERNS = [
    re.compile(r"\b[A-Z][a-zA-Z]+\s*['’]\s*\d{2}\b"),            # Smith '19
    re.compile(r"\([A-Z][a-zA-Z]+(?:\s+et\s+al\.?)?,?\s+\d{4}[a-z]?\)"),  # (Smith 2019) / (Smith et al., 2019)
    re.compile(r"\b[A-Z][a-zA-Z]+\s+et\s+al\.?\s*,?\s*\(?\d{4}\)?"),  # Smith et al. 2019
    re.compile(r"\[\d{1,3}\]"),                                       # [12] numeric cite
]


class CitationLeak(Exception):
    """Raised when the rendered prose contains a citation-like pattern (RS35)."""


def find_citations(text: str) -> List[str]:
    hits: List[str] = []
    for pat in _CITE_PATTERNS:
        for m in pat.finditer(text):
            if m.group(0) not in hits:
                hits.append(m.group(0))
    return hits


# --- prompts -----------------------------------------------------------------
SYSTEM_PROMPT = (
    "You render a completed competitive-debate round as readable prose, one speech "
    "at a time, from a structural skeleton. You are a linearizer, not a debater: you "
    "never evaluate who is winning and you never invent structure the skeleton does "
    "not contain.\n\n"
    "The skeleton gives you, per speech: a side (AFF or NEG), one or more argument "
    "components each with a register, and a list of typed events — new claims, "
    "connections between claims, refutations, turns, and extensions — each tagged "
    "with a node/edge id and a word budget.\n\n"
    "Rules:\n"
    "1. Write the prose for THIS speech only. Do not preview later speeches.\n"
    "2. Follow the component order and the event order exactly as given. Each event "
    "becomes prose in place.\n"
    "3. Treat the word budget on each event as a target length, not a hard cap; do "
    "not pad and do not truncate. A 0-word syntax edge becomes a short connective "
    "phrase ('which means', 'and that turns'), not its own sentence.\n"
    "4. Each new node has a claim (the short tag) and a warrant (the explanation). "
    "Weave both into the prose; do NOT print them as labeled fields.\n"
    "5. Warrants are REASONING ONLY — analytic argument, no evidence cards, no data "
    "citations. Never write an author-year reference of any kind (no \"Smith '19\", "
    "no \"(Smith 2019)\", no \"et al.\", no bracketed [3] numbers). This is pure "
    "analytics.\n"
    "6. An extension restates a prior claim in brief — one clause, no new warrant. "
    "Restatement is licensed ONLY by an explicit extension event (a line marked "
    "'extend'). Distinct node ids are distinct commitments: two claims that read "
    "alike but carry different ids are different arguments — render each in full, "
    "never merge them or treat one as a restatement of the other.\n"
    "7. A line marked judge-invisible still gets written (the debater made the move) "
    "and then flagged in one short parenthetical noting it did not land and why.\n"
    "8. Do not name argument types. Never write the words disadvantage, advantage, "
    "kritik, or theory. The register and the structure carry the meaning.\n"
    "9. Advocacy polarity follows side, and side alone (RS5b). The side giving this "
    "speech is named in the skeleton header. An Advocacy introduced by AFF affirms the "
    "resolution; an Advocacy introduced by NEG opposes it — a counterplan or "
    "alternative, never an endorsement of the AFF's position. Derive polarity from side "
    "alone: not from the claim's content, and not from whether the resolution is a "
    "policy or a value.\n\n"
    "Output only the speech prose — no headers, no skeleton echo, no meta-commentary."
)


def _register_gloss(line: str) -> str:
    return line  # component-label lines already carry the register verbatim


def build_user_prompt(resolution_text: str,
                      prior: List[Tuple[str, str, str]],
                      block: SpeechBlock) -> str:
    parts: List[str] = []
    parts.append(f"RESOLUTION: {resolution_text}")
    parts.append(
        "\nThe two sides are AFF (advocating change under the resolution) and NEG "
        "(opposing it). Register meanings: a 'substantive' component argues about the "
        "consequences of the AFF's advocacy; a 'framework' component argues for how to "
        "judge the round; an 'incomplete' component was started but never connected to "
        "an advocacy or framework — render its claim and note it was never tied in.")
    if prior:
        parts.append("\n--- PRIOR SPEECHES (already delivered; do not rewrite) ---")
        for sp, side, text in prior:
            parts.append(f"\n[{sp} · {side}]\n{text}")
    parts.append(f"\n--- SKELETON FOR THIS SPEECH: {block.speech} ({block.side}) ---")
    parts.append("\n".join(block.lines))
    parts.append(f"\n(Total target for this speech ≈ {block.budget} words. Write the "
                 f"{block.speech} prose now.)")
    return "\n".join(parts)


# --- graph hash / cache (RS30/RS31) ------------------------------------------
def canonical_graph_hash(rnd: Round) -> str:
    nodes = sorted((n.id, n.kind, n.side, n.speech,
                    tuple(sorted(n.liveness.items()))) for n in rnd.nodes)
    edges = sorted((e.kind, e.source, e.target) for e in rnd.edges)
    blob = json.dumps([nodes, edges], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def cache_key(rnd: Round, resolution_id: str) -> str:
    """RS30: hash(canonical_graph, model_id, prompt_template_version, render_mode, resolution)."""
    material = "|".join([canonical_graph_hash(rnd), MODEL_ID,
                         PROMPT_TEMPLATE_VERSION, RENDER_MODE, resolution_id])
    return hashlib.sha256(material.encode()).hexdigest()[:20]


def _cache_path(cache_dir: str, rnd: Round, resolution_id: str) -> str:
    return os.path.join(cache_dir, cache_key(rnd, resolution_id) + ".json")


# --- client ------------------------------------------------------------------
def make_client():
    """Return an anthropic.Anthropic client, or raise a clear error. Reads the key
    from ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / an `ant` profile (SDK default)."""
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError(
            "anthropic SDK not installed. `pip install anthropic` in the interpreter "
            "you run render/llm.py with.") from e
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise RuntimeError(
            "No Anthropic credential found. Set ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN) "
            "in the environment before rendering. render/llm.py makes real calls to "
            f"{MODEL_ID}; there is no offline fallback (fabricating prose would defeat M1).")
    return anthropic.Anthropic()


def _call(client, system: str, user: str) -> Tuple[str, dict]:
    """One deterministic speech render. temperature 0, thinking off (Sonnet 4.6)."""
    resp = client.messages.create(
        model=MODEL_ID,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
    usage = {"input_tokens": getattr(resp.usage, "input_tokens", None),
             "output_tokens": getattr(resp.usage, "output_tokens", None)}
    return text, usage


# --- render ------------------------------------------------------------------
@dataclass
class SpeechRender:
    speech: str
    side: str
    budget: int
    prose: str = ""
    words: int = 0
    citations: List[str] = field(default_factory=list)
    usage: dict = field(default_factory=dict)


@dataclass
class RoundRender:
    name: str
    resolution_id: str
    resolution_text: str
    model_id: str
    temperature: int
    template_version: str
    render_mode: str
    speeches: List[SpeechRender]
    cached: bool = False

    def transcript(self) -> str:
        head = [f"# {self.name}  |  {self.resolution_id}: {self.resolution_text}",
                f"# model={self.model_id} temp={self.temperature} "
                f"template={self.template_version} mode={self.render_mode}", ""]
        body = []
        for s in self.speeches:
            body.append(f"--- {s.speech} ({s.side}) ---  [{s.words}w / target {s.budget}]")
            body.append(s.prose)
            if s.citations:
                body.append(f"  ⚠ CITATION LEAK: {s.citations}")
            body.append("")
        return "\n".join(head + body)


def render_round(rnd: Round, resolution_id: str, name: str = "round", *,
                 client=None, cache_dir: Optional[str] = None,
                 dry_run: bool = False, strict: bool = True) -> RoundRender:
    """Render one round under one resolution. dry_run builds prompts only (no calls).

    Returns a RoundRender. With strict=True (default, production) a speech whose
    prose trips the RS35 validator raises CitationLeak — loud, no silent stripping.
    With strict=False (inspection) the leak is recorded on the speech and the run
    continues, so the transcript is fully readable and every leak is reported.
    """
    resolution_text = RESOLUTIONS[resolution_id]
    an = analyze(rnd)
    blocks = speech_skeletons(an, rnd)

    if cache_dir and not dry_run:
        cp = _cache_path(cache_dir, rnd, resolution_id)
        if os.path.exists(cp):
            data = json.load(open(cp))
            rr = RoundRender(**{**data, "speeches":
                                [SpeechRender(**s) for s in data["speeches"]]})
            rr.cached = True
            return rr

    if not dry_run and client is None:
        client = make_client()

    speeches: List[SpeechRender] = []
    prior: List[Tuple[str, str, str]] = []
    prompts: List[str] = []
    for blk in blocks:
        user = build_user_prompt(resolution_text, prior, blk)
        sr = SpeechRender(speech=blk.speech, side=blk.side, budget=blk.budget)
        if dry_run:
            prompts.append(user)
        else:
            prose, usage = _call(client, SYSTEM_PROMPT, user)
            cites = find_citations(prose)
            sr.prose, sr.usage, sr.citations = prose, usage, cites
            sr.words = len(prose.split())
            if cites and strict:
                raise CitationLeak(f"{name}/{resolution_id} {blk.speech}: {cites}")
            prior.append((blk.speech, blk.side, prose))
        speeches.append(sr)

    rr = RoundRender(name=name, resolution_id=resolution_id, resolution_text=resolution_text,
                     model_id=MODEL_ID, temperature=TEMPERATURE,
                     template_version=PROMPT_TEMPLATE_VERSION, render_mode=RENDER_MODE,
                     speeches=speeches)
    if dry_run:
        # stash the prompts on the object for inspection without a network call
        rr.__dict__["prompts"] = prompts
        return rr
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        json.dump({**asdict(rr)}, open(_cache_path(cache_dir, rnd, resolution_id), "w"), indent=1)
    return rr
