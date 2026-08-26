#!/usr/bin/env python3
"""Decompose an M1 transcript into per-node graph content (RS19 claim + warrant).

For each speech, the rendered prose (already reviewed) is split back into per-node
content via one structured LLM call: given the speech prose and the node ids it
introduces, return {id: {claim, warrant}}. claim -> node.label (shown on the graph),
warrant -> node.warrant (shown read-only in the reader's node editor). New JSON is
written; the source round is never modified. Judge-irrelevant (content, not topology).

Usage: graph_content.py <round.json> <transcript.txt> <out.json> [R1|R2|R3]
"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import serialize as mser
from model.speeches import SPEECH_ORDER, speech_index
from render.llm import make_client, _call, RESOLUTIONS

SYS = (
 "You convert ONE rendered debate speech into per-node content for an argument graph. "
 "You are given the speech's prose and the list of argument nodes it introduces (id + "
 "kind). For EACH id return: `claim` — a short tag (<= 12 words) naming that specific "
 "argument, suitable as a node label; and `warrant` — 1-2 sentences, the explanation, "
 "drawn from the prose. Each id is a DISTINCT commitment: give it its own claim/warrant, "
 "never merge two. Type-blind: do not use argument-type words ('disadvantage', 'kritik', "
 "'framework', 'advocacy') as the claim. No citations. Return ONLY JSON of the form "
 '{"<id>": {"claim": "...", "warrant": "..."}} covering EXACTLY the listed ids.')

KIND={'Advocacy':'advocacy/plan-or-counterplan','Link':'causal link','Impact':'impact',
      'Uniqueness':'uniqueness','Framework':'evaluative principle','Weighing':'weighing',
      'BallotDirective':'ballot directive'}

def speech_prose(transcript):
    out={}; cur=None; buf=[]
    for ln in transcript.splitlines():
        m=re.match(r'^---\s+(\S+)\s+\(', ln)
        if m:
            if cur: out[cur]="\n".join(buf).strip()
            cur=m.group(1); buf=[]
        elif cur is not None and not ln.startswith('#'):
            buf.append(ln)
    if cur: out[cur]="\n".join(buf).strip()
    return out

def main(round_path, transcript_path, out_path, rid='R2'):
    rnd=mser.load(round_path)
    prose=speech_prose(open(transcript_path).read())
    client=make_client()
    by_speech={}
    for n in rnd.nodes:
        by_speech.setdefault(n.speech, []).append(n)
    established=[]  # (id, claim) prior context
    content={}
    for sp in SPEECH_ORDER:
        nodes=by_speech.get(sp, [])
        if not nodes or sp not in prose: continue
        listing="\n".join(f"  {n.id} ({KIND.get(n.ntype,n.ntype)}, {n.side})" for n in nodes)
        prior = ("\nAlready established (id: claim), for consistency only:\n" +
                 "\n".join(f"  {i}: {c}" for i,c in established[-20:])) if established else ""
        user=(f"RESOLUTION: {RESOLUTIONS[rid]}\n{prior}\n\n--- {sp} PROSE ---\n{prose[sp]}\n\n"
              f"Nodes introduced in {sp} (fill each):\n{listing}\n\nReturn JSON for exactly these ids.")
        out,_=_call(client, SYS, user)
        m=re.search(r'\{.*\}', out, re.S)
        data=json.loads(m.group(0))
        for n in nodes:
            c=data.get(n.id) or {}
            claim=(c.get('claim') or '').strip()
            warr=(c.get('warrant') or '').strip()
            if claim: n.label=claim
            if warr:  n.warrant=warr
            content[n.id]=(claim,warr)
            if claim: established.append((n.id, claim))
        print(f"  {sp}: {len(nodes)} nodes", file=sys.stderr, flush=True)
    mser.save(rnd, out_path)
    filled=sum(1 for n in rnd.nodes if n.warrant)
    print(f"WROTE {out_path}  ({filled}/{len(list(rnd.nodes))} nodes with warrant)", file=sys.stderr)

if __name__=='__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv)>4 else 'R2')
