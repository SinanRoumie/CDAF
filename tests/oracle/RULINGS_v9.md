# Oracle rulings — v9 uniform-uniqueness migration (Step 4)

Hand-adjudicated verdicts for the redrawn `*.new.json` fixtures, ruled under the
**v9** semantics (`judge_spec.md` §12) — the oracle-first gate. These are derived
by hand, NOT by running the judge (which is still v8). Steps 5–6 implement §12 so
the judge produces these; Step 8 asserts them; Step 9 confirms any delta vs the
pre-migration ruling is intended.

Semantics used throughout (v9 non-unique model):

- **Conceded/unattacked uniqueness = no-op** (§12.4 step 1: nothing to zero). Adding
  uniqueness to a chain whose uniqueness is never attacked is verdict-neutral.
- **Non-unique is STATE-level, not link-level.** All links of a chain converge on one
  impact = one post-world state. A non-unique on *any live link's* uniqueness argues
  that shared state is non-unique, so it **zeros the whole chain's offense (for its
  owning side) — regardless of a parallel independent link** to the same impact
  (r33, r32). Stated side-agnostically per the orientation principle (§2.2): it holds
  whichever side owns the chain and whichever side runs the non-unique. *This refines
  §12.4: its committed "zeroing one path's node does not affect a parallel path" holds
  for delinks but NOT for non-unique — see the proposed §12.4 correction.*
- **Kick-out (delink concession).** The chain's **owning side** may neutralize a
  non-unique by conceding a **delink** on the non-unique'd link (σ(link)→0, killing
  it): the severed link no longer feeds the impact, so its non-unique has no live
  carrier to the shared state, and the owning side's other clean link carries the
  impact (r31). Symmetric by side — NEG can kick a non-unique on its disad exactly as
  AFF can on its advantage.
- **Delink ≠ non-unique.** A plain delink (defensive attack on a *link*) stays per-path
  OR (§3.3.1): it zeros only its own path; a parallel path still carries (r29).

| Fixture | Ruled ballot | reason_class | N | vs pre-migration |
|---|---|---|---|---|
| r29 | AFF | AFF offense | +1.0 | preserved |
| r31 | AFF | AFF offense | +1.0 | preserved — now the non-unique **kick-out** exemplar |
| r32 | NEG | AFF structural failure | ~0 | verdict preserved, mechanism changed |
| r33 | **NEG** | **AFF structural failure** | ~0 | **CHANGED AFF→NEG: non-unique poisons the shared impact** |
| r34 | NEG | presumption | ~0 | preserved |
| r35 | AFF | AFF offense | +1.0 | preserved |
| AFFLinkturnsNeg | AFF | AFF offense | +1.0 | preserved |
| AFFLinkturnsNeg_impactanchor | AFF | AFF offense | +1.0 | preserved |
| AFFturnOutweighed | NEG | NEG offense | <0 | preserved |
| fw_wash_no_extend | NEG | presumption | ~0 | preserved |
| fw_weigh_lockout | NEG | NEG offense | -1.0 | preserved |

Net: one intended verdict change — **r33 (AFF→NEG)** from the v9 non-unique
convergence model. r32 keeps its NEG verdict via the impact-uniqueness gate (no
longer a shared cut-vertex). The non-unique trio: **r33** (non-unique on a live link
→ poisons the shared impact → NEG) · **r31** (same, but AFF concedes a delink and
kicks the link → AFF) · **r32** (non-unique on the convergence impact itself → NEG).

---

## r29 — AFF offense (+1.0)
Two AFF paths converge on impact n3. Path A (u2→n2→n3) is clean and fully extended;
u2, u3 conceded → stand. Path B (n7→n3) has its link delinked by NEG n11
(DefensiveAttack, extended) → σ(n7)→0. Per-path aggregation (§3.3.1a): path A carries
the impact independently of B's death. **AFF, "AFF offense", N=+1.0.**
Assert: one CHAIN, sign +1, mag 1.0; σ(n2)=1.0, σ(n7)≈0.

## r31 — AFF offense (+1.0)   [non-unique kick-out]
Chain B carries BOTH a NEG non-unique (n10→u7) and a NEG delink (n11→n7); AFF concedes
the delink and kicks chain B whole (n7 and u7 only-1AC, dropped §6). Per the kick-out
rule: the conceded delink drives σ(n7)→0, **severing Link B from the shared impact**,
so its non-unique can no longer reach the impact to poison it. AFF then carries the
impact on the clean sibling A (u2, u3 stand). **AFF, "AFF offense", N=+1.0.**
Contrast r33, where the non-unique'd link stays LIVE (no delink conceded), so the
non-unique reaches the impact and zeros it → NEG. The conceded delink is exactly what
separates kick-out (AFF) from poison (NEG).
Assert: σ(n7)<ε (delinked / dropped); path B fails extension; one clean AFF chain, mag 1.0.

## r32 — AFF structural failure ⇒ NEG (~0)   [the one changed mechanism]
Links n2, n7 clean (σ=1.0 each; u2, u7 stand). The NEG non-unique n8 DefensiveAttacks
the IMPACT's uniqueness u3 (conceded, extended) → σ(u3)→0. §12.4 step 1 at n3: a zeroed
wired uniqueness zeroes n3's inbound mechanism contribution (from BOTH links) at n3 →
mag(n3)→0. Sign stays +1 (defensive kill, not a turn). A complete, extended,
still-AFF-favoring chain driven to zero magnitude is §7 **AFF structural failure**, not
presumption. **NEG, "AFF structural failure", N≈0.**
Assert: one CHAIN, sign 1, mag <ε; σ(u3)<ε; σ(n2)=σ(n7)=1.0 (links individually intact).
Pre-migration r32 ruled the SAME verdict via a shared cut-vertex n1; the uniform schema
deletes that node, so the kill now lands on the convergence impact's uniqueness. Verdict
identical, so Step 9 sees no ballot change — only the σ-assertion targets move (n1→u3).

## r33 — AFF structural failure ⇒ NEG (~0)   [non-unique poisons the convergence]
Clean path A (u2→n2→n3) + redundant path B (u7→n7→n3), **both live and fully
extended**. NEG non-unique n8 zeroes Link B's uniqueness u7 (DefensiveAttack,
extended). Because Link B is LIVE, the non-unique reaches the shared post-world state
at impact n3 and argues it is non-unique — poisoning the impact **regardless of the
clean parallel path A** (v9 non-unique convergence model). The complete, extended,
sign-+1 AFF chain is driven to magnitude 0 → §7 **AFF structural failure**. **NEG,
"AFF structural failure", N≈0.**
This is the case the migration exists to get right: with per-link uniqueness, a
non-unique on one link is NOT a per-path delink — it is a claim about the shared
state, so it does not "leave the sibling alone." AFF's only escape is the r31 kick-out
(concede a delink on B to sever it); none is conceded here → NEG.
Assert: σ(u7)<ε; σ(n2)=σ(n7)=1.0 (both link nodes individually intact); one chain,
sign 1, mag <ε (non-unique kill, not a turn).

## r34 — presumption ⇒ NEG (~0)
As r33 but n8 TURNS n7 (OffensiveAttack, Link↔Link, turn-eligible): σ preserved, sign
flips to −1, path B live (NEG-sustained). Convergence ruling 4 (§3.3.1): a live turned
path sharing an impact with a clean AFF path (equal magnitude) WASHES that impact's AFF
offense to 0, unconditionally. The turn banks for NEG only via a NEG BD anchored to it —
there is none → 0. N washes to 0; chain A is intact so this is **presumption**, not
structural failure. **NEG, "presumption", N≈0.** Assert: one CHAIN, sign "?".

## r35 — AFF offense (+1.0), single chain object
Two clean fully-extended redundant links (n1, n2) to one impact n3; u1/u2/u3 stand; no
attacks. Same-sign redundancy aggregates by MAX and emits ONE chain object per
impact-component (§3.3.1b double-count guard) → N=+1.0, not +2.0. **AFF, "AFF offense",
N=+1.0; exactly one CHAIN record.**

## AFFLinkturnsNeg — AFF offense (+1.0)
AFF keeps the plan (n4 extended), drops its own advantage after 2AC (n2/n3 and their
uniqueness u3a go dark — moot). NEG disad n9(uniq-of-trend)→n10→n11→n12(fw), all
extended; added u11 conceded → n11 stands. AFF turns n10 via n16 (OffensiveAttack),
weighs turn>disad-link (n17), anchors the AFF BD to the turning link n16. Turn captures
the chain (eff_pol[n10]=−1): side NEG, owner AFF, in scope of sole framework n12.
Conceded u11 does not disturb the capture (§12.4 no-op). **AFF, "AFF offense", N=+1.0.**

## AFFLinkturnsNeg_impactanchor — AFF offense (+1.0)
Identical to T1 except the AFF BD anchors the CAPTURED IMPACT n11 instead of the turning
link. Both anchor targets are legal (ruling 2 withdrawn) and must resolve identically.
**AFF, "AFF offense", N=+1.0.**

## AFFturnOutweighed — NEG offense (<0)
As T1 but a NEG weigh (n17) prefers the disad link n10 over the turn n16 (determinate) →
turn defeated → n10 keeps +1 via preference → disad reverts to NEG offense (side NEG,
owner NEG, sign +1); the NEG BD n19→n11 scores it. Conceded u11 no-op. **NEG, "NEG
offense", N<0.**

## fw_wash_no_extend — presumption ⇒ NEG (~0)
Two mirror framework chains (adv→link→impact→Framework→BD), each link+impact now carrying
conceded uniqueness (stand, no-op). Frameworks F_aff (1AC only) / F_neg (1NC only) fail
maker-extension → live-set cardinality 0 → WASH, no gating (§5.2). Both chains in scope,
each +1.0, N≈0 → presumption. **NEG, "presumption", N≈0; FRAMEWORK_SELECT.via =
"none_survived"** (two frameworks authored but unextended, not "no_frameworks").

## fw_weigh_lockout — NEG offense (-1.0)
Mirror chains + conceded uniqueness (no-op); frameworks extended. NEG weigh w compares
{F_aff, F_neg} preferring F_neg (determinate via resolve) → F_aff defeated → live={F_neg}
→ F_neg gates. AFF chain is not anchored to F_neg → AFF out of scope; the NEG mirror is
anchored to F_neg → in scope, N=−1.0. **NEG, "NEG offense", N=−1.0**; winning_framework
F_neg, F_aff in defeated set. (Also underwrites r21 permutation-determinism.)
