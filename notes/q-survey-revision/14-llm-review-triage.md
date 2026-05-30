# LLM-Review Triage — three reviews of v0.14

**Synced to v0.24 (2026-05-30).** Status: triage complete; adopted items
landed, the rest routed to supplement or declined with rationale.

This document records three reviews received on the v0.14 draft and the
disposition of each ask. All three are **LLM-generated** (tell-tales:
the exhaustive "must add these N named algorithms" lists, a "Technical
Appendix of formulas to paste in," and — in one — un-stripped
search-tool citation tokens). They are useful as an adversarial pass but
share two systematic biases that shaped our triage:

1. **Dominant thrust is "add more mathematics/algorithms,"** which
   collides head-on with the IEEE TAI **21-page cap** and TAI's own
   guidance ("over-use of mathematics should be avoided… proofs →
   supplementary; compare against the top two or three competitive
   algorithms"). We are *compressing*, not expanding the core.
2. **They did not register existing coverage.** Many "must add" items
   were already in the draft (CBDQ, QFIX, SICQL/ICQL, ShiQ, Q♯; the
   planning-oracle framing in §VI; Hundal non-interchangeability in §VII;
   the rliable caveat in §V).

Net policy: **rigor/clarity asks → adopt in core; math-depth asks →
supplement (where the cut derivations already live); scope-expansion
asks → decline, bounded.**

## Review 1 (v0.14) — "accept after major revision"

Enthusiastic; wanted broad content additions.

| Ask | Disposition |
|---|---|
| Continuous action spaces (DDPG, NAF, QT-Opt, PI-QT-Opt, CAQL, CQSM) — called a "fatal flaw" | **DECLINED (bounded).** Out of core scope for a discrete-focused Q-learning survey at 21pp; DDPG is actor-critic, not Q-learning. Hybrid discrete-continuous (PDQN) retained in §IV.A; the rest noted as a scope boundary. |
| DDQL (deep double Q, reciprocal bootstrapping) math + stabilization | **PARTIAL.** Conceptual treatment already in §IV.A; full equations → supplement S5. |
| AC-CDE (action-candidate clipped double estimator) | **DECLINED.** Marginal; §IV.A already covers the bias/variance frontier. |
| CBDQ (cognitive belief-driven Q) | **ALREADY PRESENT** (§IV.C). |
| QFIX / Q+FIX | **ALREADY PRESENT** (§IV.F); formula → supplement S5. |
| FQL depth | **ALREADY PRESENT** (§IV.E). |
| ShiQ / Q♯ / SICQL / ICQL / Q-shaping depth | **ALREADY PRESENT** (§IV.J / §IV.G); formal losses → supplement S5. |
| rliable / more seeds | **ADOPTED** (see cross-cutting below). |

## Review 2 (v0.14) — "major revision; retarget to a survey venue"

Skeptical; rigor and auditability focus. The strongest of the three.

| Ask | Disposition |
|---|---|
| Methodology not systematic enough | **CLOSED.** §III rewritten as an explicit systematic/PRISMA protocol (databases, search strings, 5 inclusion/exclusion criteria, ~200→120→80 screening). |
| Release auditable artifacts (search log, extraction sheets, code) | **CLOSED.** Supplement S1 (search log), S2 (method index), S3/S4 (full tables), S6 + committed reproducible script. |
| Tabular 5 seeds insufficient → 20+ | **CLOSED (exceeded).** §VI re-run at **100 seeds** with bootstrap CIs. |
| Repository audit Table VII "DQN inconsistency" | **CLARIFIED (was a misread).** The 2013-architecture (no target net) vs 2015-architecture distinction is correct; labeling clarified, full matrix → supplement S4. |
| Tighten causal / "first" language | **ADOPTED.** "First" hedged to "to our knowledge"; benchmark claims calibrated. |
| Separate consensus from interpretive reassignments | **ADOPTED.** §III states reinterpretations (distributional-as-credit-assignment, dueling-as-stability) are argued in-section. |
| Scope control; demote foundation-model alignment | **ADOPTED.** §IV.J reframed as a brief emerging-direction; whole paper distilled 47pp→19pp. |
| Remove internal note filenames (`07-prior-art-sweep.md`) | **CLOSED.** Stripped; now "supplementary material." |

## Review 3 (v0.14) — "major revision" (DDQL/Q+FIX/PQN math)

Heavy on formulas to add; one genuinely new analytical idea.

| Ask | Disposition |
|---|---|
| **Cross-axis interaction table (W1–W8)** | **ADOPTED into core** — the one substantive new idea; added to the §IV overview (origin / principal interaction / deployment failure mode). |
| Planning oracles vs model-free conflation (§VI) | **ADOPTED.** VI/PI/MPI/CVPI now reported as a *separated planning-oracle upper bound*, not a competitor. (The framing was partly present; now explicit and tabulated.) |
| 100 seeds + bootstrap CIs (§VI) | **ADOPTED.** |
| rliable for Atari (§V) | **ALREADY PRESENT;** kept. |
| Repository non-interchangeability (Hundal) | **ALREADY PRESENT** (§VII). |
| DDQL equations; Q+FIX formula + Dec-POMDP V(h,s); SICQL/ICQL losses; PQN LayerNorm-Lipschitz contraction; scaling-law formulas | **ROUTED TO SUPPLEMENT S5** (not the page-limited core), consistent with TAI's proofs-to-supplementary guidance. |

## One-line summary for co-authors

Every *legitimate* point across the three reviews is addressed — the
rigor and clarity asks in the core, the formal math in the supplement —
without inflating the core past the 21-page cap. The only outright
decline is the full continuous-action-space expansion, on scope grounds.
