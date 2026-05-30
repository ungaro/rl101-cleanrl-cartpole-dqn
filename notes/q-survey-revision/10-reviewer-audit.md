# Reviewer-Feedback Audit Against Current Draft

A structured comparison of the IEEE TAI reviewer asks (extracted from
`RL_Project_Pitch_Q_Survey.pdf`, pages 17–24) against what the current
draft delivers.

Legend:
- ✓ covered well
- ◐ partial / needs strengthening
- ✗ gap

For each item: what the reviewers asked, where the draft addresses it
(if at all), and what concrete improvement would close the gap.

---

## Feedback 1 — Algorithms feel isolated

Reviewer sub-asks (pitch deck p.18):

| Sub-ask | Status | Where in draft |
|---|---|---|
| Stronger cross-method comparisons | ✓ | Per-section F. Comparison summary tables (uniform 6-column format across §IV.A–H); 2D positioning grids in §IV.A, IV.C, IV.D, IV.E, IV.H; axis × mechanism-family matrix in §IV overview |
| Identify shared design principles | ✓ | Axis × mechanism-family matrix in §IV overview (decoupling / ensembles / architectural / behavior / distributed); mechanism families recur as subsection labels (B.1, B.2, …) across axis-sections |
| Explain trade-offs between methods | ✓ | Every axis-section §IV.X.C "Trade-offs" subsection enumerates them as named axes (bias↔variance, expressiveness↔trainability, etc.) |
| Unify algorithm families conceptually | ✓ | Genealogy figure (§IV overview); cross-references between sections (Rainbow appears in §IV.B primary + cross-refs from §IV.A, IV.C, IV.D, IV.H); legacy ↔ axis mapping in Appendix A |

**Verdict:** all four sub-asks are addressed.

**Possible polish:** explicit cross-axis comparisons. Right now the
ensemble mechanism is discussed in §IV.A (EBQL for bias control),
§IV.C (Bootstrapped DQN, UCB Q-Ensemble for exploration), and §IV.E
(EDAC for offline RL) — same architectural primitive, three different
axes. A short callout box in §IV overview ("when one mechanism serves
multiple axes") would make this connective tissue more visible.

---

## Feedback 2 — Need more conceptual insight

Reviewer sub-asks (pitch deck p.19):

| Sub-ask | Status | Where in draft |
|---|---|---|
| Why did this direction emerge? | ✓ | Every §IV.X.A "The Weakness" subsection traces the motivating problem |
| What weakness does it solve? | ✓ | §II.B Eight Weaknesses of Vanilla Q-Learning is the conceptual spine, forward-referenced by every axis-section |
| What trade-off does it introduce? | ✓ | §IV.X.C Trade-offs across every axis-section |
| Which methods are philosophically related? | ✓ | Genealogy figure annotations; legacy-indexer reverse view (Appendix A.3 shows where each conventional category fragments across axes) |

**Verdict:** the conceptual layer is the central re-framing of the
revision. All four sub-asks land directly against it.

**Possible polish:** the reviewers' "less cataloging, more synthesis"
framing could be made even more explicit in §I. Adding a paragraph
that contrasts the problem-first frame against the typical
catalogue-style survey would inoculate against any reviewer reading
the matrix or genealogy as a different kind of catalogue.

---

## Feedback 3 — Atari is not enough

Reviewer sub-asks (pitch deck p.20):

| Sub-ask | Status | Where in draft |
|---|---|---|
| Benchmark limitations | ◐ | §V.E "Reporting gaps as evidence" treats *reporting* limitations well; Atari's *inherent* limitations as a deep-RL benchmark (deterministic by default, no continuous control, no real-world physics, fixed environments) are not directly cataloged |
| Reproducibility issues | ◐ | §VII.A introduces Hundal et al. (2025) and distinguishes our taxonomic analysis from their empirical PPO audit; §V.G refrains from leaderboards on protocol-divergence grounds; deeper engagement with the reproducibility crisis (Engstrom et al. 2020, Henderson et al. 2018) is absent |
| Comparability across papers | ✓ | §V.G discusses protocol divergence (training-frame budgets, seed counts, no-op starts, sticky-action settings) |
| Benchmark-specific improvements | ✗ | No discussion of newer Atari variants (Atari-100k sample-efficiency benchmark; ALE-stochastic; sticky-action standardization). No mention of ProcGen, NetHack, BSuite as Q-learning-relevant alternatives. |
| Real-world generalization challenges | ✗ | No section on sim-to-real, robustness, deployment-distribution shift. Distribution shift is treated in §IV.E (offline RL) but only as a training-data regime, not a deployment regime. |

**Verdict:** the strongest unmet feedback area on the original five.
Three of the five sub-asks are partial or missing.

**Concrete improvements to close gaps:**

1. Add a new §V.D subsection (between current §V.C and §V.D)
   titled **"Limitations of Atari as a Q-learning benchmark."** One
   page covering: determinism (sticky actions vs. no-op starts);
   discrete action space (no continuous control); single-task per
   episode (no transfer signal); fixed environments (no procedural
   variation, OOD evaluation); compute regime (Agent57's 78B frames is
   inaccessible to most researchers).

2. Add a §V.H or §V.I subsection (probably at the end of §V)
   titled **"Newer benchmarks for Q-learning evaluation."** Half a page
   covering: Atari-100k (sample-efficiency cut), ALE-stochastic
   variants, ProcGen (procedural generation tests generalization),
   NetHack (long-horizon decision-making), BSuite (axis-stratified
   capability tests). Tie each to which of the eight weaknesses it
   most directly diagnoses.

3. Add to §VIII.C (open directions) a fourth bullet on
   **"Q-learning under real-world distribution shift."** Half a page
   covering: sim-to-real transfer with Q-functions; OOD robustness
   beyond offline-RL distribution shift; deployment-time policy
   correction (online adaptation of pretrained Q).

4. Strengthen §VII.A's Hundal et al. paragraph with a one-sentence
   acknowledgment of the broader reproducibility crisis in deep RL
   (Engstrom 2020, Henderson 2018) and how the taxonomic-coverage
   analysis here complements rather than replaces empirical audits.

---

## Feedback 4 — Too much derivation

Reviewer sub-asks (pitch deck p.21):

| Sub-ask | Status | Where in draft |
|---|---|---|
| Reduce mathematical repetition | ✓ | §II.C Notation conventions removes the need for each §IV section to redefine $\theta, \theta^-, \alpha, \gamma, \pi, \mathcal{D}$ |
| Shorten low-level derivations | ◐ | Per-method paragraphs in §IV are keyed to the *key equation* rather than full derivation, but no audit has been done against the original-draft prose to confirm a net reduction |
| Allocate more space to interpretation | ✓ | Every §IV section has C. Trade-offs, D. Empirical evidence, and E. Open questions subsections that are entirely interpretive |
| Improve readability and flow | ◐ | Five-part subsection structure (A weakness → B families → C trade-offs → D evidence → E open questions) gives consistent flow within each axis; the inter-section transitions are mostly cross-references rather than narrative arc |

**Verdict:** notation centralization addresses repetition; derivation
density is unverified.

**Concrete improvements to close gaps:**

1. **Stand up Appendix B** ("Notation and proofs") — currently
   listed as pending in `03-new-outline.md`. Move heavy derivations
   into it:
   - Parameter-noise likelihood-ratio expansion (§IV.C)
   - Distributional KL-projection operator (§IV.D)
   - Quantile-regression Huber loss (§IV.D)
   - Tabular Q-learning convergence proof sketch (§V)
   - Bellman-operator contraction in the off-policy regime (relevant
     to §IV.E and §IV.H)

2. **Per-section derivation density audit.** A pass through each
   §IV.X.B Solution Families with the question: "is this equation
   load-bearing for the trade-off in §C, or is it ornament?" Equations
   that are not referenced downstream within the section are
   candidates for trimming to one-sentence summaries.

3. **Add a "Reading guide" callout in §I or §II.C** indicating that
   derivations are concentrated in Appendix B and §IV bodies focus on
   mechanism + trade-off. Sets reader expectation.

---

## Feedback 5 — Need more modern RL

Reviewer sub-asks (pitch deck p.22):

| Sub-ask | Status | Where in draft |
|---|---|---|
| Distributed reinforcement learning | ✓ | §IV.G — Ape-X, R2D2, Agent57, PQN; compute-scaling story; bandit-controlled exploration meta-policy |
| Offline reinforcement learning | ✓ | §IV.E — full axis section covering BCQ, BRAC, AWAC, CQL, IQL, EDAC across four mechanism families; D4RL benchmarks; decision tree |
| Multi-agent coordination | ✓ | §IV.F — full axis section covering VDN, QMIX, QPLEX, QTRAN; IGM framing; SMAC benchmarks; cross-axis open question on multi-agent offline |
| Meta-learning integration | ◐ | §IV.G.B.3 — three paragraphs on MAML / Reptile-Q applied to Q-learning, plus a contrast with Agent57's "implicit meta-learning." No coverage of PEARL, ProMP, MQL, context-conditioned approaches, in-context Q-learning. |
| Recent theoretical advances | ✗ | No dedicated treatment. Scattered theoretical content: §II.B eight-weakness formalism, §IV.H deadly triad, §IV.A bias-variance discussion, per-axis E. Open Questions. No section on convergence theory of distributional RL, pessimistic value iteration for offline RL, NTK analyses, finite-time bounds, or the recent stability-theory literature. |

**Verdict:** three out of five solid; two are gaps. Identified in the
previous analysis turn.

**Concrete improvements to close gaps:**

1. **Strengthen §IV.G.B.3 (meta-learning).** Expand from three
   paragraphs to ~1 full page. Add:
   - PEARL [Rakelly et al. 2019] — context-conditioned Q-functions
     with probabilistic context inference
   - ProMP [Rothfuss et al. 2019] — proximal meta-policy search
   - Meta-Q-Learning (MQL) [Fakoor et al. 2020] — multi-task off-policy
     meta-RL with propensity estimation
   - In-context Q-learning via transformer architectures (recent
     post-2023 work)
   - Discuss the *explicit* vs. *implicit* meta-learning distinction
     more thoroughly (currently one paragraph) — this is the
     conceptually load-bearing claim

2. **Add §IV.I "Theoretical Foundations and Recent Advances"** as a
   ninth axis-section, or alternatively as a §III.C "Theoretical
   background" subsection. Covers:
   - Q-learning convergence in the tabular case [Watkins & Dayan 1992;
     Tsitsiklis 1994] — concise restatement
   - Convergence under linear function approximation; the deadly-triad
     formal account [Sutton & Barto 2018, ch. 11]
   - Distributional RL convergence: contraction properties of the
     distributional Bellman operator [Bellemare et al. 2017;
     Rowland et al. 2018]
   - Offline-RL pessimism theory [Jin et al. 2021 — *Is Pessimism
     Provably Efficient?*]; CQL's theoretical lower-bound guarantee
   - Finite-time bounds for deep Q-learning [Yang et al. 2019; Fan et
     al. 2020 *Theoretical Analysis of DQN*]
   - Recent stability theory: layer-normalization compatibility with
     the deadly triad [Lyle et al. 2023]

   Format would follow the eight axis-sections (A weakness as
   "theoretical gap" / B solution families / C trade-offs / D evidence
   / E open questions). This brings the §IV count to nine, breaking
   the 8-axis symmetry — alternative: tuck this content into §III as
   a "C. Theoretical background" subsection and weave per-axis
   citations into each §IV.X.E.

---

## Other suggestions from the deck

Pitch deck p.23 ("Concrete Team Tasks") lists:

| Task | Status |
|---|---|
| Strengthen comparative analysis | ✓ Done (axis matrix, comparison tables, positioning grids) |
| Improve conceptual organization | ✓ Done (problem-first reorganization is the whole revision) |
| Verify benchmark interpretation | ◐ Partial (Atari-limitations gap above) |
| Identify modern RL additions | ◐ Partial (meta-learning thin, theory missing) |
| Reduce redundant derivations | ◐ Unverified (Appendix B pending) |
| Improve figures/tables | ✓ Done (genealogy, axis matrix, per-section tables + 2D grids, decision tree; Tables I–VIII inline) |
| Strengthen future directions section | ✓ Done (§VIII.B repository spin-off as named deliverable; §VIII.C three cross-axis open directions) |

---

## Cross-cutting concerns not on the reviewer list but worth surfacing

1. **No References / bibliography section** in the markdown. The
   prose cites [1]–[55] and named-year authors, but there's no
   References list at the end. The team's existing `.bib` file will
   need to be referenced via pandoc-citeproc (`--citeproc`,
   `--bibliography`, `--csl`) or pasted in as a final markdown
   section.

2. **Author block is empty.** The YAML metadata has `title` and
   `abstract` but no `author`. Needs to be filled in before
   submission.

3. **Two contested re-interpretations** that may invite reviewer
   pushback and need defensible framing:
   - Distributional RL relocated from "Statistical" (uncertainty) to
     §IV.D (credit assignment). Argued in §IV.D.A and reinforced in
     §IV.D.E.
   - Dueling DQN relocated from "Q-Function Computation" to §IV.H
     (stability mechanism). Argued in §IV.H.B.2 with Rainbow ablation
     evidence.

   Both are defensible but reviewers may resist on the basis of
   convention. Possible mitigation: a "Re-interpretation rationale"
   appendix or footnote.

4. **Q-learning repository spin-off** (§VIII.B) is named as a
   deliverable but the repo is not stood up. Reviewers reading §VIII
   may ask for the URL. Either stand it up before submission or
   reframe as future commitment.

5. **No claims-evidence chain audit.** Strong claims like "the first
   multi-axis problem-first treatment of Q-learning" (§I) depend on
   the prior-art sweep in `07-prior-art-sweep.md`. Worth a final
   verification before submission.

---

## Recommended prioritization

If the team has bandwidth for **three more sessions** before
submission:

**Session 1 — Modern-RL gaps (closes Feedback 5):**
- Expand §IV.G.B.3 meta-learning subsection to a full page
- Add §IV.I (or §III.C) "Theoretical Foundations and Recent Advances"

**Session 2 — Atari limitations (closes Feedback 3):**
- Add §V.D "Limitations of Atari as a Q-learning benchmark"
- Add §V.H "Newer benchmarks for Q-learning evaluation"
- Add §VIII.C bullet on real-world distribution shift

**Session 3 — Derivation audit + appendix (closes Feedback 4):**
- Stand up Appendix B "Notation and proofs"
- Per-section derivation-density pass
- Add reading guide to §I or §II.C

Two strong-but-cosmetic items remain after that: the bibliography
generation and the repository spin-off URL.

---

## Post-audit additions — late-2025 / early-2026 coverage sweep

A May 2026 sweep of arXiv cs.LG / cs.AI for post-2025 Q-learning
work (conducted after the three audit sessions above) surfaced five
papers that merit inclusion. They have been integrated into the
draft:

| Paper | Where in draft | Why it matters |
|---|---|---|
| Nagarajan, White & Machado 2026 — Deep Double Q-Learning (DDQL) | §IV.A.B.1 | Revises a foundational citation: argues Double DQN is not equivalent to classical Double Q-learning; DDQL trains two genuinely independent networks; beats DDQN on 47/57 Atari games |
| Liu et al. 2026 — Scalable In-Context Q-Learning (SICQL), ICLR 2026 | §IV.G.B.3 | Multi-head transformer with separate policy and Q-value heads, preserves DP-bootstrap structure inside ICRL |
| Xu et al. 2026 — In-Context Compositional Q-Learning (ICQL), ICLR 2026 | §IV.G.B.3 | Linear-attention transformer infers local Q from retrieved transitions; theoretical bounds; substantial gains on Meta-World compositional subsets |
| Baisero et al. 2025 — QFIX (Fixing Incomplete Value Function Decomposition) | §IV.F.B.5 (new) | Residual-correction layer on VDN/QMIX/QPLEX that recovers full IGM-completeness; simpler than QPLEX, consistent gains on SMACv2 + Overcooked |
| Park, Li & Levine 2025 — Flow Q-Learning (FQL), ICML 2025 | §IV.E.B.5 (new) | One-step flow-matching policy with Q-learning; avoids recursive backprop through diffusion chains; strong across 73 D4RL/OGBench tasks |
| Klein et al. 2026 — Plasticity Loss in DRL: A Survey | §IV.I.C.4 | Organizes 50+ mitigation strategies; the natural successor citation to Lyle 2023 / Nikishin 2022 for the modern stability-theory thread |

The five primary additions cover the late-2025 / 2026 window the
sweep identified as a real gap. Title updated to 2026; methodology
end-date updated to "early 2026."

**Honest negatives from the sweep:** no significant new value-based
distributed system to add as a post-Agent57 successor; the
benchmark / reproducibility-audit literature is quiet since Hundal
2025; no breakthrough non-asymptotic theoretical result for deep
Q-learning in this window beyond the plasticity-loss thread above.
The field has consolidated post-2024 rather than opened new
directions.

---

*Audit grounded in: pitch deck pages 17–24 (five-point feedback +
team tasks), draft files in `draft/`, planning files in this
directory, plus a May 2026 arXiv sweep for late-2025/2026 work.*
