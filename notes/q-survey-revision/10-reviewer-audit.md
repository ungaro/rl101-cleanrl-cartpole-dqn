# Reviewer-Feedback Audit Against Current Draft

A structured comparison of the IEEE TAI reviewer asks (extracted from
`RL_Project_Pitch_Q_Survey.pdf`, pages 17–24) against what the
current draft delivers, including the three closing sessions and the
late-2025 / 2026 paper sweep.

Legend:
- ✓ covered
- ◐ partial
- ✗ gap

---

## Feedback 1 — Algorithms feel isolated

Reviewer sub-asks (pitch deck p.18):

| Sub-ask | Status | Where in draft |
|---|---|---|
| Stronger cross-method comparisons | ✓ | Per-section F. Comparison summary tables (uniform 6-column format across §IV.A–H); 2D positioning grids in §IV.A, IV.C, IV.D, IV.E, IV.H; axis × mechanism-family matrix in §IV overview |
| Identify shared design principles | ✓ | Axis × mechanism-family matrix in §IV overview (decoupling / ensembles / architectural / behavior / distributed); mechanism families recur as subsection labels (B.1, B.2, …) across axis-sections |
| Explain trade-offs between methods | ✓ | Every axis-section §IV.X.C "Trade-offs" subsection |
| Unify algorithm families conceptually | ✓ | Genealogy figure (§IV overview); cross-references between sections (Rainbow appears in §IV.B primary + cross-refs from §IV.A, IV.C, IV.D, IV.H); legacy vs. axis mapping in Appendix A |

**Verdict:** all four sub-asks covered.

**Remaining polish (optional):** explicit cross-axis comparisons.
The ensemble mechanism appears in §IV.A (EBQL for bias control),
§IV.C (Bootstrapped DQN, UCB Q-Ensemble for exploration), and §IV.E
(EDAC for offline RL) — same architectural primitive, three
different axes. A short callout in §IV overview ("when one mechanism
serves multiple axes") would make this connective tissue more
visible.

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

**Remaining polish (optional):** the "less cataloging, more
synthesis" framing could be made more explicit in §I. One paragraph
contrasting the problem-first frame against the typical
catalogue-style survey would inoculate against any reviewer reading
the matrix or genealogy as a different kind of catalogue.

---

## Feedback 3 — Atari is not enough

**Closed in Session 2.**

Reviewer sub-asks (pitch deck p.20):

| Sub-ask | Status | Where in draft |
|---|---|---|
| Benchmark limitations | ✓ | New §V.H "Limitations of Atari as a Q-learning benchmark" catalogs six inherent limits (determinism, discrete action space, single-task per episode, fixed environments, compute scale, visual-only observations); each tied to the axis it prevents Atari from diagnosing |
| Reproducibility issues | ✓ | §VII.A strengthened with one sentence citing Henderson et al. 2018 (*Deep RL That Matters*) and Engstrom et al. 2020; Hundal et al. 2025 now situated in the broader reproducibility-crisis literature |
| Comparability across papers | ✓ | §V.G discusses protocol divergence (training-frame budgets, seed counts, no-op starts, sticky-action settings) |
| Benchmark-specific improvements | ✓ | New §V.I "Newer benchmarks for Q-learning evaluation" surveys Atari-100k (sample efficiency), ALE-stochastic (sticky-action standard), ProcGen (procedural generalization), NetHack (long-horizon stochastic), BSuite (axis-stratified by design — natural fit for the eight-weakness structure); cross-references D4RL (§IV.E) and SMAC (§IV.F) |
| Real-world generalization challenges | ✓ | New §VIII.C fourth open direction "Q-learning under real-world distribution shift" — positions deployment-side distribution shift as natural extension of §IV.E's training-data treatment; calls out sim-to-real, OOD robustness, online policy correction; references ProcGen / CARL / RLBench as evaluation infrastructure |

**Verdict:** all five sub-asks now covered. This was the largest
unmet area pre-Session-2.

---

## Feedback 4 — Too much derivation

**Closed in Session 3.**

Reviewer sub-asks (pitch deck p.21):

| Sub-ask | Status | Where in draft |
|---|---|---|
| Reduce mathematical repetition | ✓ | §II.C Notation conventions removes the need for each §IV section to redefine $\theta, \theta^-, \alpha, \gamma, \pi, \mathcal{D}$; consolidated notation table now in Appendix B.1 |
| Shorten low-level derivations | ✓ | Heavy derivations moved to Appendix B: max-of-noisy-estimators bound (B.2), categorical distributional projection (B.3), Wasserstein contraction proof sketch (B.4), pessimism LCB argument (B.5), QPLEX IGM completeness (B.6), Watkins & Dayan tabular convergence (B.7). §IV body now keyed to mechanism-defining equations only |
| Allocate more space to interpretation | ✓ | Every §IV section has C. Trade-offs, D. Empirical evidence, and E. Open questions subsections that are entirely interpretive |
| Improve readability and flow | ✓ | Five-part subsection structure (A weakness → B families → C trade-offs → D evidence → E open questions) gives consistent flow; new "Reading guide" paragraph in §I tells readers that §IV bodies focus on mechanism + trade-off, with derivations concentrated in Appendix B and skippable on first read |

**Verdict:** all four sub-asks covered.

---

## Feedback 5 — Need more modern RL

**Closed in Session 1 plus the late-2025 / 2026 sweep.**

Reviewer sub-asks (pitch deck p.22):

| Sub-ask | Status | Where in draft |
|---|---|---|
| Distributed reinforcement learning | ✓ | §IV.G — Ape-X, R2D2, Agent57, PQN; compute-scaling story; bandit-controlled exploration meta-policy |
| Offline reinforcement learning | ✓ | §IV.E — full axis section covering BCQ, BRAC, AWAC, CQL, IQL, EDAC across four mechanism families; new B.5 adds flow-matching policy with Q-learning (FQL, ICML 2025); D4RL benchmarks; decision tree |
| Multi-agent coordination | ✓ | §IV.F — full axis section covering VDN, QMIX, QPLEX, QTRAN; IGM framing; new B.5 adds QFIX (Baisero et al. 2025) residual-correction layer; SMAC benchmarks; cross-axis open question on multi-agent offline |
| Meta-learning integration | ✓ | §IV.G.B.3 expanded into a full mechanism survey across three families: optimization-based (MAML, Reptile-Q, ProMP), context-based (PEARL, MQL), and forward-pass / in-context (AdA, Algorithm Distillation, SICQL ICLR 2026, ICQL ICLR 2026); implicit vs. explicit meta-learning given its own subsection |
| Recent theoretical advances | ✓ | New §IV.I "Theoretical Foundations and Recent Advances" — foundational results (Watkins-Dayan 1992, deadly triad, GTD/TDC), recent advances (distributional Wasserstein contraction Bellemare 2017 / Rowland 2018, finite-time bounds Yang 2019 / Fan 2020, offline pessimism Jin et al. 2021, stability theory Lyle 2023 + Nikishin 2022 + Klein 2026 plasticity-loss survey, QPLEX IGM completeness), evidence-theory interaction patterns, six open theoretical questions, results-by-year table |

**Verdict:** all five sub-asks now covered, including the late-2025
/ 2026 additions (DDQL, SICQL, ICQL, QFIX, FQL, Klein plasticity
survey).

---

## Other suggestions from the deck

Pitch deck p.23 ("Concrete Team Tasks"):

| Task | Status |
|---|---|
| Strengthen comparative analysis | ✓ (axis matrix, comparison tables, positioning grids) |
| Improve conceptual organization | ✓ (problem-first reorganization is the whole revision) |
| Verify benchmark interpretation | ✓ (Session 2 — §V.H limits + §V.I newer benchmarks) |
| Identify modern RL additions | ✓ (Session 1 + late-2025/2026 sweep) |
| Reduce redundant derivations | ✓ (Session 3 — Appendix B + reading guide) |
| Improve figures/tables | ✓ (genealogy, axis matrix, per-section tables + 2D grids, decision tree; Tables I–VIII inline) |
| Strengthen future directions section | ✓ (§VIII.B repository spin-off as named deliverable; §VIII.C four cross-axis open directions including real-world distribution shift) |

All seven Concrete Team Tasks now ✓.

---

## Cross-cutting concerns — still open

These are not on the reviewer-feedback list but remain unfinished as
the team approaches submission.

1. **No References / bibliography section** in the markdown. The
   prose cites [1]–[55] and named-year authors, but no References
   list is generated yet. The team's existing `.bib` file will need
   to be referenced via `pandoc --citeproc --bibliography refs.bib
   --csl ieee.csl`, or pasted in as a final markdown section.

2. **Author block is empty.** The YAML metadata in
   `draft/0-metadata.md` has `title` and `abstract` but no `author`.
   Needs to be filled in before submission.

3. **Two contested re-interpretations** that may invite reviewer
   pushback and warrant explicit defense:
   - Distributional RL relocated from "Statistical" (uncertainty) to
     §IV.D (credit assignment). Argued in §IV.D.A and reinforced in
     §IV.D.E.
   - Dueling DQN relocated from "Q-Function Computation" to §IV.H
     (stability mechanism). Argued in §IV.H.B.2 with Rainbow
     ablation evidence.

   Both are defensible but reviewers may resist on the basis of
   convention. Possible mitigation: a "Re-interpretation rationale"
   footnote at first mention of each.

4. **Q-learning repository spin-off** (§VIII.B) is named as a
   deliverable but the repo itself is not stood up. Reviewers
   reading §VIII may ask for the URL. Either stand it up before
   submission or reframe as a future commitment.

5. **Two-column layout.** Reverted in the current build because
   pandoc's default `\begin{longtable}` for markdown tables conflicts
   with the twocolumn class. Real two-column requires either
   converting every markdown table to raw LaTeX `\begin{table*}` or
   switching `documentclass=IEEEtran`. Deferred.

6. **Final visual polish on tables and figures.** Some tables still
   wrap awkwardly; TikZ rendering of the master genealogy is pending
   (mermaid versions are in the PDF as PNG placeholders).

---

## Completed work — three sessions and a late-2025 / 2026 sweep

The original audit recommended three sessions to close the open
sub-asks. All three are done, plus a follow-on sweep.

**Session 1 — Modern-RL gaps (Feedback 5).**
- Expanded §IV.G.B.3 meta-learning subsection from ~3 paragraphs to
  a full mechanism survey: optimization-based (MAML, Reptile-Q,
  ProMP), context-based (PEARL, MQL), forward-pass (AdA, Algorithm
  Distillation)
- Created §IV.I "Theoretical Foundations and Recent Advances" —
  ~2300-word ninth-section with foundational results, recent
  advances, evidence-theory interaction, open questions, and
  results-by-year table

**Session 2 — Atari limitations (Feedback 3).**
- §V.H "Limitations of Atari as a Q-learning benchmark" — six
  inherent limits tied to the axes Atari cannot diagnose
- §V.I "Newer benchmarks for Q-learning evaluation" — Atari-100k,
  ALE-stochastic, ProcGen, NetHack, BSuite
- §VII.A reproducibility-crisis citation paragraph
- §VIII.C fourth open direction on real-world distribution shift

**Session 3 — Derivation audit and Appendix B (Feedback 4).**
- Appendix B "Notation and Selected Derivations" stood up —
  consolidated notation table + six selected derivations the §IV
  body references rather than works through
- §I "Reading guide" paragraph telling readers Appendix B is the
  home of longer derivations

**Late-2025 / 2026 sweep.** A May 2026 arXiv sweep for post-2025
Q-learning work added five method citations:

| Paper | Where in draft |
|---|---|
| Nagarajan, White & Machado 2026 — Deep Double Q-Learning (DDQL) | §IV.A.B.1 |
| Liu et al. 2026 — Scalable In-Context Q-Learning (SICQL), ICLR 2026 | §IV.G.B.3 |
| Xu et al. 2026 — In-Context Compositional Q-Learning (ICQL), ICLR 2026 | §IV.G.B.3 |
| Baisero et al. 2025 — QFIX | §IV.F.B.5 (new) |
| Park, Li & Levine 2025 — Flow Q-Learning (FQL), ICML 2025 | §IV.E.B.5 (new) |
| Klein et al. 2026 — Plasticity Loss in DRL: A Survey | §IV.I.C.4 + §IV.I.F results table |

Title updated to 2026; methodology end-date updated to "early 2026."

**Honest negatives from the sweep:** no significant new value-based
distributed system as a post-Agent57 successor; benchmark /
reproducibility literature is quiet since Hundal 2025; no breakthrough
non-asymptotic theoretical result for deep Q-learning in this window
beyond the plasticity-loss thread. The field has consolidated
post-2024 rather than opened new directions.

---

*Audit grounded in: pitch deck pages 17–24 (five-point feedback +
team tasks), draft files in `draft/`, planning files in this
directory, plus a May 2026 arXiv sweep for late-2025/2026 work.*
