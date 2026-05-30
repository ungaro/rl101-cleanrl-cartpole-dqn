# Q-Survey Revision Changelog

Human-readable history of what the team added or changed in each
working session. Versions are dated rather than semver-numbered since
the manuscript is a single moving target rather than a released
artifact. Granular per-file edits are recorded in the git history;
this file summarizes the *structural* changes a reviewer or
co-author would want to know about.

Each entry lists the version tag, the date, what landed, and the
commit hash that introduced it.

---

## v0.19 — 2026-05-30 — Terminology: "method-type taxonomy" defined once, up front

Reviewer-driven clarity fix. The conventional method-organization scheme
was referred to three inconsistent ways — "legacy taxonomy", "method-type
taxonomy", "conventional six-category taxonomy" — with "legacy" wrongly
implying it was the authors' own deprecated scheme, and the actual
category↔axis mapping deferred to an end-of-paper appendix.

- **One term everywhere: "method-type taxonomy."** All "legacy"
  phrasing removed from the manuscript.
- **Defined once, up front (start of §IV), with a table** listing the
  six method-type categories (Statistical, Q-Function Computation,
  Memory/Replay, Ensemble-Based, Model-Based, Pure Q-Learning), what
  each groups, example methods, and *which problem axes each category's
  methods land in*. The reader meets the concept together with its
  mapping; every later mention is just the term.
- **Appendix A removed from the core** (the deferred per-method index);
  full per-method mapping → supplementary. All dangling "Appendix A/B"
  references (intro, methodology, conclusion, §VII) now point to
  supplementary material.

Page count: 26 (unchanged — clarity, not compression). Clean build,
zero undefined citations.

Commit: `45031e7`

---

## v0.18 — 2026-05-30 — TAI distillation: §IV.I/J demoted + appendices to supplementary

- **§IV.I (theoretical advances)** distilled 1893 → 250 words: a
  synthesis of convergence/deadly-triad, offline pessimism bounds,
  plasticity/capacity loss, and distributional theory — one sentence
  with citation per result; formal statements + proofs → supplementary.
- **§IV.J (foundation-model alignment)** distilled 1970 → 336 words and
  reframed as an *emerging direction* (accessible to TAI's wide
  readership, clearly flagged as not-yet-settled): the KL-regularized
  large-discrete-action-MDP framing plus a compact tour of
  Q-Transformer, VLM-Q, ShiQ, Q♯, SICQL/ICQL, Q-shaping.
- **Appendix A (legacy indexer)** compacted 1311 → 301 words: keeps the
  analytically interesting reverse-view crosswalk (legacy category →
  axis distribution) and the three "why the legacy taxonomy fragments"
  patterns; the full 50-method per-method mapping → supplementary.
- **Appendix B (notation & proofs)** removed from the core build
  (→ supplementary; preserved in `draft-monograph/`).
- Fixed a ` ```math ` fence in §IV.J that pandoc was rendering as a code
  block (undefined `Shaded`/`Highlighting`); now `$$…$$` display math.

**Page count: 34 → 26** (two-column). Remaining to ≤21: front-matter
compression, §V Atari synthesis, §VII repositories, and the new §VI
experiment.

Commit: `c428f16`

---

## v0.17 — 2026-05-30 — TAI distillation: compression ledger + §IV.A–H

Distillation of the TAI edition begins. The eight problem-axis sections
— the bulk of the manuscript — are rewritten to a uniform compact
template, and the steering plan is committed for the co-authors.

- **`13-tai-compression-ledger.md`** (new, in notes) — maps every
  monograph section → {core | supplementary | cut} with word budgets,
  the "distill into a lens, don't delete" principle, and where the
  analysis lives. Core-prose budget ≈ 10–11k words.
- **`draft-tai/main.tex` anonymization toggle** — `\newif\ifanon`;
  `\anontrue` for the double-anonymous submission build (current),
  `\anonfalse` for a named camera-ready copy. Author list preserved
  in the `\else` branch. Submission builds verified to leak no names.
- **§IV.A–H distilled** to the uniform per-axis template: *Weakness →
  Mechanisms (families by what they exploit) → Trade-off (the analysis)
  → Open questions → one compact comparison table.* Run-in **bold**
  lead-ins replace `###` subsubsections; §IV.x demoted to `##`
  subsections under a single §IV section. Derivations, quadrant charts,
  and per-game empirical prose move to supplementary (sourced from
  `draft-monograph/`).
  - Axis word counts: 4a 527, 4b 481, 4c 603, 4d 579, 4e 571, 4f 512,
    4g 619, 4h 549 (from 1.4k–3.2k each; §IV.A–H total ≈ 16.5k → 4.4k).
- A few method citations that were plain text in the monograph
  (HER, Ape-X, SICQL, scaling-laws) are now keyed to existing
  `refs.bib` entries. Zero undefined citations.

**Page count: 47 → 34** (two-column) from the axis distillation alone.
Front matter, §V/§VI/§VII, and the §IV.I/J + appendix demotions remain;
target ≤21.

Commit: `66a18c0`

---

## v0.16 — 2026-05-30 — IEEE TAI edition + monograph freeze

Target venue locked to **IEEE Transactions on Artificial Intelligence**
(Original Research Review Manuscript). TAI caps review papers at 15
pages (21 max, $200/page over 15), two-column IEEEtran, double-anonymous.
The comprehensive draft is now treated as the extended/source version;
a distilled TAI edition is built alongside it.

- **Monograph preserved.** Tagged `monograph-v0.15` and frozen as a
  reference copy at `draft-monograph/`. The comprehensive build
  (`draft/`, single-column, `paper.pdf`) continues unchanged.
- **New `draft-tai/` — working IEEE TAI two-column edition.**
  - `main.tex` — IEEEtran `journal` wrapper: anonymized title block +
    `\thanks`, abstract, impact statement, `IEEEkeywords`; manual
    Roman heading numerals preserved via `secnumdepth=0`; xelatex +
    TeX Gyre Termes (Times-metric) for realistic page count.
  - `build-tai.sh` — pandoc → LaTeX fragment (`--natbib` → IEEE
    `[N]` via `IEEEtran.bst`) + `tables-twocol.lua` + mermaid-filter,
    compiled with pinned system `xelatex`/`bibtex`.
  - `tables-twocol.lua` — converts pandoc `longtable` (illegal in
    two-column) into floats: wide tables (≥4 cols) → full-width
    `table*`, narrow tables → single-column `table`.
  - `IEEEtran.{cls,bst}` kept as committed fallbacks (conda's bibtex
    can't see the system copies).
  - **Baseline: 47 two-column pages** — the real figure to compress
    toward the 21-page cap (~55% reduction ahead).

- **Two correctness fixes (both editions):**
  - Impact statement rewritten to 135 words (was 97, below TAI's
    100–150) with a practitioner "so-what" framing per TAI guidance.
  - Stripped the internal note-leak: the conclusion's reference to
    `07-prior-art-sweep.md` now reads "provided as supplementary
    material." (Required for double-anonymous; no external/internal
    links in the manuscript.)

paper.pdf (monograph): 1.6 MB. main.pdf (TAI): 1.3 MB, 47 pages.

Commit: `873f52c`

---

## v0.15 — 2026-05-30 — Bibliography pipeline (pandoc citeproc + BibTeX)

References are now produced by pandoc-citeproc from a checked-in
BibTeX file plus the IEEE CSL. The hand-curated References section
in `9-references.md` was replaced with an empty `::: {#refs} :::`
anchor that pandoc populates automatically from the cited keys.

- New `draft/refs.bib` — 134 entries, semantic citation keys
  (`mnih_2015_nature`, `hessel_2018_rainbow`,
  `gallici_2024_pqn`, …) covering every paper cited in the body
  plus the 12 named-year references that were missing from the
  prior hand-curated list (CARL, RLBench, EWC, MetaWorld, REDQ,
  Crafter, IBC, RLBench, AdaptCQL, Mendonca, Yu et al. 2020/2021).
- New `draft/ieee.csl` — IEEE Reference Guide (11.29.2023) style
  pulled from `citation-style-language/styles`.
- `0-metadata.md` — YAML block now sets `bibliography: refs.bib`,
  `csl: ieee.csl`, `link-citations: true`, and a
  `reference-section-title: "References"`.
- `build-pdf.sh` — `--citeproc` added to the pandoc invocation.
- Body files — every `[1]…[55]` numeric and every `[Author 20YY]`
  named-year cite converted to `[@key]` (130+ replacements across
  16 files, plus range expansions like `[12]–[15]` → `[@k12; @k13;
  @k14; @k15]`).
- `9-references.md` — replaced with an empty `::: {#refs} :::`
  anchor; the curated numbered list is removed entirely.
- `3-methodology.md` — Table I citation row removed (raw-LaTeX
  table cells can't host `[@key]` markers; the citations live in
  the prose immediately above).

Net effect: adding a new reference now means appending one BibTeX
entry to `refs.bib` and using `[@new_key]` in the body. Renumbering
across the manuscript is no longer manual.

paper.pdf: 1.60 MB, 91 pages, References section now spans 10
pages of IEEE-format entries auto-generated from the 96 keys cited
in the body (and only those — uncited bib entries are dropped by
pandoc-citeproc).

Commit: `2111728`

---

## v0.14 — 2026-05-30 — Session 6: taxonomic and methodological refinements

Three additions closing Reviewer 2's remaining taxonomic and
methodological asks and Reviewer 1 #10.

- §II.B W7 rewritten as an explicitly *composite axis* covering
  both sample throughput (W7a, scaling) and slow adaptation (W7b,
  task transfer). Defended on the grounds that methods responding
  to either mostly respond to both (Ape-X → Agent57 → predictable
  scaling). Readers preferring finer-grained taxonomy are pointed
  at the W7a / W7b sub-labels surfaced in §IV.G.A. The composite
  framing is a deliberate methodological choice; the two-axis
  alternative would either over-classify methods that genuinely
  respond to both sides, or under-cover in-context Q-learning and
  predictable-scaling work that operates across the W7a/W7b
  boundary.

- §IV.G.A — matching composite-axis framing paragraph naming the
  W7a/W7b distinction and explaining how the methods of §IV.G
  populate both sub-axes.

- §IV.D.C trade-offs extended with the *quantile-crossing
  pathology*. QR-DQN / IQN / FQF train independent estimators per
  quantile level without enforcing monotonicity; during training
  predicted quantiles can cross, producing non-monotone
  pseudo-distributions. Non-crossing variants enforce monotonicity
  via constrained parameterization or pairwise penalties. Closes
  Reviewer 1 #10.

- §VII.A extended with two new paragraphs:
  - *Audit methodology and date.* Repository coverage matrix
    reflects state as of 2026-05-01 with audited version ranges
    listed per repository (Tianshou 0.5.x, XuanCe 1.x, CleanRL
    master, DQN Zoo current head, SB3 2.x, RLlib 2.x). Snapshot-
    nature acknowledged.
  - *Named-algorithm support vs. feature-equivalent support.* The
    matrix marks support strictly by named-algorithm presence;
    feature-equivalent availability via configuration (e.g. RLlib's
    DQNConfig with noisy/categorical/distributional flags
    composing a Rainbow-like agent) is consistently higher than
    the matrix indicates, particularly in RLlib and to a lesser
    extent in Tianshou and XuanCe.

paper.pdf: 1.54 MB.

Commit: `be7f05b`

---

## v0.13 — 2026-05-30 — Session 5: large content gaps

Closes Reviewer 1's four largest content-addition asks
(#5, #6, #7, #8 in 12-second-review-audit.md).

- §IV.A.B.4 new — Hybrid discrete-continuous action spaces.
  Parametrized Deep Q-Networks (PDQN, Xiong et al. 2018) for joint
  discrete-action / continuous-parameter Q-learning. Branching
  variants (Bester et al. 2019). Plus transfer-feature-correlation
  regularization as a related mitigation for compound
  overestimation under task transfer.

- §IV.E.B.5 expanded — Q-learning with Adjoint Matching added
  alongside FQL. Where FQL eliminates recursive backprop by
  restricting to a single flow step, adjoint matching tolerates
  the full multi-step flow generation and uses the continuous
  adjoint method to propagate a "lean" adjoint state backward
  without gradients flowing through the entire chain. The
  formulation gives an unbiased policy-improvement guarantee for
  multi-step expressive flow-matching offline-RL policies.

- §IV.G.B.6 new — Predictable scaling laws for value-based deep
  RL (Rybkin et al. 2025). Data and compute required to reach a
  given performance level lie on a strict mathematically
  predictable Pareto frontier governed by the updates-to-data
  ratio. Predictable optimal batch size and learning rate.
  Validated across SAC, Parallel Q-Learning on DM Control, Gym,
  IsaacGym. Restructures the §IV.G framing: scaling is not only
  an architectural axis (distributed parallelism) but also a
  predictable-tuning axis along which any architecture is
  calibrated. Notes the implication for §V Tables II/III: extracted
  methods were not tuned along the predictable-scaling frontier,
  so per-game numbers under-state achievable performance under
  modern UTD-aware tuning.

- §IV.J new — Q-Learning for Foundation Model Alignment
  (`4j-foundation-model-alignment.md`, ~2500 words). Tenth axis-
  section parallel to §IV.I. Covers Q-Transformer (Chebotar
  2023), ShiQ (logits-as-Q for LLM alignment), VLM Q-Learning
  (off-policy multimodal alignment), and Q♯ (distributional
  Q-learning under KL regularization). Five-part structure
  (weakness / solution families / trade-offs / empirical evidence
  / open questions) plus a four-column off-policy ✓ /
  KL-regularized ✓ comparison table. Section opens with explicit
  treatment of the three structural differences (vocabulary-scale
  action space; sparse per-token reward credit assignment;
  KL-regularization against a pretrained reference policy) that
  make foundation-model alignment distinct from classical
  Q-learning. Section closes with an honest note that the section
  deliberately breaks the eight-axis weakness framework: it
  represents a new deployment regime rather than a new mechanism
  category, and is included because Q-learning's role in foundation-
  model alignment is the most significant practical application of
  value-based methods in the current AI ecosystem.

- §IV overview roadmap updated to include §IV.J.

- build-pdf.sh perl substitution table extended with §IV.I and
  §IV.J entries so the new cross-references render as
  hyperlinks in the PDF.

paper.pdf: 1.53 MB. Math-mode wrapping applied to ♯ (Q$^\sharp$) and ✓
($\checkmark$) glyphs not in Latin Modern Roman.

Commit: `351b985`

---

## v0.12 — 2026-05-30 — References-section meta-prose cleanup

The Session 4 references file (`draft/9-references.md`) opened with
a paragraph saying the list "combines the numbered references
inherited from the original draft [1]–[55] with the named-year
additions made during the 2026 revision." Reviewer-facing prose in
the paper draft shouldn't reference an earlier version of itself.
Rewritten to describe the list structure without the meta-history.
Section headers updated: "Numbered references (carried over from
prior draft)" → "Numbered references"; "Named-year references
(added in revision)" → "Additional references." Wider sweep across
the draft confirms no other "original draft" references remain.

paper.pdf: 1.50 MB.

Commit: `76d641d`

---

## v0.11 — 2026-05-30 — Session 4: scholarly apparatus

Closes most of Reviewer 2's scholarly-apparatus and methodological
asks. Six landings:

- §IV.D.A — interpretive-lens paragraph explicitly framing the
  distributional-RL placement under credit assignment as an
  argued re-classification rather than canonical (one of the two
  contested re-interpretations called out in 10-reviewer-audit)
- §III "Source selection" rewritten as a PRISMA-flavored review
  protocol: databases (Google Scholar, arXiv, Semantic Scholar),
  search-string strategy, date cutoff (2026-05-15), inclusion vs
  exclusion criteria, screening flow (~200 → ~120 → ~80), axis-
  assignment protocol, per-paper extraction template. Honest
  about the iterative non-PRISMA character: this is "narrative
  review with empirical add-ons," not a pre-registered
  systematic review
- §III prior-art-sweep pointer for "first multi-axis problem-
  first treatment of Q-learning" claim — names the supporting
  audit so reviewers can independently verify
- §V.G — Agarwal et al. 2021 (rliable / Statistical Precipice) and
  Castro et al. 2020 / Obando-Ceron et al. 2021 (Revisiting
  Rainbow) engagements added to the no-leaderboard discussion
- §VI.E new — statistical reporting and limitations subsection
  acknowledging the five-seed convention is insufficient by
  rliable standards; soft-claim framing for the tabular results.
  Also planning baselines (VI/PI/MPI/CVPI/MCTS) explicitly
  reframed as planning oracles rather than learning peers, with
  CVPI/MPI now defined on first use
- §VIII.D new — Limitations and Ethics subsection covering
  benchmark monoculture, compute inequality, citation bias in
  narrative reviews, stale tooling, and overstating frontier
  methods relative to settled techniques. §VIII closing renamed
  to §VIII.E
- New `draft/9-references.md` — first hand-curated References
  section. Will be superseded by pandoc-citeproc + .bib wiring
  when that lands; provides a visible References section in the
  PDF now

paper.pdf: 1.50 MB.

Commit: `579e69a`

---

## v0.10 — 2026-05-30 — Second-review quick fixes + audit

Three verifiable inconsistencies flagged by the second reviewer
report fixed in-place; a new audit document
(`12-second-review-audit.md`) catalogs both new reviewer reports
against the current draft and proposes Sessions 4-6 to address the
remaining content gaps and methodological asks.

- ⚙ §VII.C "Eight methods" corrected to "Nine methods" (count error
  in the absent-from-all-repositories list)
- ⚙ PQN dating corrected from 2025 to 2024 in five draft files
  (Gallici et al. arXiv:2407.04811 is 2024)
- ⚙ MDP transition kernel in §II.A simplified from
  $P(s' \mid s, a, \theta)$ to $P(s' \mid s, a)$ — the $\theta$
  conflated environment and learned-approximator parameters
- Created `12-second-review-audit.md` with cross-reviewer status map
  (24 asks; 7 already covered, 3 quick-fixed this turn, ~14 needing
  new content sessions, ~5 cross-cutting)

paper.pdf: 1.46 MB.

Commit: `d64302b`

---

## v0.9 — 2026-05-30 — Late-2025 / 2026 paper sweep

Title updated to 2026 and five recent papers integrated from a May
2026 arXiv sweep. Sweep verdict: the field has *consolidated*
post-2024 rather than opened new directions, so the additions cover
the gap without padding.

- Title: "Understanding Q-Learning and Deep Q-Learning in 2025" → 2026
- §III methodology end-date: "1989-2025" → "1989-early 2026"
- §IV.A.B.1: **DDQL** (Nagarajan, White, Machado 2026) — revisits
  Double DQN ≠ classical Double Q-learning; two genuinely
  independent networks; beats DDQN on 47/57 Atari
- §IV.G.B.3: **SICQL** (Liu et al., ICLR 2026) + **ICQL** (Xu et
  al., ICLR 2026) — post-AdA in-context Q-learning thread now an
  established subfamily
- §IV.F.B.5 (new): **QFIX** (Baisero et al. 2025) — residual
  correction layer for VDN/QMIX/QPLEX, IGM-complete with simpler
  training
- §IV.E.B.5 (new): **FQL** (Park, Li, Levine, ICML 2025) — one-step
  flow-matching policy with Q-learning; avoids recursive backprop
- §IV.I.C.4: **Klein et al. 2026** plasticity-loss survey — natural
  successor to Lyle 2023 + Nikishin 2022

Commits: `b45bab2`, `6869cf9`, `139d9a5`

---

## v0.8 — 2026-05-30 — Session 3: Appendix B and reading guide

Closes Feedback 4 (too much derivation). Heavy derivations move out
of §IV body into a consolidated reference appendix.

- New Appendix B "Notation and Selected Derivations"
  (`B-notation-and-proofs.md`, ~2,300 words)
  - B.1 consolidated notation table
  - B.2 Smith & Winkler max-of-noisy-estimators bound
  - B.3 categorical distributional projection operator
  - B.4 Wasserstein contraction proof sketch (Bellemare 2017)
  - B.5 pessimism LCB argument for offline RL (Jin et al. 2021)
  - B.6 QPLEX IGM completeness (Wang et al. 2020)
  - B.7 Watkins & Dayan 1992 tabular convergence proof outline
  - B.8 pointers to canonical sources
- §I closing extended with a *Reading guide* paragraph naming
  Appendix B as the home of longer derivations

Commit: `f518a14`

---

## v0.7 — 2026-05-30 — Session 2: Atari-limitations gaps

Closes Feedback 3 (Atari is not enough). All five sub-asks now
covered.

- §V.H new — "Limitations of Atari as a Q-learning benchmark."
  Six inherent limits: determinism, discrete action space,
  single-task per episode, fixed environments, compute scale,
  visual-only observations
- §V.I new — "Newer benchmarks for Q-learning evaluation."
  Atari-100k, ALE-stochastic, ProcGen, NetHack, BSuite (each pinned
  to the axis it diagnoses); D4RL and SMAC cross-referenced
- §VII.A strengthened: Henderson et al. 2018, Engstrom et al. 2020
  cited; Hundal 2025 now situated in the reproducibility-crisis
  literature
- §VIII.C fourth open direction added — "Q-learning under
  real-world distribution shift": sim-to-real, OOD robustness,
  online policy correction

Commit: `3da697e`

---

## v0.6 — 2026-05-30 — Session 1: Modern-RL content gaps

Closes Feedback 5 (need more modern RL). Two structural additions.

- §IV.G.B.3 meta-learning expanded from ~3 paragraphs to a full
  mechanism survey across three families:
  - Optimization-based: MAML, Reptile-Q, ProMP
  - Context-based: PEARL, MQL
  - Forward-pass / in-context: AdA, Algorithm Distillation
- Implicit-vs-explicit meta-learning given its own subsection
- New §IV.I "Theoretical Foundations and Recent Advances"
  (`4i-theoretical-advances.md`, ~2,300 words). Five-part structure
  - A. The theoretical landscape
  - B. Foundational results (Watkins-Dayan 1992, deadly triad,
    GTD/TDC)
  - C. Recent advances (distributional Wasserstein contraction,
    finite-time bounds, offline pessimism, stability theory,
    QPLEX IGM completeness)
  - D. Evidence-theory interaction
  - E. Six open theoretical questions
  - F. Results-by-year comparison summary table

Commit: `9df016b`

---

## v0.5 — 2026-05-29 — Reviewer-feedback audit document

Created `10-reviewer-audit.md` as a structured audit of the IEEE TAI
reviewer asks (extracted from pitch deck pp. 17–24) against the
current draft. Each of the five feedback areas got its own
status-and-gap table with concrete next-step actions. Recommended
three sessions to close the open sub-asks.

Commit: `37c78c5`

---

## v0.4 — 2026-05-29 — Typography and rendering polish

PDF build pipeline matured through several rounds. Mermaid
quadrantChart styling, Latin Modern font, Table I raw LaTeX,
math-mode glyph fixes.

- Font: DejaVu Serif → Latin Modern (Computer Modern's unicode-
  modernized descendant) via xelatex
- Mermaid quadrantCharts: `.mermaid-config.json` (renamed from
  `.mermaid-filter.yaml`) makes mermaid-filter pick up the
  zinc-light palette
- Table I rewritten as raw LaTeX with explicit column widths so the
  Aspect column no longer wraps one-word-per-line
- Glyph compatibility fixes for Latin Modern: ↔ in Appendix A title
  swapped for "vs."; ●/○ in §VII coverage table wrapped in math
  mode ($\bullet$ / $\circ$); ε-greedy / ≈ / λ-returns wrapped in
  math mode across the body; ◯ U+25EF in §IV overview matrix swapped
  for $\circ$
- Two-column layout deferred: pandoc's default `\begin{longtable}`
  for markdown tables can't live inside a single column; real
  two-column requires `documentclass=IEEEtran` or a Lua filter

Commits: `3e93d9a`, `139d9a5`

---

## v0.3 — 2026-05-29 — Title page, abstract, inline tables

- New `0-metadata.md` — YAML metadata block produces a proper title
  page, abstract, impact statement, index terms via pandoc
- Running header (fancyhdr `\leftmark` showing current section
  name); TOC depth bumped to 3
- Eight inline tables added to the draft:
  - Table I → §III (legacy survey comparison; Ghasemi 2024/25 fifth
    comparator)
  - Tables II, III → §V (Atari per-game scores, both parts)
  - Tables IV, V, VI → §VI (FrozenLake, Taxi, CliffWalking)
  - Tables VII, VIII → §VII (repository coverage + pros/cons)

Commit: `ecbd095`

---

## v0.2 — 2026-05-29 — Figures, cross-refs, mermaid pipeline

- Figure captions: `render-mermaid.mjs` reads `%% caption:` from
  each mermaid block; pandoc's implicit_figures auto-numbers as
  "Figure N: …"
- Clickable cross-references: each top-level heading gets
  `{#sec-…}` anchors; perl pass in `build-pdf.sh` converts plain
  §I-§VIII references into markdown links
- Mermaid rendering pipeline finalized: beautiful-mermaid (zinc-
  light theme) for 4 supported flowchart blocks (via
  `render-mermaid.mjs`), mermaid-filter for the 5 quadrantCharts
- CSS var() and color-mix() pre-resolution in the SVG so sharp /
  libvips renders the zinc-light shades correctly

Commits: `04f91d2`, `2587d93`, `68f2e10`

---

## v0.1 — 2026-05-29 — Build pipeline + PDF

- `build-pdf.sh` script wraps the pandoc / xelatex / mermaid-filter
  pipeline
- First end-to-end paper.pdf rendered
- Inserted master genealogy + modern-RL subgraph into
  `draft/4-overview.md` (§IV intro); inserted exploration-branch
  genealogy into §IV.C
- Axis × mechanism-family matrix added as a markdown table in §IV
  overview

Commits: `66b1497`, `b3f8660`

---

## v0.0 — 2026-05-29 — Initial structural pivot

The first round of work: take the original IEEE TAI draft and
reorganize it around the eight foundational weaknesses of vanilla
Q-learning. Every section drafted from scratch in markdown.

- Analysis and planning artifacts: `01-pitch-analysis.md`,
  `03-new-outline.md`, `04-method-remap.md`,
  `06-genealogy-figure.md`, `07-prior-art-sweep.md`,
  `08-figure-proposals.md`, `09-section-notes.md`
- Paper sections drafted: §I Introduction, §II Background with the
  Eight Weaknesses formal statement, §III Methodology, §IV overview
  + §IV.A-H (the eight axis sections), §V Atari benchmark analysis,
  §VI Tabular empirical, §VII Repositories, §VIII Conclusion,
  Appendix A Legacy Indexer
- Three structural decisions established as load-bearing claims:
  - Eight-axis problem-first organization replaces the six-category
    method-type taxonomy
  - Distributional RL relocated from "Statistical" (uncertainty) to
    §IV.D (credit assignment)
  - Dueling DQN relocated from "Q-Function Computation" to §IV.H
    (stability mechanism), defended via Rainbow ablation
- Prior-art sweep confirms the problem-first multi-axis angle is
  open in the 2024–2026 literature
- 18 modern-RL methods added that have no row in the original
  six-category taxonomy (CQL, IQL, BCQ, EDAC, AWAC, VDN, QMIX,
  QPLEX, QTRAN, Ape-X, R2D2, Agent57, MAML-Q, HER, RND, Go-Explore,
  Maximin Q, REDQ)

Commits: `97b3b06` through `6869cf9` (initial sweep), then
session-tagged commits onward.

---

## Conventions

- **Versions** are tagged `v0.N` rather than semver. The manuscript
  has not been submitted; semver applies to released artifacts.
- **Dates** are ISO `YYYY-MM-DD` and reflect the working session,
  not when each individual edit was made.
- **Commit hashes** point to the load-bearing commit that introduced
  the change. Multi-commit sessions list the final one.
- **Status markers** in entries:
  - ⚙ verifiable inconsistency fix
  - ⊕ structural / load-bearing change
  - ⊘ new content addition
  - ✓ closes a previously-open audit item
- **What this file is *not*:** a substitute for git history. For
  per-file diff detail, use `git log --follow <file>`.

To add an entry, append a new `## v0.N+1 — DATE — short title`
section at the top of the file (after this introduction). Keep
entries skimmable; longer rationale belongs in the audit files or
commit messages.

**Update cadence:** after every commit that rebuilds `paper.pdf`,
add or amend a version entry recording (a) what changed, (b) the
new PDF size, (c) the commit hash. Multi-commit sessions can be
bundled under a single version when the intermediate commits don't
each ship a PDF; the version entry then names the final commit.
