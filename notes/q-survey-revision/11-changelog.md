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

(Commit pending at time of writing this entry.)

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
