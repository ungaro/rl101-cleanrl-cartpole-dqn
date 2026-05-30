# Q-Survey Revision — Working Notes (folder index)

Internal planning notes and source for the IEEE TAI submission
*"Understanding Q-Learning and Deep Q-Learning in 2025:
A Methodological and Empirical Survey."*

> **Synced to v0.24 (2026-05-30).**
> **Venue: IEEE Transactions on Artificial Intelligence (TAI)** —
> "Original Research Review Manuscript."
> **Status: SUBMISSION-READY** — main **19 pp** two-column IEEEtran
> + separate **10 pp** supplement (S1–S6). All automated hygiene
> checks pass (`\anontrue`, no identity leakage, abstract 179 w,
> title 12 w, impact 135 w, 5 keywords, no acks/funding).
> Remaining work is user-side: pick keywords from the TAI dropdown,
> run iThenticate (≤20% target), add ORCIDs, and flip `\anonfalse`
> only for camera-ready.

All content here lives on branch `alp/q-survey-revision-prep` and
is **not merged to main** — this is exploratory revision work, not
course material. This is an INTERNAL notes folder: meta-commentary,
distillation history, and references to the monograph are fine here
(none of that leaks into the submitted PDFs).

---

## TAI hard constraints (the box we built to)

Review papers: **15 pp normal / 21 pp max** ($200/page over 15,
mandatory); **two-column IEEEtran**; **double-anonymous** review;
abstract ≤250 w; impact statement 100–150 w; title ≤15 w; 3–6
keywords (TAI dropdown); **explicit systematic methodology
required**; ≤20% iThenticate similarity.

---

## Two editions + supplement

The work now exists as a distilled submission carved out of a
larger frozen source — *"distill into a lens, don't delete."*

- **`draft-monograph/`** — the comprehensive ~47-page single-column
  edition, **FROZEN** (git tag `monograph-v0.15`). The "quarry"
  every cut paragraph is preserved here and curated into the
  supplement rather than discarded.
- **`draft-tai/`** — the **SUBMISSION**: 19-page two-column
  IEEEtran, anonymized via an `\ifanon` toggle (`\anontrue` for
  review; the named author list — Colby Wang, Ti, Divya, Kevin,
  Hamna, Logan, Eason Yishan Wu, Charles Jiahao Zhang, Alp
  Guneysel — is preserved in the `\else` branch for camera-ready).
  Plus the **10-page supplement** (`supplement.pdf`, sections
  S1–S6).

**Distillation arc** (page counts): 47 → 34 (§IV.A–H) → 26 (§IV.I/J
demoted + appendices → supp) → 27 (cross-axis table) → 21 (§V/§VII
distilled, §VI re-run) → **19** (front-matter compression). See
`13-tai-compression-ledger.md` for the page-by-page accounting.

---

## Folder layout

```
notes/q-survey-revision/
├── 00-README.md                ← this file (folder entry point / index)
│
├── draft-tai/                  ← THE SUBMISSION (19 pp main + 10 pp supp)
│   ├── main.tex / body.tex         main paper (§I–§VIII)
│   ├── main.pdf                    built main paper
│   ├── 1-introduction.md … 8-conclusion.md   per-section markdown source
│   │                               (incl. 4a–4j axis/theory/foundation sections)
│   ├── supplement.tex / supp-body.tex   supplement wrapper + body
│   ├── supplement.pdf              built supplement
│   ├── supp-S1-search-log.md … supp-S6-experiment-details.md   supp sources
│   ├── build-tai.sh                main build (pandoc→LaTeX → main.pdf)
│   ├── build-supp.sh               supplement build (→ supplement.pdf)
│   ├── tables-twocol.lua           pandoc filter: longtable → table* (2-col)
│   ├── render-mermaid.mjs          figure renderer
│   ├── refs.bib (~136 entries) + IEEEtran.cls/.bst
│   └── data/tabular_results.json   committed §VI experiment results
│
├── draft-monograph/            ← FROZEN extended edition (tag monograph-v0.15)
│   └── 0-metadata.md … B-notation-and-proofs.md, build-pdf.sh, paper.pdf
│
├── scripts/tabular_experiments.py  ← reproducible §VI experiment (writes
│                                     draft-tai/data/tabular_results.json)
│
└── (planning docs 01–15, indexed below)
```

**Build pipeline.** `build-tai.sh` → `main.pdf` and `build-supp.sh`
→ `supplement.pdf`. Both run pandoc → LaTeX fragment with
`--natbib` (IEEE `[N]` citations via `IEEEtran.bst`) and the
`tables-twocol.lua` filter (longtable → `table*`), then system
xelatex/bibtex over `refs.bib`. The `\ifanon` toggle in `main.tex`
governs anonymization. The §VI experiment is reproducible via
`scripts/tabular_experiments.py` → `data/tabular_results.json`.

> Note: a legacy `draft/` directory (the pre-split working copy) may
> still be present on disk; it is superseded by `draft-monograph/`
> and is not part of the canonical layout.

---

## Planning-doc index

One line each — consult as needed.

- **`00-README.md`** — this file; folder entry point / index and
  current-state snapshot.
- **`01-pitch-analysis.md`** — the analytical brief: reviewer
  feedback root-cause read and the three original proposals
  (A problem-first reorg; B centerpiece figures; C repo spin-off).
- **`03-new-outline.md`** — proposed table of contents / section
  spine that became the §I–§VIII structure.
- **`04-method-remap.md`** — working notes mapping legacy
  categories onto the weakness axes (now the method-type taxonomy /
  supplement S2 index).
- **`06-genealogy-figure.md`** — genealogy figure design (ASCII +
  mermaid) for the §IV overview.
- **`07-prior-art-sweep.md`** — 2024–2026 competing-survey check
  (defends the axis spine; feeds Table I + supp S1 overlap).
- **`08-figure-proposals.md`** — figure / comparison-artifact
  catalogue (genealogy, branches, axis×mechanism matrix,
  cross-axis interaction table).
- **`09-section-notes.md`** — per-section bibliography / scope notes.
- **`10-reviewer-audit.md`** — first-round IEEE TAI reviewer audit
  with post-execution status.
- **`11-changelog.md`** — versioned history of draft changes
  (through v0.24; records the monograph freeze and distillation).
- **`12-second-review-audit.md`** — second-round reviewer audit.
- **`13-tai-compression-ledger.md`** — page-by-page accounting of
  the 47 → 19 pp distillation and what moved to the supplement.
- **`14-llm-review-triage.md`** *(new)* — triage of three LLM
  reviews of v0.14: what was adopted into core, routed to
  supplementary, or declined/bounded.
- **`15-submission-readiness.md`** *(new)* — final hygiene checklist
  (anon, word/page counts, keywords, leakage) + user-side TODO.

---

## Final structure of the submission (draft-tai)

- **§I Introduction** — framework-first; five contributions;
  problem-first reframe.
- **§II Background** — MDP/Q-learning formalism + eight weaknesses
  **W1–W8** (one sentence each; W7 is a COMPOSITE axis: W7a sample
  throughput + W7b slow adaptation).
- **§III Methodology** — explicit **systematic/PRISMA** protocol
  (Google Scholar, arXiv cs.LG/cs.AI, Semantic Scholar; search
  strings; 5 inclusion/exclusion criteria; screening ~200→120→80);
  Table I prior-survey comparison (six dimensions vs five surveys).
- **§IV "Q-Learning Methods by Weakness"** (retitled from "Related
  Works") — opens with the **method-type taxonomy** defined once via
  a table (six categories → axes); "legacy" terminology killed.
  Overview adds genealogy figure, branches figure, axis×mechanism
  matrix, and the **cross-axis interaction table** (W1–W8 origin /
  principal interaction / deployment failure mode). §IV.A–H use a
  uniform compact template (Weakness → Mechanisms → Trade-off →
  Open questions → one comparison table; run-in **bold** lead-ins).
  §IV.I theoretical advances (~250 w; proofs → supp S5). §IV.J
  foundation-model alignment (~336 w, framed as EMERGING:
  Q-Transformer, VLM-Q, ShiQ, Q♯, SICQL/ICQL, Q-shaping).
- **§V Atari** — diagnostic synthesis (task-category × axis);
  rliable/point-estimate caveat kept; full per-game tables → supp S3.
- **§VI Tabular** — reproducible experiment (Q-learning / SARSA /
  Expected SARSA / 3-step Q over **100 seeds + 95% bootstrap CIs**
  on FrozenLake/Taxi/CliffWalking; VI/PI/MPI/CVPI reported as a
  separated **planning oracle** / upper bound, not a model-free
  competitor).
- **§VII Repositories** — six repos (Tianshou, XuanCe, CleanRL, DQN
  Zoo, SB3, RLlib) by axis; **nine methods absent from all six**;
  Hundal non-interchangeability; full matrix → supp S4.
- **§VIII Conclusion** — per-axis summary; community-repository
  proposal; open directions; limitations & ethics.
- Appendices A/B moved to the supplement (S2 method-type index,
  S5 notation/proofs).

**Supplement (S1–S6):** S1 search log + prior-art overlap; S2 full
~50-method method-type index; S3 full Atari per-game tables; S4 full
repository coverage matrix; S5 notation + 7 derivations/proofs; S6
tabular-experiment config + full results (±std, CIs).

---

## How to hand off to the team

Point co-authors at this README first, then `01-pitch-analysis.md`
(the "why"), then `13-tai-compression-ledger.md` and
`14-llm-review-triage.md` to see what was cut/kept and why. The
canonical artifacts to review are `draft-tai/main.pdf` and
`draft-tai/supplement.pdf`. The monograph (`draft-monograph/`,
tag `monograph-v0.15`) is the frozen reference for anything that was
distilled out.
