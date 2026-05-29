# Q-Survey Revision — Working Notes

Working notes and a markdown draft for revising the IEEE TAI
submission *"Understanding Q-Learning and Deep Q-Learning in 2025:
A Methodological and Empirical Survey."*

All content here lives on branch `alp/q-survey-revision-prep` and
is **not merged to main** — this is exploratory revision work, not
course material.

---

## Reading order for reviewers

1. **Start with `01-pitch-analysis.md`** — the analytical brief.
   What the reviewers asked for, the root-cause read of the
   feedback, and the three proposals (A: problem-first
   reorganization; B: centerpiece figures; C: Q-learning repo
   spin-off). Status of each proposal at the top.

2. **Then `03-new-outline.md`** — the proposed table of contents
   for the revised paper, with per-section status markers and
   links to drafted files.

3. **Then `draft/`** — the actual revised paper prose, organized
   one file per section.

The other planning files (`04`, `06`, `07`, `08`, Appendix A) are
reference material to consult as needed.

---

## Layout

```
notes/q-survey-revision/
├── 00-README.md            ← this file
├── 01-pitch-analysis.md    ← analysis & direction
├── 03-new-outline.md       ← outline / TOC with status
├── 04-method-remap.md      ← working notes; superseded by Appendix A
├── 06-genealogy-figure.md  ← genealogy figure design (ASCII + mermaid)
├── 07-prior-art-sweep.md   ← 2024–2026 competing-surveys check
├── 08-figure-proposals.md  ← figure / comparison artifact catalogue
└── draft/
    ├── 1-introduction.md       (§I)
    ├── 2-background.md         (§II + 8 weaknesses)
    ├── 3-methodology.md        (§III)
    ├── 4a-overestimation-bias.md  (§IV.A, W1)
    ├── 4b-sample-inefficiency.md  (§IV.B, W2)
    ├── 4c-brittle-exploration.md  (§IV.C, W3)
    ├── 4d-reward-sparsity.md      (§IV.D, W4)
    ├── 4e-distribution-shift.md   (§IV.E, W5) ← new offline-RL
    ├── 4f-multi-agent.md          (§IV.F, W6) ← new multi-agent
    ├── 4g-scaling-adaptation.md   (§IV.G, W7) ← new distributed/meta
    ├── 4h-stability.md            (§IV.H, W8)
    ├── 5-atari-benchmarks.md      (§V)
    ├── 6-tabular-empirical.md     (§VI)
    ├── 7-repositories.md          (§VII)
    ├── 8-conclusion.md            (§VIII)
    └── A-legacy-indexer.md        (Appendix A — legacy ↔ axis map)
```

Totals: ~15,500 words of paper prose across 16 draft files;
~5,000 words of planning artifacts across 7 top-level files.

---

## Current status

| Workstream | Status |
|---|---|
| §I Introduction | drafted |
| §II Background + 8 Weaknesses | drafted |
| §III Methodology | drafted |
| §IV.A–H (all eight axis-sections) | drafted (with per-section comparison artifacts) |
| §V Atari benchmarks | drafted (preserves legacy row grouping) |
| §VI Tabular empirical | drafted (preserves Tables IV–VI) |
| §VII Repositories | drafted (preserves Tables VII/VIII + Hundal 2025 positioning) |
| §VIII Conclusion | drafted (promotes repo spin-off to named deliverable) |
| Appendix A — legacy ↔ axis indexer | drafted |
| Per-section comparison tables (Part A.1) | drafted in each axis-section |
| Per-section 2D positioning grids (Part A.2) | drafted in mermaid for §IV.A, §IV.C, §IV.D, §IV.E, §IV.H |
| Decision tree for §IV.E | drafted in mermaid |
| Genealogy figure (ASCII design) | drafted; mermaid companion in `06-genealogy-figure.md` |
| Axis × method-family matrix (paper-level F-B1) | design only; TikZ rendering pending |
| Axis-stratified Atari grouped bars (F-B2) | proposed; data assembly pending |
| Complexity–performance scatter (F-B4) | proposed; data assembly pending |
| Compute log-log (F-C1) | drafted as mermaid quadrant in §IV.G |
| D4RL degradation curve (F-C2) | proposed; data extraction pending |
| Q-learning repo spin-off (Suggestion C) | **pending** — stub not yet stood up |
| Bibliography assembly (~25 new entries) | pending |
| LaTeX integration into team's `.tex` source | pending |

---

## Three structural decisions that need team validation

Before LaTeX integration, three claims merit explicit team review:

1. **The eight-axis problem-first reorganization** as the
   replacement spine for §IV. Argued in `01-pitch-analysis.md` and
   §II.B; defended against prior art in `07-prior-art-sweep.md`.

2. **Two contested method reinterpretations** that affect §IV
   structure:
   - Distributional methods (C51 / QR-DQN / IQN / FQF) treated
     as credit-assignment (§IV.D) rather than uncertainty (the
     original draft's §IV.A "Statistical Methods" placement).
   - Dueling DQN treated as a stability mechanism (§IV.H) rather
     than a Q-function-computation mechanism (the original
     draft's placement).

   Both reinterpretations are argued explicitly in their respective
   axis-sections with Rainbow ablation evidence; reviewers may
   push back.

3. **The three modern-RL additions** (§IV.E offline, §IV.F
   multi-agent, §IV.G distributed) cover methods entirely new to
   the existing draft. Each section needs review by a co-author
   with domain background before integration. Per-method technical
   claims have been cross-checked against original papers, but
   framing decisions (the four-family taxonomy in §IV.E, the
   four-method canonical set in §IV.F, the
   distributed/recurrent/meta tripartite in §IV.G) are first-pass.

---

## What's preserved from the original draft

- Tables II and III (Atari extracted scores) — row grouping by the
  six legacy categories, column grouping by task category.
- Tables IV, V, VI (tabular empirical evaluation) — preserved
  structurally with axis-attribution prose added.
- Tables VII and VIII (repository comparison) — preserved with
  axis-aware annotation.
- The five distinguishing contributions in Table I — reframed but
  preserved in number and intent.
- All per-method writeups in the original §IV — re-homed to new
  axis-sections, with the *prose paragraphs themselves preserved
  nearly verbatim* (the structural change is which subsection each
  paragraph lives in, not the paragraph content).

The original taxonomy is not discarded; it lives on as Tables
II/III's row spine and as Appendix A. The eight-axis structure adds
a third lens (analytical organization in §IV prose) without
removing the existing two.

---

## How to handoff to the team

When sending for review, point co-authors at:

- This README first
- Then `01-pitch-analysis.md` (the "why")
- Then `03-new-outline.md` (the "what")
- Then specific axis-sections in `draft/` (the "how")

Flag the three structural decisions above explicitly. Domain
owners (offline RL, multi-agent, distributed) should be assigned
review of §IV.E, §IV.F, §IV.G respectively before LaTeX
integration begins.
