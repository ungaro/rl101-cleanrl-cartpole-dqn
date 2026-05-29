# Q-Learning Survey — Direction Analysis

## The core read of the reviewer feedback

The five reviewer points (algorithms feel isolated; need conceptual
insight; Atari is limited; too much derivation; need more modern RL)
all rhyme with **one structural complaint**:

> *The paper is organized as a catalogue. We want analysis.*

Every individual feedback point is a symptom of the catalogue
organization, not an independent problem:

| Surface complaint | Root cause |
|---|---|
| "Algorithms feel isolated" | Methods are presented in independent buckets with no axis of comparison |
| "Need conceptual insight" | The catalogue structure doesn't surface *why* a method exists |
| "Atari isn't enough" | Benchmark section is descriptive (what scores were obtained) rather than analytical (what does Atari fail to test) |
| "Too much derivation" | Derivation fills space that synthesis should occupy |
| "Need more modern RL" | The six-category taxonomy is built around 1989–2018 methods; modern RL doesn't fit the categories cleanly |

Fixing surface complaints one by one will produce a *bigger* catalogue.
Fixing the root cause means **reorganizing around a different axis** —
and once the axis is right, the surface complaints largely resolve
themselves.

---

## What the draft already does well

It's worth being precise about what's load-bearing in the current
version, because the structural pivot below preserves it rather than
discarding it.

- **"Critical Reflections" subsections already exist** at the end of
  each Related Works category. The bones of synthesis are there. The
  problem is the content — most reflections read as *open question
  prompts* ("future work could explore X") rather than substantive
  analysis, and they're trapped inside their method-type bucket so
  they cannot make cross-bucket comparisons.
- **The repository comparison (Tables VII/VIII) is genuinely useful.**
  Table VII's coverage matrix is exactly the kind of concrete
  synthesis the reviewers want; it just isn't *linked* to the method
  discussion.
- **Tabular benchmarks (Tables IV–VI)** are real first-party
  contributions and should stay where they are structurally.

The structural pivot below is an *upgrade* to the synthesis layer, not
a rewrite of the empirical layer.

---

## The modern-RL gap is structural, not cosmetic

Reviewer point 5 ("need more modern RL") looks like a small ask. The
draft makes clear it isn't:

| Modern Q-learning area | In the draft? |
|---|---|
| Offline RL (CQL, IQL, BCQ, EDAC) | **No** |
| Multi-agent (QMIX, VDN, QPLEX) | **No** |
| Distributed scale (Ape-X, R2D2, Agent57) | **No** |
| Meta-RL / continual Q-learning | **No** |
| Hindsight Experience Replay (HER) | **No** |
| World-models with Q-targets (MuZero family) | **No** |

The newest non-2025 deep-RL entries are MeDQN (2023) and PSDQN (2023).
Post-2018 deep-RL coverage is essentially two papers. CBDQ (2025) and
PQN (2025) round out the modernity story, but neither represents the
directions the field actually moved.

"Add a Modern RL appendix" is not enough — the current taxonomy has no
slot where offline RL or multi-agent value decomposition belongs.
This is what makes the structural pivot below *load-bearing* rather
than ornamental: those families naturally land in *distribution shift*
and *coordination* buckets that the current taxonomy doesn't admit.

---

## Suggestion A (high leverage, structural): pivot the organizing axis from method type to problem solved

The current taxonomy has six method-type buckets (Statistical,
Q-Function Computation, Memory/Replay, Ensemble, Model-Based, Minimal).
This is a *botanist's* taxonomy — it tells you what *kind* of thing
each method is. The reviewers want a *physician's* taxonomy — what
*symptom* does each method treat.

Proposed alternative organizing axis: **the foundational weaknesses of
vanilla Q-learning that modern methods address**. Sketch:

| Weakness of vanilla Q | Methods that target it |
|---|---|
| **Overestimation bias** (max operator + noisy estimates) | Double Q, Double DQN, Maximin Q, Weighted Double Q, EBQL, REDQ (under-est. pushback) |
| **Sample inefficiency** (replay uniformity, single trajectory) | PER, Rainbow, Ape-X, R2D2, HER, DQfD, MeDQN |
| **Brittle exploration** (ε-greedy is dithered) | NoisyNet, Bootstrapped DQN, IQN, UCB-ensembles, CBDQ, RND, Go-Explore |
| **Reward sparsity / credit assignment** | Distributional (C51, QR-DQN, IQN, FQF), n-step, λ-returns |
| **Distribution shift** (online → offline) | CQL, IQL, BCQ, BRAC, EDAC |
| **Coordination** (single-agent → multi-agent) | VDN, QMIX, QPLEX, QTRAN |
| **Slow adaptation** (per-task training) | Meta-Q (MAML on Q), Reptile-Q, distillation-based transfer |
| **Function-approximation instability** (the deadly triad) | Target networks, Polyak averaging, dueling decomposition, PQN's recipe |

Each section becomes:

1. **The weakness** — formal statement, why it's hard, what it costs.
2. **Solution families** — grouped by mechanism, compared head-to-head.
3. **Trade-offs introduced** — every fix has a cost; name it.
4. **Empirical evidence** — what ablation/benchmark data we have.
5. **Open question** — what's still unsolved in this dimension.

**Why this is the right pivot:**

- Directly addresses reviewer points 1 and 2 (synthesis + conceptual
  insight). The structure forces cross-method comparison — Double Q
  vs. Maximin vs. REDQ all share a section.
- Pulls modern RL into the existing structure rather than tacking it
  on. *Distribution shift* is the natural home for offline RL.
  *Coordination* is the home for multi-agent value decomposition. No
  "Modern RL" appendix needed.
- Addresses "Atari is limited" indirectly: a *problem-first* paper
  doesn't lean on Atari as the spine. The dashes in Tables II/III stop
  being apologies and start being evidence — Montezuma at 0 across the
  board *is* the brittle-exploration argument. Classic control becomes
  evidence for *stability*; offline benchmarks (D4RL) become evidence
  for *distribution shift*. The benchmark landscape becomes coherent.
- Justifies cutting derivations: a problem-first section is about the
  *mechanism* of the fix, not the formal derivation. Derivations move
  to a single appendix.

**Cost:** ~3–4 weeks of distributed work with 3+ contributors, gated
on lining up domain owners for offline RL, multi-agent, and distributed.

**Risk to flag:** if the draft is already in late revision, the cost
may be higher than the team can absorb in one cycle. The genealogy
figure in Suggestion B captures most of the synthesis benefit without
rewriting the prose.

---

## Suggestion B (medium leverage, visual): add a centerpiece *design-space* figure

If we cannot afford a full structural pivot, the next-highest-leverage
move is a **single well-designed figure that does the synthesis work**.
The reviewers complain about isolated methods because they have no
visual anchor for how methods relate. One good figure makes the
relationships visible at a glance.

Two candidates:

### B1. Q-learning *genealogy tree*

A directed graph of methods, with each edge annotated by the weakness
the child method addresses in the parent. Annotated edges *are* the
synthesis the reviewers want. The absence of CQL/IQL/QMIX appears as a
visible empty region in the figure — itself an argument for the
structural pivot.

(ASCII first pass in `06-genealogy-figure.md`.)

### B2. Q-learning *design-space scatter plot*

A 2D figure with axes like:
- x: sample efficiency (e.g. median DQN-normalized score at 10M frames)
- y: implementation complexity (or: number of components on top of vanilla DQN)
- color: family
- size: typical compute budget required

Each method is a point. The clusters make the relationships visible:
"Rainbow is high-complexity, high-efficiency; minimal Q-learning is
low-complexity, low-efficiency; ensembles are high-complexity,
medium-efficiency."

**Recommendation:** start with B1 (the genealogy). It's the cheaper of
the two, it doubles as the framing artifact for Suggestion A (the
discussion of "which edge label goes where" *is* the seed of the
structural pivot), and it makes the modern-RL gap visible.

---

## Suggestion C (low cost, high signal): spin off the Q-learning repo the Conclusion already promises

The current Conclusion contains:

> "Looking forward, a clear avenue for future research lies in the
> development of a dedicated, community-maintained repository focused
> exclusively on Q-learning and its deep variants."

This is a real spin-off opportunity. Even if the repo is just a
coverage-tracker plus stubs for the missing variants
(Bootstrapped DQN, EBQL, UCB Ensemble, PSDQN, MeDQN, CBDQ — *none* of
which are in any of the six surveyed repos per Table VII), the act of
building it converts the survey's findings into a community
deliverable. Costs ~1 day for the stub.

**First-week deliverable:** GitHub repo skeleton, README is Table VII,
tracking issues for the six absent variants with "wanted:
implementation" tags.

---

## Strategic recommendation

- **A (structural pivot) — done.** All eight axis-sections plus §I,
  §II, §III, §IV overview, §V–§VIII, and Appendix A are drafted in
  `draft/`. See `00-README.md` for the full status map.
- **B (genealogy figure) — done.** Design source in
  `06-genealogy-figure.md`; mermaid renderings in
  `draft/4-overview.md` (master genealogy + modern-RL subgraph) and
  `draft/4c-brittle-exploration.md` (exploration branch). Mermaid
  `quadrantChart` 2D positioning grids in §IV.A, §IV.C, §IV.D,
  §IV.E, §IV.H. Final TikZ/PGF rendering for the IEEE template is
  the only pending sub-item.
- **C (repo spin-off) — pending.** A named deliverable in
  `draft/8-conclusion.md` with four design priorities and a
  prioritized roadmap (nine methods absent from all six surveyed
  repositories per Table VII). The repository itself is not yet
  stood up — that's the remaining concrete action.

---

## Open questions for the team before committing to a direction

1. **How locked is the taxonomy?** If the six categories are
   load-bearing in the cover letter or already accepted by the editor
   as the paper's contribution, the structural pivot becomes more
   expensive.
2. **Is there appetite for a co-author taking ownership of modern
   families?** Offline RL alone (CQL/IQL/BCQ/EDAC) is a real lift; a
   PhD researcher with offline-RL chops would be the right owner.
3. **What's the editor's tone?** "Major revision" vs. "reject and
   resubmit" maps to different ambition levels.
4. **Should the repository comparison stay as a standalone section?**
   It currently reads like a sidebar. Alternative: fold each
   repository's choices into the method sections they implement —
   one-line callouts ("supported in Tianshou, XuanCe; absent from
   CleanRL, SB3"). Reproducibility-by-axis, which the reviewers
   explicitly want.

---

## Differentiation check against 2024–2026 prior art

A focused web sweep (Google Scholar + arXiv cs.LG/cs.AI + Semantic
Scholar) was run to check whether other recent surveys threaten the
differentiation claims. Verdict: **the problem-first reorganization
angle is open.** No 2024–2026 Q-learning / DQN survey takes that
organizing structure; no paper unifies tabular + deep Q-learning into
a problem-axis framework. The structural pivot in Suggestion A is a
clean differentiation claim.

Three citations need to be added to the revision regardless of which
suggestion the team adopts:

- **Ghasemi et al. 2024/2025** (arXiv:2411.18892) — closest
  competitor, broad RL survey organized by method family. The
  proposed finer taxonomy + repo + Atari analysis differentiates.
- **Springer NCAA 2026** offline-RL distribution-shift survey
  (10.1007/s00521-026-11966-8) — single-axis problem-first precedent
  to cite at the head of §IV.E.
- **Hundal et al. 2025** (arXiv:2503.22575) — empirical
  reproducibility audit of RL libraries on PPO. Not a survey, but
  directly adjacent to our repository-comparison contribution.
  Requires one sentence in §VII distinguishing the angles.

Full findings, borderline cases, and follow-up checks in
`07-prior-art-sweep.md`.

---

*Companion files (all on `alp/q-survey-revision-prep`, not merged to main):*

- `03-new-outline.md` — problem-first table of contents
- `04-method-remap.md` — working notes for the legacy ↔ axis mapping; canonical version is `draft/A-legacy-indexer.md`
- `06-genealogy-figure.md` — genealogy figure design source (ASCII + mermaid)
- `07-prior-art-sweep.md` — 2024–2026 competing-surveys check
- `08-figure-proposals.md` — figure / comparison artifact catalogue
- `09-section-notes.md` — per-section material notes (bibliography, citations, scope decisions)
- `draft/` — the paper itself, one file per section

---

## Status snapshot (suggestion-by-suggestion)

- **Suggestion A — problem-first structural pivot:** Drafted in full.
  See `draft/1-introduction.md`, `draft/2-background.md`, all eight
  `draft/4*.md` axis-sections, `draft/4-overview.md`,
  `draft/5-atari-benchmarks.md`, `draft/6-tabular-empirical.md`,
  `draft/7-repositories.md`, `draft/8-conclusion.md`, and
  `draft/A-legacy-indexer.md`. Differentiation against 2024–2026
  prior art confirmed in `07-prior-art-sweep.md`.
- **Suggestion B — centerpiece figure:** Genealogy figure design
  drafted in `06-genealogy-figure.md`; mermaid renderings inserted
  into `draft/4-overview.md` (master genealogy + modern-RL branches)
  and `draft/4c-brittle-exploration.md` (exploration branch). Mermaid
  `quadrantChart` 2D positioning grids inserted per-section in §IV.A,
  §IV.C, §IV.D, §IV.E, §IV.H. Final TikZ/PGF rendering for the paper
  itself is pending.
- **Suggestion C — Q-learning repo spin-off:** Promoted to a named
  deliverable in `draft/8-conclusion.md` with four design priorities
  and a prioritized roadmap (nine methods absent from all six
  surveyed repositories per Table VII). Repository stub not yet
  stood up — outstanding action item.
