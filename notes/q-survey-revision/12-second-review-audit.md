# Second-Round Reviewer Audit

Two new reviewer reports on a prior version of the manuscript surfaced
after Sessions 1–3. This file audits them against the current state
of the draft, distinguishing:

- ✓ already closed (often by Sessions 1–3 or the late-2025/2026 sweep)
- ⚙ quick fix (verifiable inconsistency, addressed in this turn)
- ⊘ new content gap (needs a focused session)
- ⊕ new structural concern (architectural, beyond a single session)

Both reviewers keep the eight-axis problem-first framework as the
paper's central contribution — neither asks the team to abandon it.
The substantive asks are all *content additions* and *methodological
tightening*, not a structural retreat.

---

## Reviewer 1 — "Peer Review Report" (verdict: conditional accept, major revisions)

Reviewer 1 is broadly enthusiastic about the structural pivot and
treats the asks as additions to strengthen an already-publishable
draft. Their twelve substantive asks, mapped to the current draft:

| # | Ask | Status | Note |
|---|---|---|---|
| 1 | QFIX superseding QPLEX for IGM-completeness | ✓ | Added in §IV.F.B.5 during the late-2025/2026 sweep |
| 2 | Cognitive Belief-Driven Q-Learning | ✓ | CBDQ is already in §IV.C.B.3 (we always had it) |
| 3 | Plasticity loss / capacity crisis | ✓ | Lyle 2023 + Nikishin 2022 + Klein 2026 plasticity-loss survey covered in §IV.I.C.4 and §IV.H |
| 4 | Flow-matching policies with Q-learning | ✓ | FQL (Park, Li, Levine 2025) added in §IV.E.B.5 during the sweep |
| 5 | Q-learning with Adjoint Matching | ⊘ | Distinct from FQL — uses continuous adjoint method instead of one-step flow; should be added alongside FQL in §IV.E.B.5 |
| 6 | Predictable scaling laws for value-based DRL (Rybkin et al.) | ⊘ | Significant gap. The Rybkin scaling-law result directly contradicts our §IV.G framing that "single-learner Q is sample-bottlenecked"; needs integration |
| 7 | Q-learning for LLM/VLM alignment (ShiQ, VLM Q-Learning, Q♯) | ⊘ | Significant gap. This is the largest single content addition; could be a new §IV.J or expanded §IV.G subsection |
| 8 | Parameterized Q-Networks for hybrid discrete-continuous action spaces | ⊘ | Moderate gap. Most natural home: §IV.A or a new subsection on action-space-induced bias |
| 9 | Hamilton-Jacobi-Bellman continuous-time Q-learning | ⊘ | Minor; one-paragraph mention in §IV.I or §IV.E |
| 10 | Crossing-quantiles problem and non-crossing QR-DQN variants | ⊘ | Moderate gap in §IV.D; one paragraph + non-crossing-variant citation |
| 11 | Transfer learning + feature-correlation regularization for overestimation | ⊘ | Minor; one paragraph in §IV.A |
| 12 | Branching parameterized Q-networks for compound overestimation | ⊘ | Companion to #8; bundled |

**Reviewer 1 verdict:** of twelve asks, **four are already done** (mostly by the late-2025 sweep), one is moderate (#10), and seven are new content additions. The biggest single ask is #7 (LLM/VLM alignment).

---

## Reviewer 2 — "Reviewer Report" (verdict: major revision / R&R; lean reject if binary)

Reviewer 2 is more critical and focuses on scholarly apparatus, not
content. Their asks are organized into bibliographic, methodological,
and taxonomic categories.

### Bibliographic / scholarly apparatus (highest priority)

| Ask | Status | Note |
|---|---|---|
| References list / bibliography missing | ⊕ | Already on our outstanding list (`10-reviewer-audit.md` cross-cutting concerns #1). Pandoc-citeproc + `.bib` integration is the path; haven't done it yet |
| Audit every citation, year, venue | ⊕ | Same — needs the bibliography first |
| Fix internal count error ("Eight methods" vs nine listed) | ⚙ | §VII.C says "Eight" but lists nine. Fixed in this turn |
| PQN dating: 2025 vs 2024 | ⚙ | Gallici et al. *Simplifying Deep Temporal Difference Learning* (arXiv:2407.04811) is 2024. Tagged as 2025 in five draft files. Fixed in this turn |
| MDP notation $P(s' \mid s, a, \theta)$ — $\theta$ doesn't belong in environment kernel | ⚙ | §II.A line 11. Fixed in this turn |

### Methodological transparency (PRISMA-style)

| Ask | Status | Note |
|---|---|---|
| Search databases, queries, screening counts | ⊘ | §III.A names five inclusion criteria but no search-string protocol, screening flow, or PRISMA-style log. Real gap; significant rewrite of §III |
| Date-stamped literature cutoff | ⊘ | §III currently says "1989-early 2026" but no specific cutoff date. Add explicit cutoff |
| Axis-assignment protocol | ⊘ | §III.C describes the philosophy but no per-method audit sheet. Could publish as supplementary material via the planning files |
| Per-paper extraction template | ⊘ | Would let readers replicate the synthesis |
| Whether paper is narrative / scoping / systematic | ⊘ | Reviewer 2's framing question; we should explicitly say "narrative survey with empirical add-ons" or "scoping review" |

### Reproducibility of empirical sections

| Ask | Status | Note |
|---|---|---|
| §VI tabular study under-specified | ⊕ | Tables IV-VI report mean ± std over 5 seeds, but no code/seed/hyperparam release. Need to either link a repo or substantially soften comparative claims |
| §VI baseline framing (VI/PI/MCTS as oracle vs peer) | ⊘ | Quick fix — relabel these in the text |
| §VI define abbreviations (CVPI, MPI) on first use | ⚙ | Quick prose fix |
| §VII repository audit needs version pins | ⊘ | Pin commit hashes / dates for each surveyed repo. Important since RLlib's `DQNConfig` exposes feature support that our named-algorithm matrix may understate |
| Distinguish named-algorithm support vs feature-equivalent support | ⊘ | Reviewer 2's specific concern about Table VII; would tighten claim |
| Statistical reporting: confidence intervals, effect sizes, rliable engagement | ⊘ | Should engage Agarwal et al. 2021 *Deep RL at the Edge of the Statistical Precipice* (rliable framework) in §V and §VI |
| Castro et al. 2020 *Revisiting Rainbow* engagement | ⊘ | Methodological precedent for axis-stratified evaluation; natural cite in §V |

### Taxonomic / framing concerns

| Ask | Status | Note |
|---|---|---|
| Distributional RL W4 placement (credit assignment vs uncertainty) | ⊕ | We chose this re-interpretation explicitly. Already flagged in `10-reviewer-audit.md` as a contested re-interpretation. Reviewer 2 wants us to "frame as interpretive lens rather than established classification" — this is a one-paragraph framing fix in §IV.D.A, not a structural change |
| W7 "Slow adaptation" too coarse (mixes distributed + recurrence + meta) | ⊘ | Reviewer 2's clearest taxonomic concern. Options: (a) split W7 into W7a "Scaling and distributed systems" and W7b "Slow adaptation and meta-learning"; (b) keep as single axis but explicitly relabel as composite ("composite systems-and-adaptation axis"); (c) defend the current bundling. Option (b) is cheapest, (a) is most defensible |
| Dueling DQN re-classification under §IV.H | ✓ | Already flagged as contested in `10-reviewer-audit.md`; defended via Rainbow ablation evidence |
| "Claims of firstness" need backing | ✓ | The prior-art sweep in `07-prior-art-sweep.md` is the documentation; need to cite it more prominently in §I |
| Engage Castro 2020 + Agarwal 2021 (eval methodology) | ⊘ | See above; significant addition |

### Limitations and ethics

| Ask | Status | Note |
|---|---|---|
| Dedicated limitations / ethics section | ⊘ | Benchmark monoculture, compute inequality, citation bias, stale tooling. §V.H and §VIII.C touch some of these but scattered. A consolidated subsection would address this |

**Reviewer 2 verdict:** of ~22 distinct asks, **three are quick fixes** (this turn), **three are already done** (or substantially handled), **two need bibliography work** (cross-cutting), and **fourteen are new methodological or content additions**. Reviewer 2's harsher tone reflects that most of their asks concern the scholarly apparatus rather than content depth.

---

## Cross-reviewer summary

| Category | Already done | Quick fix this turn | New session needed |
|---|---|---|---|
| Reviewer 1 content asks | 4 of 12 | 0 | 8 |
| Reviewer 2 structural asks | 3 of 22 | 3 | 14 (mostly small) |

**Both reviewers agree on the core contribution** — the eight-axis
problem-first framework. Neither asks the team to abandon it.

**Headline gap analysis:**

1. **Two content gaps** are individually large:
   - Q-learning for LLM/VLM alignment (Reviewer 1 #7) — natural new §IV.J or major §IV.G expansion
   - Predictable scaling laws (Reviewer 1 #6) — directly updates our §IV.G claims

2. **One scholarly apparatus gap** is non-negotiable for top-venue publication:
   - The bibliography. Already on our outstanding list. Needs pandoc-citeproc + `.bib` wiring before any submission, regardless of how much content we add.

3. **One methodological gap** would substantially strengthen the paper:
   - PRISMA-style review methodology in §III.A (search strings, screening flow, date cutoff, exclusion log). This is what makes a *narrative survey* defensible as a *systematic review*.

4. **One taxonomic gap** is cheap and worth doing:
   - Split or relabel W7 — Reviewer 2's cleanest taxonomic critique

5. **Two minor framing fixes** would inoculate against pushback:
   - One paragraph framing the distributional RL W4 placement as "an interpretive lens, defended below" rather than canonical (§IV.D.A)
   - One paragraph in §I pointing to the prior-art sweep when claiming "first multi-axis problem-first survey"

---

## Recommended next sessions

If the team has bandwidth for three more sessions before final
submission:

**Session 4 — quick wins + scholarly apparatus (highest priority).**
- ⚙ Fixed in this turn: PQN year (2024 not 2025), MDP notation θ removed, §VII.C count corrected
- ⊕ Stand up the References section: wire pandoc-citeproc, build `.bib` file, audit every in-text citation
- ⊘ Add PRISMA-style review protocol to §III.A
- ⊘ Add limitations/ethics subsection
- ⊘ Cite Agarwal et al. 2021 (rliable / Statistical Precipice) and Castro et al. 2020 (Revisiting Rainbow) in §V and §VI
- ⊘ One paragraph framing distributional RL W4 placement as interpretive (§IV.D.A); one paragraph pointing at prior-art sweep in §I

**Session 5 — large content gaps (most reviewer impact).**
- ⊘ §IV.J "Q-learning for Large Language Model and Vision-Language Model Alignment" — ShiQ, VLM Q-Learning, Q♯, KL-regularized Q-learning. Natural new ninth axis-section (the eight-axis frame already broken to nine by §IV.I theoretical advances)
- ⊘ §IV.G integration of Rybkin et al. predictable scaling laws — restructures the section's "single-learner Q is sample-bottlenecked" framing
- ⊘ §IV.E.B.5 addition: Q-learning with Adjoint Matching alongside FQL
- ⊘ §IV.A addition: Parameterized Q-Networks for hybrid action spaces

**Session 6 — taxonomic refinement and remaining methodological asks.**
- ⊘ Split or composite-label W7
- ⊘ §IV.D: crossing-quantiles problem + non-crossing variants
- ⊘ §VII: repository audit version pinning; named-vs-feature-support distinction
- ⊘ §VI: code/seed/hyperparameter release or claim softening; CVPI/MPI definitions
- ⊘ §VI baseline framing (planning baselines as oracles)

**Beyond three sessions:**
- Author block fill-in
- Two-column layout (deferred; needs IEEEtran or longtable workaround)
- Visual polish (TikZ genealogy, table overflow)

---

## Honest note

**Reviewer 1's report is significantly more positive than Reviewer 2's.** This may reflect different reviewer standards (Reviewer 1 reads as more enthusiastic, Reviewer 2 reads as more rigor-focused). For a top-tier venue, Reviewer 2's bar is the operative one — the bibliography gap and methodological transparency asks are the kind that block acceptance regardless of content quality. Closing Reviewer 1's content gaps without closing Reviewer 2's scholarly apparatus gaps would not be enough.

The path forward is therefore *both*: fix scholarly apparatus first
(Session 4), close the largest content gaps second (Session 5), then
refine (Session 6).

---

*Audit grounded in: two new reviewer reports (provided in chat),
draft files in `draft/`, prior audit `10-reviewer-audit.md`.*
