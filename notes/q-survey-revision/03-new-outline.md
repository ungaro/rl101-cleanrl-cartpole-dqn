# New Outline — Problem-First Q-Learning Survey

The section-level skeleton of the revised paper. Each top-level
Related Works section is an *axis* — a weakness of vanilla Q-learning
that a family of methods addresses — rather than a *method type*.

The old taxonomy (Statistical / Q-Function Computation / Memory & Replay
/ Ensemble / Model-Based / Minimal) is preserved as the **row spine of
Tables II/III** (see `draft/5-atari-benchmarks.md` §V.A) and as
Appendix A (`draft/A-legacy-indexer.md`).

**Status:** structure below is fully drafted. Each section heading
includes a link to the corresponding markdown draft file. Status
markers:

- ✓ drafted
- (modern) needs domain-owner review before integration
- ⊘ pending

---

## Structure with status

### I. Introduction ✓ — [`draft/1-introduction.md`](draft/1-introduction.md)
*Largely as-is.* Updates:
- The pitch shifts from "we built a unified taxonomy" to "we
  reorganize Q-learning's evolution around the *problems* its variants
  were designed to solve."
- Adds a one-paragraph framing claim: the field is not a sequence of
  parallel inventions; it is eight running attempts to fix eight
  weaknesses of vanilla Q-learning.
- Pre-announces the eight axes (table on first page).

### II. Background and Problem Setup ✓ — [`draft/2-background.md`](draft/2-background.md)
*Largely as-is.* The MDP setup stays. Add ~1 page: **"Eight Weaknesses
of Vanilla Q-Learning."** This is the conceptual spine — each weakness
is named, formally stated in 2–4 lines of math, and given an empirical
illustration (one paragraph). The eight Related Works sections that
follow are responses to these eight named weaknesses.

The eight weaknesses (this is the spine of the paper):

1. **Overestimation bias** — `max` over noisy estimates is biased upward
2. **Sample inefficiency** — uniform replay wastes high-information transitions
3. **Brittle exploration** — ε-greedy is dithered, not directed
4. **Reward sparsity & credit assignment** — bootstrapping a delayed
   signal through a deep network is unstable
5. **Distribution shift** — Q-learning is off-policy *in theory* but
   fragile when the behavior policy is fixed (offline RL)
6. **Multi-agent coordination** — single-agent Q does not factor across
   cooperative agents
7. **Slow adaptation** — Q-learning trains per-task; transfer is hard
8. **Function-approximation instability** — the "deadly triad" of
   off-policy + bootstrapping + function approximation

### III. Methodology ✓ — [`draft/3-methodology.md`](draft/3-methodology.md)
*Trimmed.* What stays:
- How we chose papers
- The contribution table (Table I)
- The repository analysis methodology

What is cut from the current Methodology:
- The "six categories" preamble (now an appendix indexer)

### IV. Related Works (eight axis-sections) — overview ✓ — [`draft/4-overview.md`](draft/4-overview.md)

The §IV overview holds the paper's two structural figures (master
genealogy + modern-RL subgraph, both mermaid) plus the axis ×
mechanism-family matrix. It frames the eight axis-sections that
follow.

Each section follows the same five-part template:

> **A. The Weakness**
> Formal statement (math). Why it matters. The simplest case where it
> bites.
>
> **B. Solution Families**
> Methods grouped by *mechanism*, not chronology. For each family:
> one-paragraph sketch of how it addresses the weakness, the key
> equation, citation.
>
> **C. Trade-offs**
> Every fix has a cost. What does each family give up — compute,
> wall-clock, sample efficiency on benign tasks, exploration, etc.
>
> **D. Empirical Evidence**
> A table or paragraph drawing from the Atari results (Table II/III in
> the current draft) showing where each family pulls ahead and where
> it fails. **The dashes in the current tables become *evidence* here**
> — methods that don't report on Montezuma cannot be claimed to solve
> exploration.
>
> **E. Open Questions**
> What's unsolved on this axis. Replaces the current "Critical
> Reflections" but is sharper and references work the methods could
> not have tried.

The eight sections:

**IV.A. Overestimation Bias** ✓ — [`draft/4a-overestimation-bias.md`](draft/4a-overestimation-bias.md)
- Double Q-learning, Double DQN, Dueling DQN (partial), Rainbow, EBQL,
  Maximin Q (added), under-estimation as the opposite failure mode
- Trade-off axis: bias ↔ variance, with EBQL as the explicit
  ensemble-mediated point on this trade-off

**IV.B. Sample Inefficiency** ✓ — [`draft/4b-sample-inefficiency.md`](draft/4b-sample-inefficiency.md)
- DQN (replay buffer as baseline), Prioritized ER, MeDQN,
  DQfD, HER (added), Rainbow (re-appears via PER)
- Trade-off axis: stored memory cost ↔ sample efficiency gain

**IV.C. Brittle Exploration** ✓ — [`draft/4c-brittle-exploration.md`](draft/4c-brittle-exploration.md)
- Parameter Space Noise, NoisyNet, Bootstrapped DQN, UCB Q-Ensemble,
  CBDQ (belief-driven), RND / pseudo-counts (added briefly)
- Empirical evidence subsection leans hard on Montezuma's Revenge,
  Pitfall, Private Eye — the games that *expose* the axis. Atari
  dashes here are the argument

**IV.D. Reward Sparsity & Credit Assignment (distributional methods + n-step)** ✓ — [`draft/4d-reward-sparsity.md`](draft/4d-reward-sparsity.md)
- C51, QR-DQN, IQN, FQF, Multi-Step Q-Learning, λ-returns
- This is the natural home for distributional RL because the distribution
  *is* a richer credit-assignment signal, not just an uncertainty thing
- Trade-off axis: distribution flexibility ↔ compute cost (FQF is 20%
  slower than IQN)

**IV.E. Distribution Shift (offline RL)** ✓ (modern) — [`draft/4e-distribution-shift.md`](draft/4e-distribution-shift.md)
- CQL, IQL, BCQ, BRAC, EDAC, AWAC — all added
- This is the section where modern Q-learning lives
- Empirical evidence shifts to D4RL benchmarks, not Atari
- This is also where "Atari is limited" is most directly addressed:
  the offline-RL literature uses different benchmarks because Atari
  cannot test distribution shift

**IV.F. Multi-Agent Coordination (value decomposition)** ✓ (modern) — [`draft/4f-multi-agent.md`](draft/4f-multi-agent.md)
- VDN, QMIX, QPLEX, QTRAN — all added
- Even a 2-page treatment here would close reviewer point 5 substantially
- Empirical evidence: SMAC (StarCraft Multi-Agent Challenge)

**IV.G. Scaling and Slow Adaptation (distributed / meta)** ✓ (modern) — [`draft/4g-scaling-adaptation.md`](draft/4g-scaling-adaptation.md)
- Ape-X, R2D2, Agent57 (distributed scale — added)
- Meta-Q / MAML-Q (meta-RL — added)
- Deep Recurrent Q-Network (DRQN — relocated from §IV.B for
  contextual-adaptation framing)
- PQN (re-appears: it's a "remove the architecture overhead" answer
  to the scale axis)

**IV.H. Function-Approximation Instability (the deadly triad)** ✓ — [`draft/4h-stability.md`](draft/4h-stability.md)
- Nature DQN (target network), Polyak averaging, Dueling decomposition
  (as a stability response, not a value-estimation response — this is
  *the* re-interpretation argued in subsection B.2), layer norm, PQN's
  normalization recipe, MeDQN's consolidation loss, Munchausen DQN
- This is where the foundational *deep RL* recipe lives, separated from
  the more exotic enhancements

**IV.I. Theoretical Foundations and Recent Advances** ✓ — [`draft/4i-theoretical-advances.md`](draft/4i-theoretical-advances.md)
- Watkins & Dayan 1992 tabular convergence; Tsitsiklis & Van Roy
  1997 deadly-triad counterexamples; Baird 1995; GTD/TDC line
- Distributional Bellman contraction (Bellemare 2017, Rowland 2018);
  finite-time bounds (Yang 2019, Fan 2020); pessimism in offline RL
  (Jin et al. 2021); stability theory (Lyle 2023, Nikishin 2022);
  QPLEX IGM completeness theorem (Wang 2020)
- Meta-section over the eight axes — addresses the "recent
  theoretical advances" reviewer ask directly

### V. Atari Benchmark Analysis ✓ — [`draft/5-atari-benchmarks.md`](draft/5-atari-benchmarks.md)
*Tables II/III kept structurally*, with the legacy six-category row
grouping preserved (§V.A — dual-view organization). Surrounding
prose rewritten:
- Drop the "Reporting gaps limit fair ranking" apologia
- Re-frame the tables as evidence stratified by axis (the
  Reaction-Time / Strategic Planning / Sparse / Dense column
  categories map to axes — this is now made explicit)
- Each axis-section in §IV references the relevant column range
- The dashes are reframed as evidence, not apology

*Note on the foundational methods:* the original outline proposed a
separate "Foundations" section consolidating Q-Learning, SARSA,
Multi-Step Q, NFQ, PQN, etc. In the actual draft, these are kept
within their respective axis-sections (Multi-Step Q in §IV.D, PQN
in §IV.H, and the tabular baselines as cross-references from §VI).
A separate foundations section would have duplicated material; the
consolidation is achieved through Appendix A's legacy indexer
instead.

### VI. Empirical Evaluation (Tabular) ✓ — [`draft/6-tabular-empirical.md`](draft/6-tabular-empirical.md)
*As-is structurally* (Tables IV/V/VI preserved). Discussion reframed
to tie each algorithm to its §IV axis-section and to surface the
methodological purpose of tabular evaluation: isolating algorithmic
design from architectural confound.

### VII. Repositories ✓ — [`draft/7-repositories.md`](draft/7-repositories.md)
*Tables VII/VIII preserved.* Prose rewritten with:
- Each repository's coverage annotated *by axis* (§VII.B)
- One paragraph distinguishing this paper's taxonomic analysis
  from Hundal et al. 2025's empirical PPO audit
- §VII.C identifies nine methods absent from *all six* surveyed
  repositories — the roadmap for §VIII's repository spin-off

### VIII. Conclusion and Q-Learning Repository Spin-Off ✓ — [`draft/8-conclusion.md`](draft/8-conclusion.md)
*Repository proposal promoted from future-work bullet to a named
deliverable with four design priorities and a prioritized roadmap.*
Summary subsection (§VIII.A) organized by the eight weakness axes.
Three cross-axis open directions surfaced — only visible from this
paper's vantage.

### Appendices

**A. Legacy Indexer.** ✓ — [`draft/A-legacy-indexer.md`](draft/A-legacy-indexer.md)
Full bidirectional map between the six legacy categories and the
eight axes. Forward view (every method → primary + secondary axes),
reverse view (each legacy category → distribution across axes), and
treatment of the 18 new methods that have no row in the original
taxonomy. The reverse view is the data backing the structural pivot.

**B. Notation and Selected Derivations.** ✓ — [`draft/B-notation-and-proofs.md`](draft/B-notation-and-proofs.md)
Notation reference table consolidated from §II.C, plus selected
derivations the §IV body references rather than works through:
maximum-of-noisy-estimators bound (Smith & Winkler), categorical
distributional projection, Wasserstein contraction proof sketch,
pessimism LCB argument for offline RL, QPLEX IGM completeness, and
the Watkins & Dayan 1992 tabular convergence proof. Reading guide
in §I points readers here.

**C. Repository support matrix.** ⊘ pending. Table VII expanded with
axis annotations, complementing §VII.

---

## What this buys us, mapped to reviewer feedback

| Reviewer complaint | How the new structure addresses it |
|---|---|
| Algorithms feel isolated | Every method lives in a *family* compared head-to-head inside its axis-section |
| Need conceptual insight | The axis-sections *are* the conceptual layer; each section starts with the weakness, not the method |
| Atari is limited | Offline RL (IV.E) and multi-agent (IV.F) introduce D4RL and SMAC; Atari results are stratified by axis and the dashes become evidence |
| Too much derivation | Derivations move to Appendix B; in-section text is mechanism + trade-off |
| Need more modern RL | IV.E, IV.F, and IV.G are *new sections*, not appendices — they expose the modern-RL areas the current draft misses |

---

## What this *costs*

| Cost item | Estimate |
|---|---|
| Restructure existing 30+ method writeups into new sections | 2 weeks distributed work |
| Write three new sections (Offline / Multi-Agent / Distributed) | 1 week per section per owner — needs domain owners |
| Update Tables II/III re-framing prose | 2 days |
| Update Table VII to axis-aware version | 1 day |
| Write new section II expansion (eight weaknesses) | 2 days |
| Set up legacy-indexer appendix | 1 day |
| Genealogy figure (Suggestion B) | 2–3 days |
| Q-learning repo spin-off (Suggestion C) | 1 day for stub |

Rough total: **3–4 weeks of distributed work** with 3+ contributors,
assuming domain owners are lined up for offline / multi-agent /
distributed.

---

## What we do *not* need to do

- Rewrite the abstract / impact statement materially (the pitch shifts
  but stays at the same scope)
- Rewrite the existing per-method paragraphs (they get *re-homed*, not
  re-written, with cosmetic edits)
- Re-run any of the existing tabular benchmarks
- Re-extract any Atari numbers

The leverage of this restructure is exactly that *most of the existing
prose survives*. We're rearranging deck chairs *deliberately and
explicitly* — and the reviewers asked for the rearrangement.

---

## Risks

- **Section IV.E and IV.F need domain owners.** If no co-author has
  offline-RL or multi-agent background, the new sections will be
  thin and reviewers will notice.
- **The eight-axis framing is a claim.** A reviewer could argue we
  collapsed two axes that should be separate (e.g. exploration vs.
  reward sparsity) or split one that should be unified. Defensible —
  the framing has citations and forced cross-comparisons — but it is
  an editorial commitment we'd own.
- **Re-homing some methods is contested.** Distributional RL especially:
  the current draft puts it under "Statistical" (uncertainty); we'd
  move it under "Reward Sparsity & Credit Assignment" (richer signal).
  We should note both readings in the section and pick the load-bearing
  one for the structure.

---

*Companion files:*
- `04-method-remap.md` — every method, current section vs. new axis (the spine)
- `draft/4a-overestimation-bias.md` — IV.A written end-to-end as proof of concept
- `draft/1-introduction.md`, `draft/2-background.md` — revised §I and §II
- `06-genealogy-figure.md` — ASCII genealogy visual
