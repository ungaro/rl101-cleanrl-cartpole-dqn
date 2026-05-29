# Proposed New Outline — Problem-First Q-Learning Survey

This is the section-level skeleton of the rewritten paper. Each top-level
Related Works section is an *axis* — a weakness of vanilla Q-learning
that a family of methods addresses — rather than a *method type*.

The old taxonomy (Statistical / Q-Function Computation / Memory & Replay
/ Ensemble / Model-Based / Minimal) is preserved as a **secondary
indexing system** in an appendix and in inline tags on each method
("[Statistical, Replay]"), so we don't lose the bibliometric work
already done.

---

## Proposed structure

### I. Introduction
*Largely as-is.* Updates:
- The pitch shifts from "we built a unified taxonomy" to "we
  reorganize Q-learning's evolution around the *problems* its variants
  were designed to solve."
- Adds a one-paragraph framing claim: the field is not a sequence of
  parallel inventions; it is eight running attempts to fix eight
  weaknesses of vanilla Q-learning.
- Pre-announces the eight axes (table on first page).

### II. Background and Problem Setup
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

### III. Methodology
*Trimmed.* What stays:
- How we chose papers
- The contribution table (Table I)
- The repository analysis methodology

What is cut from the current Methodology:
- The "six categories" preamble (now an appendix indexer)

### IV. Related Works (eight axis-sections)
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

**IV.A. Overestimation Bias**
- Double Q-learning, Double DQN, Dueling DQN (partial), Rainbow, EBQL,
  Maximin Q (added), under-estimation as the opposite failure mode
- Trade-off axis: bias ↔ variance, with EBQL as the explicit
  ensemble-mediated point on this trade-off

**IV.B. Sample Inefficiency**
- DQN (replay buffer as baseline), Prioritized ER, MeDQN,
  DQfD, HER (added), Rainbow (re-appears via PER)
- Trade-off axis: stored memory cost ↔ sample efficiency gain

**IV.C. Brittle Exploration**
- Parameter Space Noise, NoisyNet, Bootstrapped DQN, UCB Q-Ensemble,
  CBDQ (belief-driven), RND / pseudo-counts (added briefly)
- Empirical evidence subsection leans hard on Montezuma's Revenge,
  Pitfall, Private Eye — the games that *expose* the axis. Atari
  dashes here are the argument

**IV.D. Reward Sparsity & Credit Assignment (distributional methods + n-step)**
- C51, QR-DQN, IQN, FQF, Multi-Step Q-Learning, λ-returns
- This is the natural home for distributional RL because the distribution
  *is* a richer credit-assignment signal, not just an uncertainty thing
- Trade-off axis: distribution flexibility ↔ compute cost (FQF is 20%
  slower than IQN)

**IV.E. Distribution Shift (offline RL)** *(NEW)*
- CQL, IQL, BCQ, BRAC, EDAC, AWAC — all added
- This is the section where modern Q-learning lives
- Empirical evidence shifts to D4RL benchmarks, not Atari
- This is also where "Atari is limited" is most directly addressed:
  the offline-RL literature uses different benchmarks because Atari
  cannot test distribution shift

**IV.F. Multi-Agent Coordination (value decomposition)** *(NEW)*
- VDN, QMIX, QPLEX, QTRAN — all added
- Even a 2-page treatment here would close reviewer point 5 substantially
- Empirical evidence: SMAC (StarCraft Multi-Agent Challenge)

**IV.G. Scaling and Slow Adaptation (distributed / meta)**
- Ape-X, R2D2, Agent57 (distributed scale — added)
- Meta-Q / MAML-Q (meta-RL — added)
- Deep Recurrent Q-Network (DRQN — kept here as the original
  partial-observability response)
- PQN (re-appears: it's a "remove the architecture overhead" answer
  to the scale axis)

**IV.H. Function-Approximation Instability (the deadly triad)**
- Nature DQN (target network), Polyak averaging, Dueling decomposition
  (as a stability response, not a value-estimation response — this is
  *the* re-interpretation we'd argue for), layer norm, PQN's normalization
  recipe, MeDQN's consolidation loss
- This is where the foundational *deep RL* recipe lives, separated from
  the more exotic enhancements

### V. Foundational Methods (kept as a single chronological section)
*Reframed, not cut.* The tabular foundations the current draft puts in
section F (Q-learning, SARSA, Multi-step Q, NFQ, PQN) become a single
**Foundations** section, separated from the axis sections because they
predate the weakness framing. This addresses the reviewer feedback
"too much derivation" by *concentrating* the derivations here instead
of spreading them across every section.

### VI. Empirical Evaluation (tabular)
*As-is structurally* (Tables IV/V/VI keep their place). Reframe the
discussion to tie each algorithm back to which axis-section it was
introduced under.

### VII. Empirical Evaluation (Atari from the literature)
*Tables II/III kept*, but the surrounding prose is rewritten:
- Drop the "Reporting gaps limit fair ranking" apologia
- Re-frame the tables as evidence stratified by axis (the
  Reaction-Time / Strategic Planning / Sparse / Dense categories
  already roughly map to axes — we make this explicit)
- Each axis-section in IV references the relevant column range of
  Tables II/III directly

### VIII. Repositories
*As-is in content*, but with light edits:
- Each repository's coverage is annotated *by axis* (axis-aware
  coverage matrix), not just by method
- Cross-references back to method sections: "Bootstrapped DQN
  (see IV.C) — absent from all six repositories"

### IX. Conclusion and Q-Learning Repository Spin-Off
*Promote the Conclusion's promise.* The new Q-learning–specific
repository becomes a named deliverable with a paragraph-long roadmap,
not a future-work bullet. If we stand the repo up before submission,
cite it.

### Appendices

**A. Legacy six-category indexing.** Table that maps every method in
the paper to its position in *both* taxonomies (axis + method-type).
Lets readers who came in expecting the old structure find their way.

**B. Notation and proofs.** Pull any heavy derivations (parameter-noise
likelihood-ratio expansion, distributional projections, convergence
analysis) into the appendix. This is the single biggest answer to
"too much derivation."

**C. Repository support matrix (Table VII expanded).**

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
