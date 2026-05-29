# Q-Learning Survey — Pitch Analysis & Direction Suggestions

Working notes for shaping this week's discussion. Based on the
`RL_Project_Pitch_Q_Survey.pdf` deck (24 slides, May 30 2026) and the
five-point IEEE TAI reviewer summary it surfaces.

This is a **direction-setting note**, not polished output. Goal: 1–2
concrete moves we could rally the team around, plus the reasoning so
the team can argue with it.

---

## My read of the reviewer feedback

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

This frames the suggestions below.

---

## Suggestion 1 (high leverage): pivot the organizing axis from
## *method type* to *problem solved*

The current taxonomy has six method-type buckets (Statistical, Q-Function
Computation, Memory/Replay, Ensemble, Model-Based, Minimal). This is a
*botanist's* taxonomy — it tells you what *kind* of thing each method
is. The reviewers want a *physician's* taxonomy — what *symptom* does
each method treat.

Proposed alternative organizing axis: **the foundational weaknesses of
vanilla Q-learning that modern methods address**. Sketch:

| Weakness of vanilla Q | Methods that target it |
|---|---|
| **Overestimation bias** (max operator + noisy estimates) | Double Q, Double DQN, Maximin Q, Weighted Double Q, REDQ (under-estimation), …|
| **Sample inefficiency** (replay uniformity, single trajectory) | PER, Rainbow, Ape-X, R2D2, hindsight (HER), demonstration learning (DQfD), … |
| **Brittle exploration** (ε-greedy is poor at deep exploration) | NoisyNet, Bootstrapped DQN, IQN, UCB-ensembles, BBQ, RND, … |
| **Reward sparsity / credit assignment** | Distributional (C51, QR-DQN, IQN, FQF), n-step, λ-returns, MuZero-style, … |
| **Distribution shift** (online → offline) | CQL, IQL, BCQ, BRAC, EDAC, … (Q-learning's offline-RL chapter) |
| **Coordination** (single-agent → multi-agent) | VDN, QMIX, QPLEX, Q-DPP, … |
| **Slow adaptation** (per-task training) | Meta-Q (MAML on Q), Reptile-Q, distillation-based transfer, … |
| **Function-approximation instability** (the deadly triad) | Target networks, Polyak averaging, dueling decomposition, distributional, … |

Each section becomes:

1. **The weakness** — formal statement, why it's hard, what it costs.
2. **Solution families** — grouped by mechanism, compared head-to-head.
3. **Trade-offs introduced** — every fix has a cost; name it.
4. **Empirical evidence** — what ablation/benchmark data we have.
5. **Open question** — what's still unsolved in this dimension.

**Why this is the right pivot:**

- It directly addresses reviewer points 1 and 2 (synthesis +
  conceptual insight). The structure forces cross-method comparison —
  Double Q vs. Maximin vs. REDQ all share a section.
- It pulls modern RL into the existing structure rather than tacking
  it on. *Distribution shift* is the natural home for offline RL.
  *Coordination* is the home for multi-agent value decomposition. No
  "Modern RL" appendix needed; modern Q-methods slot in where they
  belong.
- It addresses the "Atari is limited" complaint indirectly: a
  *problem-first* paper doesn't lean on Atari as the spine. Atari
  becomes one evidence stream for *exploration* and *sample
  efficiency*; classic control becomes evidence for *stability*;
  offline benchmarks (D4RL, RL Unplugged) become evidence for
  *distribution shift*. The benchmark landscape becomes coherent.
- It justifies cutting derivations: a problem-first section is about
  the *mechanism* of the fix, not the formal derivation. Derivations
  move to an appendix or a single foundational section.

**What this looks like as a deliverable for this week:**
A 2–3 page outline that re-maps the current six categories to the
problem-first structure. Each existing method (DQN, PER, Rainbow,
C51, …) gets a new home, and a few methods land in *multiple* homes
(Rainbow especially — that's the *point*, and surfacing the
cross-cutting nature is the synthesis).

**Risk to flag:** this is a *structural* change. If the draft is
already in late revision, the cost may be higher than the team can
absorb in one revision cycle. Mitigation: the genealogy figure in
Suggestion 2 captures most of the synthesis benefit without rewriting
the prose.

---

## Suggestion 2 (lower-cost): add a centerpiece *design-space* figure

If we cannot afford a full structural pivot, the next-highest-leverage
move is a **single well-designed figure that does the synthesis work**.
The reviewers complain about isolated methods because they have no
visual anchor for how methods relate. One good figure makes the
relationships visible at a glance.

Two candidates:

### 2a. Q-learning *genealogy tree*

A directed graph of methods, with each edge annotated by the weakness
the child method addresses in the parent:

```
                          tabular Q-learning (1989)
                                    │
                                    │ (function approximation)
                                    ▼
                               DQN (2013)
                  ┌─────────────────┼─────────────────┐
                  │ overestimation  │ uniform replay  │ value/advantage
                  ▼                 ▼                 ▼
            Double DQN (2015)   PER (2015)      Dueling DQN (2015)
                  │                 │                 │
                  └──────────┬──────┴──────────┬──────┘
                             │   integrated    │
                             ▼                 ▼
                       Rainbow (2017)    Distributional
                                              │
                                              ▼
                                         C51 → QR-DQN → IQN → FQF
                 ┌───────────────────────────────┐
                 │ distribution shift            │ coordination
                 ▼                               ▼
            CQL, IQL, BCQ (offline)         QMIX, VDN (multi-agent)
```

Annotate every edge. The annotations *are* the synthesis the
reviewers want.

### 2b. Q-learning *design-space scatter plot*

A 2D figure with axes like:
- x: sample efficiency (e.g. median DQN-normalized score at 10M
  frames)
- y: implementation complexity (or: number of components on top of
  vanilla DQN)
- color: family (statistical / replay / ensemble / model-based / …)
- size: typical compute budget required

Each method is a point. The clusters make the relationships visible:
"Rainbow is high-complexity, high-efficiency; minimal Q-learning is
low-complexity, low-efficiency; ensembles are high-complexity,
medium-efficiency."

Reviewers asking "how do these methods relate" get a literal *map*.

**What this looks like as a deliverable for this week:** the genealogy
figure is the cheaper of the two — it can be drafted in TikZ or even
ASCII first, then handed off for proper rendering. A first-pass
genealogy by end of week is realistic.

**Why this is a meaningful move even if we don't do Suggestion 1:**
the figure can be inserted into the existing structure with minimal
prose disruption, and it gives reviewers a concrete artifact that
demonstrates "yes, we synthesized." It's a high-visibility,
low-prose-cost win.

---

## Strategic recommendation

**Run both, sequenced.** Suggestion 2 (the figure) is the cheap, fast,
visible win — it can land this week and gives the team something to
rally around. Suggestion 1 (the structural pivot) is the bigger
intellectual move and the one that actually transforms the paper from
"collection" to "analytical paper." Suggestion 2 *informs* Suggestion 1
— the act of drawing the genealogy is itself a synthesis exercise that
will surface which problem-first sections make sense.

If the team has bandwidth for only one move this week, do the
genealogy figure first. Use the discussion of which edges to label as
the seed for the structural pivot in a later iteration.

---

## Open questions for the group

1. **How far along is the current draft?** If it's near-final, the
   structural pivot may be too disruptive and the figure-only path is
   what we should commit to. If it's mid-revision, we have room for
   the full pivot.
2. **What's the timeline back to the editor?** The structural pivot
   takes ~2–4 weeks of distributed work; the figure takes 1 week.
3. **Who owns the modern-RL expansion?** Offline RL (CQL/IQL line),
   distributed RL (Ape-X/R2D2/Agent57), and multi-agent (QMIX/VDN)
   each need a domain owner. The PhD researchers on the team are the
   natural fits; the engineers can do the empirical comparison work.
4. **Should the repository comparison stay as a standalone section?**
   It currently reads like a sidebar. Alternative: fold each
   repository's choices into the method sections they implement —
   "Rainbow as implemented in Tianshou vs. CleanRL" reads as
   reproducibility analysis, which the reviewers explicitly want.

---

## What's *not* in this note

- I have not read the actual paper draft, only the pitch deck. The
  structural pivot suggestion is informed but not specific to what's
  written. If the team has the draft accessible, the next pass of this
  note should be a section-by-section re-mapping rather than abstract
  guidance.
- I have not surveyed what other 2024–2026 Q-learning surveys look
  like. A quick search before finalizing direction would be cheap and
  useful — e.g. is anyone else doing the problem-first organization?
  If yes, our angle changes; if no, we have a clean differentiation
  claim.

---

*Drafted on branch `alp/q-survey-revision-prep`. Not merged to main —
this is exploratory thinking, not course content.*
