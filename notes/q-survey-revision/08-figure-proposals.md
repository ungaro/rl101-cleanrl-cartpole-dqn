# Figure Proposals — Comparison Artifacts for the Revised Paper

Concrete proposals for visual comparisons in the revised paper.
Organized into:

- **Part A** — *per-section* comparison artifacts at the end of each
  Section IV subsection (the primary ask)
- **Part B** — paper-level structural figures supporting the
  problem-first thesis
- **Part C** — optional per-section evidence charts

The principle behind Part A: every axis-section ends with a single
visual that captures "what to take away from this section in 30
seconds." The format need not be an x/y chart — a comparison table,
capability matrix, or 2D positioning grid is often more informative.

## Status summary

| Artifact | Status | Where it lives |
|---|---|---|
| A.1 Comparison table per section | **drafted in all 8 axis-sections** | `draft/4a` through `draft/4h`, subsection F |
| A.2 2D positioning grid (mermaid quadrantChart) | **drafted for 5 sections** | §IV.A, §IV.C, §IV.D, §IV.E, §IV.H |
| A.3 Decision tree (mermaid flowchart) | **drafted for §IV.E** | `draft/4e-distribution-shift.md` |
| B.1 Axis × method-family matrix | **design drafted** (markdown table); TikZ rendering pending | `06-genealogy-figure.md` § axis matrix |
| B.2 Axis-stratified Atari grouped bars | ⊘ pending — data assembly needed | proposed only |
| B.3 TikZ genealogy | **design drafted** (ASCII + mermaid); TikZ rendering pending | `06-genealogy-figure.md` |
| B.4 Complexity vs. performance Pareto scatter | ⊘ pending — component-count list needed | proposed only |
| C.1 Compute log-log for §IV.G | **drafted as mermaid quadrantChart** | `draft/4g-scaling-adaptation.md` F |
| C.2 D4RL degradation curve for §IV.E | ⊘ pending — D4RL data extraction needed | proposed only |
| C.3 Per-game spread within axis | ⊘ pending — seed-level variance not always reported | proposed only |
| C.4 Distributional resolution vs. performance | **partially drafted as mermaid quadrantChart in §IV.D F** | `draft/4d-reward-sparsity.md` |

**Net:** the per-section comparison artifacts (Part A) are fully
in place; the paper-level figures (Part B) have design drafts but
need final rendering; the per-section evidence charts (Part C)
have two of four drafted as mermaid quadrantCharts.

Remaining design-heavy work: TikZ rendering of B.1 and B.3 for the
paper proper. Remaining data-heavy work: assembling component-count
lists for B.4 and D4RL benchmark extraction for C.2.

---

## Part A — Per-section comparison artifacts

### A.1. Uniform template: end-of-section comparison table

Every axis-section ends with a five-column comparison table summarizing
the methods discussed. Columns:

1. **Method (year)**
2. **Mechanism** — one-phrase description of *how* it addresses the
   weakness
3. **Primary cost** — what is traded for the gain (compute / memory /
   data / hyperparameter sensitivity / training instability)
4. **Best at** — the specific task type where the method excels
5. **Empirical anchor** — one canonical benchmark number from the
   paper

This format is uniform across sections, takes ~1/4 page in the IEEE
template, and gives the reader an at-a-glance comparison. Worked
example below for §IV.A (Overestimation Bias):

| Method (yr) | Mechanism | Primary cost | Best at | Anchor |
|---|---|---|---|---|
| Double Q-Learning (2010) | Two estimators, decoupled action selection / evaluation | None vs. tabular Q | Tabular discrete MDPs | GridWorld: higher reward/step |
| Double DQN (2016) | Online net selects, target net evaluates | Free (uses existing target net) | Most Atari games | Q*bert: 14,875 (vs 10,596 DQN) |
| Dueling DQN (2016)* | Separate V(s) and A(s,a) streams | Modest architectural | Many-action states | Dense-reward Atari |
| EBQL (2021) | K-ensemble, target = avg of $K-1$ others | $K\times$ compute, $K\times$ memory | Bias-variance tuning | 11 Atari games >Double DQN |
| REDQ (2021) | min of $M$ random ensemble members | $K\times$ compute, pushes under-est. | Continuous-control SAC | MuJoCo (off-Atari) |

*Dueling DQN's primary contribution is contested (§IV.A vs §IV.H).
Listed here as a secondary entry.

### A.2. Optional supplement: 2D trade-off positioning grid

For sections with a *clean two-dimensional trade-off*, supplement the
comparison table with a 2D positioning grid — each method placed as
a labeled marker on axes that capture the section's central tension.

**Sections where this is natural:**

| Section | X axis | Y axis |
|---|---|---|
| IV.A Overestimation | bias direction (under ← → over) | compute multiplier (1× → $K$×) |
| IV.B Sample efficiency | external data requirement (none → demos) | sample efficiency gain |
| IV.C Exploration | temporal consistency (per-step → per-episode) | effectiveness on Montezuma |
| IV.D Reward sparsity | distributional resolution (none → learned) | compute cost vs. one-step Q |
| IV.E Distribution shift | behavior-policy similarity (BC-like → unconstrained) | robustness to low-quality data |
| IV.F Multi-agent | representational capacity (additive → unconstrained) | trainability |
| IV.G Scaling | parallelism (sync → async distributed) | exploration heterogeneity |
| IV.H Stability | classical recipe (target net + replay) → modern recipe (norm-only) | sample efficiency at fixed compute |

**Format:** simple 2D grid, ~1/3 page, methods plotted as labeled
points. No quantitative axes — the goal is *qualitative positioning*
that captures the section's argument. Color/shape coding optional.

**Sections where it's not natural:** §IV.F (Multi-Agent) is partially
1D (expressiveness vs. trainability is the only meaningful axis, but
they're not independent — high expressiveness *causes* low
trainability in this literature, so the methods sit on a line, not a
plane). For that section, the comparison table alone suffices.

Worked sketch for §IV.A (Overestimation):

```
                 high compute
                      ▲
                      │
        EBQL(K=10)    │
            ●         │
                      │   REDQ
                      │    ●
   Dueling            │
      ◯               │
                      │
   Double DQN  ───────┼─────────  vanilla DQN
       ●              │              ●
                      │
                      │
                 low compute
                      │
       under-estimated ◀──────────▶ over-estimated
                      bias
```

Vanilla DQN sits upper-right of the bias-direction axis (heavy
over-estimation, low cost). Double DQN moves left (slight
under-estimation, no additional cost). EBQL at $K=10$ sits middle
(calibratable bias) at high compute. REDQ sits far left at high
compute (heavy under-estimation pressure). Dueling is a partial
contributor (open marker).

Worked sketch for §IV.C (Brittle Exploration):

```
  high Montezuma
        ▲
        │
        │              ● Go-Explore
        │
        │       ● RND
        │
        │
        │  ● Bootstrapped DQN
        │
        │       ◯ NoisyNet
        │   ● PSDQN
        │     ◯ UCB Q-Ens
        │  ● Param Space Noise
        │
   ε-greedy DQN ●        ● CBDQ
                                ▶
       per-step                per-episode
       adaptive                temporally consistent
```

NoisyNet (per-step adaptive) vs. Bootstrapped DQN (per-episode
consistent) trade-off is visible. RND and Go-Explore land high on the
Montezuma axis — the cluster the section actually wants to highlight.

### A.3. Decision tree variant (optional, for §IV.E only)

§IV.E (Distribution Shift / Offline RL) has a strong practitioner
audience and an unusually clean decision logic. A decision tree at
the end of the section serves practitioner readers well:

```
                  Have offline data?
                       │
              ┌────────┴────────┐
              No                Yes
              │                  │
            §IV.B           Data quality?
       (sample eff.)             │
                ┌────────┬──────┴───────┬──────────┐
              Expert   Medium       Medium-replay  Random
                │       │              │           │
                BC    AWAC/IQL        CQL        EDAC
              (good) (offline       (penalty   (ensemble
                      to online      handles    diversification)
                      bridge)        OOD)
```

This is the only section where a decision tree fits naturally; the
other axis-sections have more orthogonal method comparisons.

### A.4. Cost of Part A

- The end-of-section comparison table (A.1): ~1/4 page per section.
  Eight sections × 1/4 page = ~2 pages added.
- The 2D positioning grid (A.2): ~1/3 page per section where used
  (7 of 8). Adds ~2.5 pages.
- The decision tree (A.3): ~1/4 page, in §IV.E only.
- Total Part A page cost: ~5 pages added across all of §IV.

Trade-off: the paper grows by ~5 pages but every section becomes
visually navigable. For a survey paper this trade is favorable.

---

## Part B — Paper-level structural figures

These carry the paper's overall thesis. Either F-B1 or F-B2 should
appear; ideally both. F-B3 is the genealogy figure already drafted in
`06-genealogy-figure.md`.

### F-B1. Axis × method-family matrix

8 × 5 heatmap. Rows = the eight weakness axes from §II.B. Columns =
mechanism families (decoupling, ensembles, architectural, behavior
modulation, distributed). Cells filled (primary), open (secondary),
or blank (not addressed).

Goes in §IV introduction, immediately after recalling the
eight-weakness statement.

Sketch already in `06-genealogy-figure.md` near the end ("axis matrix
companion figure"). Effort: ~1 day for clean TikZ.

### F-B2. Axis-stratified Atari performance

Grouped bars or per-method radar chart. X = task category from
current Tables II/III (Reaction-Time, Strategic Planning, Sparse
Rewards, Dense Rewards, etc.). Y = mean human-normalized score.

Visualizes the structural claim that methods specialize on their
target axis. Goes in §V (Atari analysis), replacing the prose that
currently apologizes for dashes in Tables II/III.

Effort: ~2 days for data re-grouping + rendering.

### F-B3. Method genealogy with edge labels

ASCII sketch in `06-genealogy-figure.md`. Final version in TikZ,
edges color-coded by weakness addressed, [NEW] subtrees visually
distinguished.

Goes in §IV introduction or §I. Effort: ~3–5 days for a
publication-quality TikZ rendering.

### F-B4. Complexity vs. performance Pareto scatter

X = number of components over vanilla DQN. Y = median Atari HNS.
Points per method, colored by primary axis.

Shows Rainbow (upper-right) and PQN (upper-left) as opposite
extremes reaching similar performance. Goes in §I as a visual hook.

Effort: ~2 days for component-count assembly and rendering.

---

## Part C — Optional per-section evidence charts

If space permits beyond the per-section comparison artifacts (Part
A), these per-section evidence figures provide quantitative depth.

### F-C1. Compute-vs-performance log-log (for §IV.G)

X = log10(training frames). Y = median HNS. Points for DQN,
Rainbow, PQN at 200M; Ape-X, R2D2, Agent57 at higher scales.
Shows compute scaling and the PQN/Rainbow co-located point.

### F-C2. D4RL dataset-quality degradation (for §IV.E)

X = behavior quality (random → medium-replay → medium → medium-
expert → expert). Y = D4RL normalized score. Lines per method.
Shows offline-RL's central trade-off.

### F-C3. Per-game spread within axis (for §IV.C)

Horizontal violin plot of Montezuma scores per method.
Spread is itself diagnostic — high-mean methods with wide spread
are luckier than systematic.

### F-C4. Distributional resolution vs. performance (for §IV.D)

X = number of return-distribution parameters (51 atoms, 200
quantiles, sampled continuous, learned). Y = median HNS. Shows
the diminishing-returns curve along the distributional family.

---

## Recommended package

**Minimum viable visual upgrade** (high leverage, low cost):

1. Part A.1 (end-of-section comparison table) in **every section**
2. Part B.1 (axis × family matrix)
3. Part B.3 (TikZ genealogy from `06-genealogy-figure.md`)

This package gives every section a 30-second navigability artifact,
plus two paper-level figures carrying the structural thesis. Total
page cost: ~5 pages added.

**Recommended package** (covers the structural argument plus
practitioner needs):

1. Above three, plus
2. Part A.2 (2D positioning grid) for IV.A, IV.C, IV.D, IV.E, IV.H —
   sections where the trade-off is naturally 2D
3. Part A.3 (decision tree) for IV.E
4. Part B.2 (axis-stratified Atari grouped bars)

Total page cost: ~7 pages added.

**Maximum visual investment** (every section has both A.1 and A.2
plus paper-level figures and all Part C charts):

Total page cost: ~10–12 pages. Probably too much for a single
revision cycle; reasonable target for a longer-form journal version.

---

## Data we already have vs. need to assemble

| Artifact | Data source | Status |
|---|---|---|
| A.1 comparison tables | Section prose itself | Mechanical; per-section author can compile in ~30 min each |
| A.2 2D positioning grids | Section trade-off discussions | Qualitative; ~1 hr design + render each |
| A.3 decision tree (IV.E only) | §IV.E content | Ready to render |
| B.1 axis × family matrix | `04-method-remap.md` | Ready |
| B.2 axis-stratified Atari | Tables II/III | Re-grouping needed |
| B.3 TikZ genealogy | `06-genealogy-figure.md` | Ready to render |
| B.4 complexity-perf scatter | Tables II/III + lit | Component-count list needed |
| C.1–C.4 evidence charts | Original papers | Data extraction needed per chart |

---

*Notes for integration:*

- The end-of-section comparison table (A.1) is the cheapest and
  most-impactful visual addition. Even with no other figure changes,
  every section ending with a uniform comparison table gives the
  paper a substantial navigability improvement.
- Color schemes should be colorblind-safe (Wong palette or
  equivalent) and reduce gracefully to black-and-white printing.
  IEEE TAI accepts color but many readers print BW.
- All 2D positioning grids (A.2) are *qualitative* — they capture
  the structure of the trade-off, not precise quantitative values.
  Resist the urge to add quantitative axes; the value is in the
  *positioning*, not the *measurements*.
