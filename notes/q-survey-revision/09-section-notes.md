# Section Author Notes

Per-section material notes for LaTeX integration. Bibliography
additions, citation placement, scope decisions, and cross-references
that are useful to keep track of but do not belong in the paper body.

Organized by section.

---

## §I Introduction

- **Citation placement.** Ghasemi et al. 2024/2025 (arXiv:2411.18892)
  cited as the closest contemporary broad-RL survey. Place alongside
  the existing comparator surveys [12]–[15].
- **§VII positioning.** A single sentence in §VII distinguishes this
  paper's taxonomic repository analysis from Hundal et al. (2025)'s
  empirical PPO reproducibility audit. The Hundal citation enters in
  §I and is paid off in §VII; see `07-prior-art-sweep.md`.

## §II Background

- **Bibliography to verify.** Thrun & Schwartz 1993 ("Issues in
  using function approximation for reinforcement learning") and
  Sutton & Barto 2018 (2nd ed., for the deadly-triad framing) are
  cited in §II.B; confirm both are in the bibliography.
- The notation conventions subsection (II.C) is shared by every
  subsequent section. Subsequent sections do not redefine
  $\theta, \theta^-, \alpha, \gamma, \pi, \mathcal{D}$.

## §III Methodology

- **Table I update.** Add Ghasemi et al. 2024/2025 (arXiv:2411.18892)
  as a fifth comparator column. Check-mark pattern remains favorable;
  pattern follows `07-prior-art-sweep.md`.

## §IV Overview

- **Figure rendering.** The two mermaid figures (master genealogy
  and modern-RL branches) are design renderings; the final paper
  figures should be TikZ/PGF. The master genealogy in particular
  should sit as a single landscape half-page figure rather than two
  vertical ones.
- **Priority.** The axis × mechanism-family matrix is the most
  central visual to the paper's thesis; if only one paper-level
  figure is rendered, prioritize this matrix.

## §IV.B Sample Inefficiency

- **Bibliography.** HER: Andrychowicz et al. 2017, "Hindsight
  Experience Replay," NeurIPS, arXiv:1707.01495.
- **Cross-references.** §IV.A (Rainbow), §IV.C (Private Eye
  exploration), §IV.E (offline RL sample-efficiency transfer),
  §IV.G (distributed training), §IV.H (MeDQN consolidation loss).

## §IV.C Brittle Exploration

- **Bibliography.**
  - RND: Burda et al. 2018, arXiv:1810.12894.
  - Go-Explore: Ecoffet et al. 2019/2021, *Nature* 2021.
- **Cross-references.** §IV.A (EBQL ensemble architecture shared),
  §IV.B (DQfD demonstration trade-off), §IV.D (distributional methods
  on the exploration diagonal), §IV.G (Agent57's bandit-controlled
  exploration meta-policy).

## §IV.D Reward Sparsity and Credit Assignment

- **Cross-references.** §IV.A (Rainbow's ablation evidence), §IV.B
  (PER as the other major Rainbow contributor), §IV.C (the
  exploration-versus-credit-assignment empirical separation), §IV.E
  (distributional methods in offline RL — EDAC, IQL).

## §IV.E Distribution Shift (Offline RL)

- **Bibliography (large set).**
  - BCQ: arXiv:1812.02900
  - BRAC: arXiv:1911.11361
  - AWAC: arXiv:2006.09359
  - CQL: arXiv:2006.04779
  - IQL: arXiv:2110.06169
  - EDAC: arXiv:2110.01548
  - D4RL: arXiv:2004.07219
  - Cal-QL: arXiv:2303.05479
  - Decision Transformer: arXiv:2106.01345 *(open-questions only)*
  - Gato: arXiv:2205.06175 *(open-questions only)*
- **Single-axis precedent.** Springer NCAA 2026 distribution-shift
  survey (10.1007/s00521-026-11966-8) cited at the head of this
  section as the existing single-axis problem-first precedent. See
  `07-prior-art-sweep.md`.
- **Scope decision.** Sequence-modeling approaches (Decision
  Transformer, Trajectory Transformer, Gato) are deliberately
  scoped out as they are not Q-learning. They appear in the
  open-questions discussion because they dominate the data-scaling
  regime, but the section's scope is the Q-learning family.
- **Cross-references.** §IV.A (ensembles for bias control, EDAC),
  §IV.B (HER trajectory relabeling as a related sample-efficiency
  mechanism), §IV.H (stability mechanism interaction).

## §IV.F Multi-Agent Coordination

- **Bibliography.**
  - VDN: arXiv:1706.05296
  - QMIX: arXiv:1803.11485
  - QPLEX: arXiv:2008.01062
  - QTRAN: arXiv:1905.05408
  - SMAC: arXiv:1902.04043
- **Empirical numbers.** SMAC results table drawn from QPLEX's
  reported numbers under the standard SMAC evaluation protocol.
  Independent replication results — Hu et al. 2021, "Rethinking the
  Implementation Tricks and Monotonicity Constraint in Cooperative
  MARL" — report different absolute numbers but consistent relative
  rankings.
- **Scope decision.** Competitive multi-agent RL (Nash Q-learning,
  opponent modeling) and mixed cooperative-competitive settings are
  scoped out. The paper's scope is cooperative value decomposition.
- **Cross-references.** §IV.E (offline + multi-agent as a cross-axis
  open problem), §IV.H (multi-agent function-approximation
  instability inheritance).

## §IV.G Scaling and Slow Adaptation

- **Bibliography.**
  - Ape-X: arXiv:1803.00933
  - R2D2: ICLR 2019 (+ DeepMind blog post)
  - Agent57: arXiv:2003.13350
  - NGU (Never Give Up): arXiv:2002.06038
  - MAML: arXiv:1703.03400
  - DRQN: arXiv:1507.06527 *(may already be cited)*
- **Scope decision.** IMPALA, SEED RL, Sample Factory, Podracer are
  mentioned only at cross-reference; full coverage of the
  distributed-RL architecture literature would substantially expand
  the section.
- **Cross-axis observation.** Agent57's composition of mechanisms
  from §IV.B (distributed replay), §IV.C (exploration via
  bandit-controlled policy portfolio + NGU intrinsic motivation),
  and this section (slow adaptation via implicit task-distribution
  training) is the strongest empirical evidence for the problem-first
  organization. Frontier agents are increasingly cross-axis.
- **Cross-references.** §IV.B (PER at distributed scale), §IV.C
  (exploration heterogeneity), §IV.H (PQN's synchronous parallelism).

## §IV.H Function-Approximation Instability

- **Bibliography.** Munchausen DQN: Vieillard et al. 2020, NeurIPS,
  arXiv:2007.14430.
- **Cross-references.** §IV.A (Dueling's overestimation aspect),
  §IV.B (MeDQN as consolidation; PQN's replay-replacement argument),
  §IV.D ($n$-step interaction with target staleness), §IV.E
  (offline-RL stability mechanisms), §IV.G (PQN's parallelism for
  scaling).

## §V Atari Benchmark Analysis

- **Optional figure.** F-B2 (axis-stratified Atari grouped bars,
  per `08-figure-proposals.md`) sits naturally between §V.A and
  §V.B as a quantitative visualization of the category-stratification
  claim.

## §VI Tabular Empirical Evaluation

- **Possible expansion.** A simple offline-RL baseline (FQE — Fitted
  Q Evaluation) on a fixed FrozenLake dataset would provide tabular
  evidence for the §IV.E distribution-shift axis. Optional addition
  if bandwidth allows; strengthens axis-attribution.
- The methodological motivation ("isolating algorithmic design from
  architectural confound") should also be surfaced in the abstract
  or §I.

## §VII Repository Comparison

- **Roadmap for §VIII.** The "absent from all six repositories" list
  in §VII.C serves dual purposes: as evidence for the implementation
  gap in §VII, and as the first-tranche roadmap for the
  community-maintained Q-learning repository proposed in §VIII.B.

## §VIII Conclusion

- **Repository spin-off URL.** If a Q-learning–specific repository is
  stood up before submission (cf. `01-pitch-analysis.md`
  Suggestion C), the URL can be cited inline in §VIII.B. If not, the
  description in §VIII.B suffices as a future-work commitment.

## Appendix A — Legacy Indexer

- **Placement.** Recommended as Appendix A, before any other
  appendices. The most likely reference for readers cross-checking
  against the conventional method-type taxonomy.
- **Analytical payload.** The reverse view (A.3 — each legacy
  category's distribution across the eight axes) is the analytical
  contribution. The forward view (A.1) is reference.
