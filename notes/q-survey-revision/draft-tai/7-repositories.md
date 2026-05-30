# VII. Comparative Analysis of Deep Q-Learning Repositories {#sec-vii}

To assess how the methodological landscape of §IV is reflected in
practice, we audited six widely-used open-source deep-RL repositories —
Tianshou [@weng_2022_tianshou], XuanCe [@liu_2023_xuance], CleanRL
[@huang_2022_cleanrl], DQN Zoo [@quan_2020_dqnzoo], Stable Baselines3
[@raffin_2021_sb3], and RLlib [@liang_2018_rllib] — for Q-learning
coverage, organized by the eight axes of §IV. The full coverage matrix
(supplementary material) records each method against each repository;
here we report the structure it reveals.

**Coverage findings.** Coverage is sharply skewed toward the early-era
families. Overestimation methods ([§IV.A](#sec-iv-a), via Double and
Dueling DQN) and distributional methods ([§IV.D](#sec-iv-d), via C51,
QR-DQN, IQN, FQF) are the best-supported axes, an artifact of the
2016–2019 deep-RL boom during which most of these repositories were
architected. Sample-efficiency ([§IV.B](#sec-iv-b)) and exploration
([§IV.C](#sec-iv-c)) are patchy — prioritized replay and HER appear in
roughly half, NoisyNet in only one. The newer axes are largely absent:
offline RL ([§IV.E](#sec-iv-e)) is implemented natively by none of the
six (practitioners defer to d3rlpy and similar), and multi-agent value
decomposition ([§IV.F](#sec-iv-f)) and distributed-scale methods
([§IV.G](#sec-iv-g)) have no native VDN/QMIX/QPLEX or Ape-X/R2D2/Agent57
implementations — a structural gap, since these demand infrastructure
beyond single-machine scope. Most pointed: nine methods surveyed in §IV
— DRQN, CBDQ, DQfD, MeDQN, Bootstrapped DQN, UCB Q-Ensemble, EBQL,
Posterior Sampling DQN, and PQN — are absent from *all six*
repositories. This set overlaps heavily with the directions §IV flags as
most promising, indicating that implementation availability is itself a
bottleneck on adoption — the central motivation for the
community-repository proposal in [§VIII](#sec-viii).

**Non-interchangeability.** Coverage is necessary but not sufficient.
Hundal et al. [@hundal_2025_interchangeable] show that different
implementations of the *same* algorithm diverge materially in measured
performance, because silent code-level choices — gradient clipping,
observation and reward normalization, weight initialization — vary
across repositories without appearing in the algorithm's name. A "Double
DQN" cell in one repository is therefore not directly comparable to the
same cell in another. Cross-repository comparison consequently requires
controlling for these implementation details rather than trusting
nominal algorithm labels.

**Named-algorithm vs. feature-equivalent support.** Our audit marks a
method supported only when a repository ships it as a *named* algorithm
with exposed hyperparameters — the strictest reading. Feature-equivalent
availability is higher: RLlib's `DQNConfig`, for instance, exposes noisy,
distributional, and prioritized-replay flags that compose into a
Rainbow-like agent the matrix does not credit as a named-Rainbow
implementation, an effect also present to a lesser degree in Tianshou
and XuanCe.

**Design trade-offs.** The repositories occupy distinct points on a
Pareto frontier rather than dominating one another. CleanRL
[@huang_2022_cleanrl] optimizes single-file readability, which limits
modularity and makes compositional agents such as Rainbow awkward to
express; DQN Zoo [@quan_2020_dqnzoo] optimizes reproduction fidelity at
the cost of scope. At the other end, RLlib [@liang_2018_rllib] optimizes
distributed scale at the cost of debuggability and per-algorithm
transparency, while Stable Baselines3 [@raffin_2021_sb3] optimizes API
stability and broad adoption at the cost of Q-learning-variant breadth.
Tianshou [@weng_2022_tianshou] and XuanCe [@liu_2023_xuance] trade
single-method depth for coverage breadth. Repository selection is thus
itself an axis decision, and no single repository serves the Q-learning
community comprehensively.

| Repository | Strength | Trade-off |
|---|---|---|
| Tianshou | Broad coverage; best distributional/overestimation support | Reproducibility unverified across full suite |
| XuanCe | Modular YAML config; high code reuse | Defaults diverge from papers; steep curve |
| CleanRL | Single-file readability; easy to audit | Low modularity; compositional agents awkward |
| Stable Baselines3 | Stable API; widely adopted | Few Q-learning variants beyond DQN |
| RLlib | Distributed scale out-of-the-box | Heavyweight; hard to debug per-algorithm |
| DQN Zoo | High reproduction fidelity to DeepMind | Scope limited to the DQN family |

: Per-repository strengths and the trade-offs they accept.

This audit is a dated snapshot, reflecting repository state as of
2026-05-01 (Tianshou 0.5.x, XuanCe 1.x, CleanRL master, DQN Zoo head,
Stable Baselines3 2.x, RLlib 2.x); coverage evolves quickly and should
be re-checked against current releases.
