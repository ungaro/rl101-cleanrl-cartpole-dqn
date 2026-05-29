# VII. Comparative Analysis of Deep Q-Learning Repositories {#sec-vii}

This section analyzes algorithmic coverage of Q-learning variants
across six widely-used open-source deep RL repositories — Tianshou
[6], XuanCe [7], CleanRL [8], DQN Zoo [9], Stable Baselines3 [10],
and RLlib [11] — together with their stated design priorities.
Table VII (Appendix A) reports a method-level coverage matrix;
Table VIII summarizes design trade-offs.

### A. Scope of this analysis

This paper's repository analysis is *taxonomic*: we report which
algorithms each repository implements, with brief commentary on the
design choices that shape coverage. We do *not* benchmark the
repositories' implementations against one another; that question is
addressed empirically by Hundal et al. [Hundal 2025], whose
controlled comparison of PPO across five repositories shows
substantial reproducibility variance even for a single algorithm
under nominally equivalent configurations. Hundal et al.'s
empirical-reproducibility audit and the taxonomic-coverage analysis
here are complementary: practitioners selecting a repository care
both *whether their algorithm is supported* (this paper's question)
and *whether the implementation reproduces published results*
(Hundal et al.'s question).

### B. Axis-aware coverage

Reading Table VII through the eight axis-sections of §IV reveals
patterns invisible in the method-type organization of the original
matrix:

**Well-supported axes.** §IV.A (overestimation, via Double DQN /
Dueling DQN) and §IV.D (distributional, via C51 / QR-DQN / IQN /
FQF) are the best-covered axes across repositories. Tianshou alone
implements all six method-level entries in these axes; XuanCe and
RLlib follow. The bias toward overestimation- and
distributional-method coverage reflects the empirical strength of
these families during the 2016–2019 deep-RL boom, when most of the
repositories were architected.

**Patchily-supported axes.** §IV.B (sample inefficiency) is covered
unevenly: PER is in three of six repositories, MeDQN in zero, DQfD
in zero, HER in three of six (but typically only as part of
goal-conditioned baselines, not as a Q-learning enhancement). §IV.C
(exploration) is similarly patchy: NoisyNet is in one of six
repositories (RLlib), Bootstrapped DQN in zero, UCB Q-Ensemble in
zero. The patchiness reflects the rapid evolution of these axes
and the difficulty of integrating per-method changes into
production-grade architectures.

**Entirely-absent axes.** §IV.F (multi-agent value decomposition)
and §IV.G (distributed-scale methods) are functionally absent from
the surveyed Q-learning repositories. RLlib supports multi-agent
RL via its actor framework but does not implement VDN, QMIX, QPLEX,
or QTRAN as native Q-learning methods. Ape-X, R2D2, and Agent57
have no implementations in any of the six repositories at the time
of writing. The absence is structural: these methods require
distributed infrastructure beyond the single-machine scope of
most repositories.

**§IV.E (offline RL)** is in transition. None of the six surveyed
repositories implement CQL, IQL, BCQ, or EDAC natively as of the
draft's evaluation cutoff. Practitioners use d3rlpy, Clean-Offline-RL,
or repository forks for these methods. The gap between
mainstream-DRL repository coverage and the offline-RL practical
landscape is one of the larger inconsistencies the axis-aware view
exposes.

### C. Methods absent from all six repositories

Eight methods covered in §IV are absent from *every* surveyed
repository:

| Method | Axis | §IV reference |
|---|---|---|
| Deep Recurrent Q (DRQN) | W7, scaling/adaptation | §IV.G.B.4 |
| Cognitive Belief-Driven Q (CBDQ) | W3, brittle exploration | §IV.C.B.3 |
| DQfD | W2, sample inefficiency | §IV.B.B.2 |
| Memory-Efficient DQN | W2, sample inefficiency | §IV.B.B.3 |
| Bootstrapped DQN | W3, brittle exploration | §IV.C.B.2 |
| UCB Q-Ensemble | W3, brittle exploration | §IV.C.B.2 |
| Ensemble Bootstrapping (EBQL) | W1, overestimation | §IV.A.B.2 |
| Posterior Sampling DQN | W3, brittle exploration | §IV.C.B.3 |
| Parallel Q-Learning (PQN) | W8, function-approx stability | §IV.H.B.3 |

The list overlaps substantially with the methods identified in §IV
as productive directions for future work, suggesting that
*implementation availability* is itself a bottleneck on practical
adoption of methodological advances. This observation motivates
the future-work proposal in §VIII for a community-maintained
Q-learning–specific repository.

### D. Tables VII and VIII — coverage and design trade-offs

\footnotesize

Table: **Repository support for deep Q-learning algorithms, grouped by category.** ● = implemented; ○ = not implemented.

| Category | Method (year) | Tianshou | XuanCe | CleanRL | SB3 | RLlib | DQN Zoo |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Statistical | Param Space Noise (2017) | ○ | ● | ○ | ○ | ○ | ○ |
| Statistical | C51 (2017) | ● | ● | ● | ○ | ● | ● |
| Statistical | NoisyNet (2018) | ○ | ○ | ○ | ○ | ● | ○ |
| Statistical | QR-DQN (2018) | ● | ● | ○ | ● | ○ | ● |
| Statistical | IQN (2018) | ● | ○ | ○ | ○ | ○ | ● |
| Statistical | FQF (2019) | ● | ○ | ○ | ○ | ○ | ○ |
| Q-Func. Comp. | Nature DQN (2015) | ● | ● | ● | ● | ● | ● |
| Q-Func. Comp. | DRQN (2015) | ○ | ○ | ○ | ○ | ○ | ○ |
| Q-Func. Comp. | Double DQN (2016) | ● | ● | ○ | ○ | ● | ● |
| Q-Func. Comp. | Dueling DQN (2016) | ● | ● | ○ | ● | ● | ○ |
| Q-Func. Comp. | Rainbow DQN (2018) | ● | ○ | ○ | ○ | ● | ● |
| Q-Func. Comp. | CBDQ (2025) | ○ | ○ | ○ | ○ | ○ | ○ |
| Memory/Replay | DQN (2013) | ○ | ○ | ○ | ○ | ○ | ○ |
| Memory/Replay | Prioritized ER (2016) | ○ | ● | ○ | ○ | ● | ● |
| Memory/Replay | DQfD (2018) | ○ | ○ | ○ | ○ | ○ | ○ |
| Memory/Replay | MeDQN (2023) | ○ | ○ | ○ | ○ | ○ | ○ |
| Ensemble | Bootstrapped DQN (2016) | ○ | ○ | ○ | ○ | ○ | ○ |
| Ensemble | UCB Q-Ensemble (2018) | ○ | ○ | ○ | ○ | ○ | ○ |
| Ensemble | Ensemble Bootstrapping (2021) | ○ | ○ | ○ | ○ | ○ | ○ |
| Model-Based | Posterior Sampling DQN (2023) | ○ | ○ | ○ | ○ | ○ | ○ |
| Pure Q | Parallel Q (PQN, 2025) | ○ | ○ | ○ | ○ | ○ | ○ |

\normalsize

Table: **Comparison of popular Deep RL repositories — design pros and cons.**

| Repository | Pros | Cons |
|---|---|---|
| **Tianshou** | Dual API (high-level and low-level); emphasis on reproducibility with agent-level tests; native TensorBoard support | Logs/results not available for all algorithms; reproducibility guarantees not empirically verified across full benchmark suite |
| **XuanCe** | Modular YAML config files; W&B integration for hyperparameter tuning; high modularity for code reuse | Default hyperparameters often diverge from original papers; incomplete support across environments; steep learning curve |
| **CleanRL** | Single-file implementations; easy to audit and understand; lightweight and minimal dependencies | Limited support for large-scale experiments; less modular, harder to extend |
| **Stable Baselines3** | Clean API with sklearn-style interface; well-maintained and widely adopted; compatible with VecEnv, Gym, etc. | Focused more on policy-gradient methods; limited Q-learning variants beyond DQN |
| **RLlib** | Distributed training out-of-the-box; Ray Tune integration; production-grade scalability | High complexity and heavyweight; harder to debug or customize individual components |
| **DQN Zoo** | Faithful reimplementation of DQN variants; reproducibility aligned with DeepMind practices; highly organized training logs and configs | Focused exclusively on DQN family; less beginner-friendly |

Reading Table VIII, the design priorities of each repository emerge:

- **Tianshou** and **XuanCe** prioritize coverage breadth across
  methods at the cost of single-method depth or reproducibility
  guarantees.
- **CleanRL** prioritizes single-file readability at the cost of
  modularity, which limits coverage of compositional methods
  (Rainbow, for instance, is harder to express in CleanRL's
  single-file architecture).
- **Stable Baselines3** prioritizes API stability and broad user
  adoption, with the cost of slower integration of new methods.
- **RLlib** prioritizes distributed scalability and production
  use, with the cost of complexity and difficulty in debugging
  individual algorithms.
- **DQN Zoo** prioritizes reproduction fidelity to DeepMind's
  published results, with the cost of restricted scope (DQN
  family only).

The design priorities are not in conflict but represent different
positions on a Pareto frontier: a repository optimizing for one
property necessarily relaxes another. The practitioner-side
implication is that selection of a repository is itself an axis
decision: a researcher exploring overestimation methods is well-
served by Tianshou; a practitioner deploying at scale is well-served
by RLlib; an educator demonstrating algorithms is well-served by
CleanRL. No single repository serves the field optimally — and the
axis-aware coverage gaps reported above suggest that no repository
currently serves the Q-learning research community comprehensively.

### E. Toward a Q-learning–specific repository

The axis-aware analysis above motivates a specific deliverable:
**a community-maintained repository focused exclusively on Q-learning
and its deep variants**, providing unified training scripts,
standardized benchmarks, modular algorithm implementations, and
shared logging tools. Such a repository would prioritize coverage of
the methods identified in §VII.C as absent from all existing
repositories, with empirical reproducibility audits in the style of
Hundal et al. as a release-quality gate.

Section VIII discusses this proposal as the paper's principal
forward-looking deliverable.
