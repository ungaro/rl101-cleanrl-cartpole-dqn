# S4. Full Repository Coverage Matrix {#supp-s4}

This is the full method-level coverage matrix that main §VII defers to supplementary material, spanning six widely-used deep RL repositories — Tianshou [@weng_2022_tianshou], XuanCe [@liu_2023_xuance], CleanRL [@huang_2022_cleanrl], DQN Zoo [@quan_2020_dqnzoo], Stable Baselines3 [@raffin_2021_sb3], and RLlib [@liang_2018_rllib]. It is a dated snapshot reflecting the state of each repository as of 2026-05-01, for the audited version ranges Tianshou 0.5.x, XuanCe 1.x, CleanRL master (commit-pinned during the audit), DQN Zoo (current head), Stable Baselines3 2.x, and RLlib 2.x. Support is marked by named-algorithm presence — a method counts as supported only if the repository implements it as a named algorithm with appropriate hyperparameter exposure — so the named-vs-feature-equivalent caveat from main §VII applies, and feature-equivalent availability of compound methods is consistently higher than the matrix indicates.

Table: **Repository support for deep Q-learning algorithms, grouped by method-type.** $\bullet$ = implemented; $\circ$ = not implemented.

| Method-type | Method (year) | Tianshou | XuanCe | CleanRL | SB3 | RLlib | DQN Zoo |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Statistical | Param Space Noise (2017) | $\circ$ | $\bullet$ | $\circ$ | $\circ$ | $\circ$ | $\circ$ |
| Statistical | C51 (2017) | $\bullet$ | $\bullet$ | $\bullet$ | $\circ$ | $\bullet$ | $\bullet$ |
| Statistical | NoisyNet (2018) | $\circ$ | $\circ$ | $\circ$ | $\circ$ | $\bullet$ | $\circ$ |
| Statistical | QR-DQN (2018) | $\bullet$ | $\bullet$ | $\circ$ | $\bullet$ | $\circ$ | $\bullet$ |
| Statistical | IQN (2018) | $\bullet$ | $\circ$ | $\circ$ | $\circ$ | $\circ$ | $\bullet$ |
| Statistical | FQF (2019) | $\bullet$ | $\circ$ | $\circ$ | $\circ$ | $\circ$ | $\circ$ |
| Q-Func. Comp. | Nature DQN (2015) | $\bullet$ | $\bullet$ | $\bullet$ | $\bullet$ | $\bullet$ | $\bullet$ |
| Q-Func. Comp. | DRQN (2015) | $\circ$ | $\circ$ | $\circ$ | $\circ$ | $\circ$ | $\circ$ |
| Q-Func. Comp. | Double DQN (2016) | $\bullet$ | $\bullet$ | $\circ$ | $\circ$ | $\bullet$ | $\bullet$ |
| Q-Func. Comp. | Dueling DQN (2016) | $\bullet$ | $\bullet$ | $\circ$ | $\bullet$ | $\bullet$ | $\circ$ |
| Q-Func. Comp. | Rainbow DQN (2018) | $\bullet$ | $\circ$ | $\circ$ | $\circ$ | $\bullet$ | $\bullet$ |
| Q-Func. Comp. | CBDQ (2025) | $\circ$ | $\circ$ | $\circ$ | $\circ$ | $\circ$ | $\circ$ |
| Memory/Replay | DQN (2013) | $\circ$ | $\circ$ | $\circ$ | $\circ$ | $\circ$ | $\circ$ |
| Memory/Replay | Prioritized ER (2016) | $\circ$ | $\bullet$ | $\circ$ | $\circ$ | $\bullet$ | $\bullet$ |
| Memory/Replay | DQfD (2018) | $\circ$ | $\circ$ | $\circ$ | $\circ$ | $\circ$ | $\circ$ |
| Memory/Replay | MeDQN (2023) | $\circ$ | $\circ$ | $\circ$ | $\circ$ | $\circ$ | $\circ$ |
| Ensemble | Bootstrapped DQN (2016) | $\circ$ | $\circ$ | $\circ$ | $\circ$ | $\circ$ | $\circ$ |
| Ensemble | UCB Q-Ensemble (2018) | $\circ$ | $\circ$ | $\circ$ | $\circ$ | $\circ$ | $\circ$ |
| Ensemble | Ensemble Bootstrapping (2021) | $\circ$ | $\circ$ | $\circ$ | $\circ$ | $\circ$ | $\circ$ |
| Model-Based | Posterior Sampling DQN (2023) | $\circ$ | $\circ$ | $\circ$ | $\circ$ | $\circ$ | $\circ$ |
| Pure Q | Parallel Q (PQN, 2024) | $\circ$ | $\circ$ | $\circ$ | $\circ$ | $\circ$ | $\circ$ |

The design trade-offs behind these coverage patterns are summarized below; this per-repository table is more detailed than the prose summary carried in main §VII. The taxonomic-coverage view here is complementary to the empirical-reproducibility audit of Hundal et al. [@hundal_2025_interchangeable], which compares a single algorithm across repositories under nominally equivalent configurations.

Table: **Comparison of popular Deep RL repositories — design pros and cons.**

| Repository | Pros | Cons |
|---|---|---|
| **Tianshou** | Dual API (high-level and low-level); emphasis on reproducibility with agent-level tests; native TensorBoard support | Logs/results not available for all algorithms; reproducibility guarantees not empirically verified across full benchmark suite |
| **XuanCe** | Modular YAML config files; W&B integration for hyperparameter tuning; high modularity for code reuse | Default hyperparameters often diverge from original papers; incomplete support across environments; steep learning curve |
| **CleanRL** | Single-file implementations; easy to audit and understand; lightweight and minimal dependencies | Limited support for large-scale experiments; less modular, harder to extend |
| **Stable Baselines3** | Clean API with sklearn-style interface; well-maintained and widely adopted; compatible with VecEnv, Gym, etc. | Focused more on policy-gradient methods; limited Q-learning variants beyond DQN |
| **RLlib** | Distributed training out-of-the-box; Ray Tune integration; production-grade scalability | High complexity and heavyweight; harder to debug or customize individual components |
| **DQN Zoo** | Faithful reimplementation of DQN variants; reproducibility aligned with DeepMind practices; highly organized training logs and configs | Focused exclusively on DQN family; less beginner-friendly |

These priorities are not in conflict but occupy different positions on a Pareto frontier: a repository optimizing for one property necessarily relaxes another. Selection of a repository is therefore itself an axis decision — Tianshou serves overestimation-method exploration well, RLlib serves at-scale deployment, and CleanRL serves educational demonstration — and no single repository currently serves the Q-learning research community comprehensively.
