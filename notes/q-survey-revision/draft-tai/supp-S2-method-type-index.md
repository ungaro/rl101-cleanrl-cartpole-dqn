# S2. Full Method-Type Index {#supp-s2}

This supplement is the full reference index promised by §III, §IV,
and §VIII. It supports readers who approach the material through the
conventional method-type taxonomy used in prior Q-learning surveys
[@urtans_2018_pygame; @jang_2019_qsurvey; @boppiniti_2021_evolution; @hafiz_2023_dqnsurvey].
The eight-axis problem-first organization of §IV is the analytical
spine of the paper; the six method-type categories are preserved as a
secondary indexing system. A reader interested in all distributional
methods navigates via the method-type view (the Statistical Methods
rows below); a reader interested in all methods addressing brittle
exploration navigates via §IV.C. The two views are complementary.

## S2.1. Full method-to-axis mapping

For each method covered in §IV, the table gives its **Method type**
(the conventional grouping: Statistical Methods, Q-Function
Computation, Memory/Replay, Ensemble-Based, Model-Based, Pure
Q-Learning), its **Primary axis** (the weakness $W_i$ from §II.B that
the method principally addresses, with its full-treatment section),
and **Secondary axes** (weaknesses it incidentally mitigates).

| Method (year) | Method type | Primary axis | Secondary axes |
|---|---|---|---|
| Parameter Space Noise (2017) | Statistical | §IV.C (W3) | — |
| NoisyNet (2018) | Statistical | §IV.C (W3) | — |
| C51 (2017) | Statistical | §IV.D (W4) | §IV.A |
| QR-DQN (2018) | Statistical | §IV.D (W4) | §IV.A |
| IQN (2018) | Statistical | §IV.D (W4) | §IV.A |
| FQF (2019) | Statistical | §IV.D (W4) | §IV.A |
| Double Q-Learning (2010) | Q-Function Comp. | §IV.A (W1) | §V foundations |
| Nature DQN (2015) | Q-Function Comp. | §IV.H (W8) | §V foundations |
| Deep Recurrent Q (DRQN, 2015) | Q-Function Comp. | §IV.G (W7) | — |
| Double DQN (2016) | Q-Function Comp. | §IV.A (W1) | — |
| Dueling DQN (2016) | Q-Function Comp. | §IV.H (W8) | §IV.A (incidental) |
| Rainbow (2018) | Q-Function Comp. | §IV.B (W2) | §IV.A, §IV.C, §IV.D, §IV.H |
| MCTS for FrozenLake (2024) | Q-Function Comp. | §V planning baseline | — |
| Cognitive Belief-Driven Q (2025) | Q-Function Comp. | §IV.C (W3) | §IV.A |
| DQN (2013) | Memory/Replay | §IV.B (W2) | §V foundations |
| Prioritized ER (PER, 2016) | Memory/Replay | §IV.B (W2) | — |
| DQfD (2018) | Memory/Replay | §IV.B (W2) | §IV.C (demos for exploration) |
| Memory-Efficient DQN (MeDQN, 2023) | Memory/Replay | §IV.B (W2) | §IV.H (consolidation) |
| Bootstrapped DQN (2016) | Ensemble-Based | §IV.C (W3) | §IV.A |
| Ensemble Bootstrapped Q (EBQL, 2021) | Ensemble-Based | §IV.A (W1) | §IV.C |
| UCB Q-Ensemble (2018) | Ensemble-Based | §IV.C (W3) | §IV.A |
| Value/Policy Iteration (1960s) | Model-Based | §V foundations | — |
| Bayesian Q-Learning (1998) | Model-Based | §V foundations | §IV.C (Thompson sampling) |
| Expected SARSA (2009) | Model-Based | §V foundations | — |
| Posterior Sampling DQN (2023) | Model-Based | §IV.C (W3) | §IV.A |
| Q-Learning (Watkins 1992) | Pure Q-Learning | §V foundations | — |
| SARSA (1994) | Pure Q-Learning | §V foundations | — |
| Multi-Step Q-Learning (1996) | Pure Q-Learning | §IV.D (W4) | §V foundations |
| Neural Fitted Q (NFQ, 2005) | Pure Q-Learning | §V foundations | §IV.B (batch updates) |
| Parallel Q Learning (PQN, 2024) | Pure Q-Learning | §IV.H (W8) | §IV.G (parallelism) |

## S2.2. Methods outside the six method-type categories

Eighteen methods covered in §IV have no row in the conventional
six-category taxonomy and do not appear in the main paper's row-spine
tables. They are treated in §IV under the new axes; the table below
maps them to the method-type taxonomy as they *would* have been
classified if the conventional categories were extended to admit them.

| Method (year) | Would-be method type | Primary axis |
|---|---|---|
| Maximin Q-Learning (2020) | Q-Function Comp. | §IV.A (W1) |
| REDQ (2021) | Ensemble-Based | §IV.A (W1) |
| HER (2017) | Memory/Replay | §IV.B (W2) |
| RND (2018) | Statistical (intrinsic motivation) | §IV.C (W3) |
| Go-Explore (2019/2021) | Memory/Replay (archive) | §IV.C (W3) |
| BCQ (2019) | New category — Offline RL | §IV.E (W5) |
| BRAC (2019) | New category — Offline RL | §IV.E (W5) |
| AWAC (2020) | New category — Offline RL | §IV.E (W5) |
| CQL (2020) | New category — Offline RL | §IV.E (W5) |
| IQL (2021) | New category — Offline RL | §IV.E (W5) |
| EDAC (2021) | New category — Offline RL + Ensemble | §IV.E (W5) |
| VDN (2018) | New category — Multi-Agent | §IV.F (W6) |
| QMIX (2018) | New category — Multi-Agent | §IV.F (W6) |
| QPLEX (2020) | New category — Multi-Agent | §IV.F (W6) |
| QTRAN (2019) | New category — Multi-Agent | §IV.F (W6) |
| Ape-X (2018) | New category — Distributed | §IV.G (W7) |
| R2D2 (2019) | New category — Distributed + Recurrent | §IV.G (W7) |
| Agent57 (2020) | New category — Distributed + Meta-policy | §IV.G (W7) |
| MAML-Q (2017/2019) | New category — Meta-RL | §IV.G (W7) |
| Munchausen DQN (2020) | Q-Function Comp. (would have been) | §IV.H (W8) |

The "New category" entries cannot be cleanly placed in the six
method-type categories: they are Q-learning families that emerged
after the taxonomy was conventionalized. Their absence is a central
argument in §I for the structural pivot — the six categories were
defined before these families existed, and accommodating them
requires adding categories that were never load-bearing. The
eight-axis structure has principled homes for all eighteen methods
without category-set expansion.

## S2.3. Reverse view — method type to §IV section coverage

The table inverts S2.1: for each method-type category, what fraction
of its constituent methods land in each axis-section of §IV.

| Method type | Primary axis distribution |
|---|---|
| **Statistical Methods** (6 methods) | 2 → §IV.C (Param Noise, NoisyNet); 4 → §IV.D (C51, QR-DQN, IQN, FQF) |
| **Q-Function Computation** (8 methods) | 2 → §IV.A (Double Q, Double DQN); 2 → §IV.H (Nature DQN, Dueling); 1 → §IV.B (Rainbow); 1 → §IV.C (CBDQ); 1 → §IV.G (DRQN); 1 → §V (MCTS) |
| **Memory/Replay** (4 methods) | 4 → §IV.B (DQN, PER, DQfD, MeDQN) |
| **Ensemble-Based** (3 methods) | 1 → §IV.A (EBQL); 2 → §IV.C (Bootstrapped DQN, UCB Q-Ensemble) |
| **Model-Based** (4 methods) | 1 → §IV.C (PSDQN); 3 → §V (VI/PI/MPI, Bayesian Q, Expected SARSA) |
| **Pure Q-Learning** (5 methods) | 4 → §V (Q-learning, SARSA, NFQ, multi-step); 1 → §IV.H (PQN) |

Three patterns emerge from the inverted view:

1. *Memory/Replay* is the most internally coherent method-type
   category: all four methods land in §IV.B (sample inefficiency).
   The method-type category and the axis category coincide.
2. *Ensemble-Based* fragments: ensembles serve overestimation control
   (§IV.A) and exploration (§IV.C) via the same architectural
   mechanism but different mechanisms-of-use. The axis view
   distinguishes them; the method-type view does not.
3. *Statistical Methods* fragments into noise-based exploration
   (§IV.C) and distributional credit-assignment (§IV.D) — the
   sharpest single demonstration of why the conventional method-type
   taxonomy conflates mechanism with effect.

These patterns are the data backing the problem-first organization,
verifiable method-by-method.
