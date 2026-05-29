# Appendix A — Legacy Indexer (Six Categories ↔ Eight Axes)

This appendix supports readers approaching the paper through the
conventional method-type taxonomy used in prior Q-learning surveys
[12]–[15] and in the original draft of this paper. The eight-axis
problem-first organization of §IV is the analytical spine of the
revised paper, but the six-category taxonomy is preserved as a
secondary indexing system — in Tables II/III row groupings, in
Appendix A's mapping below, and in inline tags within each §IV
subsection.

A reader interested in *all distributional methods* (a method-type
view) can navigate via the *Statistical Methods* row of Table II
and the corresponding rows of this appendix table. A reader
interested in *all methods addressing brittle exploration* (an
axis view) navigates via §IV.C. The two views are complementary;
neither is privileged for navigation.

---

## A.1. Full method-to-axis mapping

For each method covered in §IV, the table below shows:

- **Original legacy category** — the row grouping in the original
  draft's §III and Tables II/III. Six categories: *Statistical
  Methods*, *Q-Function Computation*, *Memory/Replay*,
  *Ensemble-Based*, *Model-Based*, *Pure Q-Learning (Minimal)*.
- **Primary axis** — the weakness $W_i$ from §II.B that the method's
  contribution principally addresses. The axis-section in §IV
  where the method receives full treatment.
- **Secondary axes** — additional weaknesses the method incidentally
  addresses or partially mitigates. Sections where the method is
  cross-referenced but not centrally treated.

| Method (year) | Legacy category | Primary axis | Secondary axes |
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
| Parallel Q Learning (PQN, 2025) | Pure Q-Learning | §IV.H (W8) | §IV.G (parallelism) |

## A.2. Methods added in the revised paper

The revised paper adds eighteen methods that have no row in the
original draft. These methods do not appear in Tables II/III
(which are preserved structurally) but receive treatment in §IV
under the new axes. For navigation, the table below maps them to
the legacy taxonomy as they *would* have been classified.

| Method (year) | Would-be legacy category | Primary axis |
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

The "New category" entries are the methods that *cannot* be cleanly
placed in the six legacy categories — Q-learning families that
emerged after the legacy taxonomy was conventionalized. Their
absence from the legacy structure is one of the central arguments
in §I for the structural pivot: the six categories were defined
when these families did not yet exist, and accommodating them in
the taxonomy requires adding categories that were never
load-bearing originally. The eight-axis structure has principled
homes for all eighteen methods without category-set expansion.

## A.3. Reverse view — legacy category to §IV section coverage

The table below inverts A.1: for each legacy category, what
fraction of its constituent methods land in each axis-section of
§IV.

| Legacy category | Primary axis distribution |
|---|---|
| **Statistical Methods** (6 methods) | 2 → §IV.C (Param Noise, NoisyNet); 4 → §IV.D (C51, QR-DQN, IQN, FQF) |
| **Q-Function Computation** (8 methods) | 2 → §IV.A (Double Q, Double DQN); 2 → §IV.H (Nature DQN, Dueling); 1 → §IV.B (Rainbow); 1 → §IV.C (CBDQ); 1 → §IV.G (DRQN); 1 → §V (MCTS) |
| **Memory/Replay** (4 methods) | 4 → §IV.B (DQN, PER, DQfD, MeDQN) |
| **Ensemble-Based** (3 methods) | 1 → §IV.A (EBQL); 2 → §IV.C (Bootstrapped DQN, UCB Q-Ensemble) |
| **Model-Based** (4 methods) | 1 → §IV.C (PSDQN); 3 → §V (VI/PI/MPI, Bayesian Q, Expected SARSA) |
| **Pure Q-Learning** (5 methods) | 4 → §V (Q-learning, SARSA, NFQ, multi-step); 1 → §IV.H (PQN) |

Three patterns are visible from the inverted view:

1. *Memory/Replay* is the most internally coherent legacy category:
   all four methods land in §IV.B (sample inefficiency). The legacy
   category and the axis category coincide here.
2. *Ensemble-Based* fragments: ensembles serve overestimation
   control (§IV.A) and exploration (§IV.C) via the same
   architectural mechanism but different mechanisms-of-use. The
   axis view distinguishes them; the legacy view does not.
3. *Statistical Methods* fragments into noise-based exploration
   (§IV.C) and distributional credit-assignment (§IV.D), the
   sharpest single demonstration of why the original taxonomy
   conflated mechanisms with effects.

These patterns are the data backing the structural pivot. The
appendix exists to let readers verify the claim themselves
method-by-method.

---

*Notes for integration:*
- This appendix is new content. It is the artifact that makes the
  structural pivot defensible — a reader who suspects the
  reorganization has cost something can check the indexer and find
  every method still present, just under a different organizing
  principle.
- Recommended placement: as Appendix A, before any other
  appendices. It is the most likely reference for readers
  cross-checking against the original taxonomy.
- The "Reverse view" subsection (A.3) is the analytical payload of
  the appendix. The forward-view (A.1) is reference; the
  reverse-view (A.3) is argument.
