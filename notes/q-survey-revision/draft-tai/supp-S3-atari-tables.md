# S3. Full Atari Per-Game Benchmark Tables {#supp-s3}

The two tables below give the full per-game Atari scores that main §V
defers to this supplement. All values are scores exactly as reported in
the original papers introducing each method; we have not re-run any
experiment or rescaled any number. Evaluation protocols differ across
publications — no-op starts versus sticky actions, frameskip settings,
training-frame budgets, and seed counts all vary — so the cells are not
mutually calibrated. These tables must therefore be read with the
rliable-style caveats stated in main §V [@agarwal_2021_rliable]: as a
record of what each paper reported under its own protocol, not as a
leaderboard, and never as a per-cell "best method" ranking.

Table: Reported Atari benchmark performance (raw per-game scores), Part I — Reaction-Time Control / Strategic Planning / Sparse Rewards / Dense Rewards. "—" indicates the original paper did not report a score for that game.

| Method (year) | Method type | Breakout | Sp. Inv. | Ms. Pac-Man | Q*bert | Montezuma | Pitfall! | Boxing | Enduro |
|---|---|---|---|---|---|---|---|---|---|
| Param Space Noise (2017) | Statistical | 390 | 1,205 | — | 7,525 | 0 | -100 | — | 1,672 |
| C51 (2017) | Statistical | 748 | 5,747 | 3,415 | 23,784 | 0 | 0 | 98 | 3,454 |
| NoisyNet (2018) | Statistical | 516 | 2,186 | 2,722 | 15,545 | 3 | 0 | 89 | 1,240 |
| QR-DQN (2018) | Statistical | 742 | 20,972 | 5,821 | 572,510 | 0 | 0 | 100 | 2,355 |
| IQN (2018) | Statistical | 734 | 28,888 | 6,349 | 25,750 | 0 | 0 | 100 | 2,359 |
| FQF (2019) | Statistical | 854 | 140 | 7,632 | 27,524 | 0 | 0 | 98 | 2,371 |
| Nature DQN (2015) | Q-Func. Comp. | 401 | 1,976 | 2,311 | 10,596 | 0 | — | 72 | 302 |
| Deep Recurrent Q (2015) | Q-Func. Comp. | — | — | 2,048 | — | — | — | — | — |
| Double DQN (2016) | Q-Func. Comp. | 375 | 3,155 | 3,210 | 14,875 | 0 | — | 82 | 320 |
| Dueling DQN (2016) | Q-Func. Comp. | 345 | 6,427 | 6,284 | 19,220 | 0 | 0 | 99 | 2,258 |
| Rainbow DQN (2018) | Q-Func. Comp. | 418 | 18,789 | 5,380 | 33,818 | 384 | 0 | 100 | 2,126 |
| CBDQ (2025) | Q-Func. Comp. | — | — | — | — | — | — | — | — |
| DQN (2013) | Memory/Replay | 168 | 581 | — | 1,952 | — | — | — | 470 |
| Prioritized ER (2016) | Memory/Replay | 481 | 1,697 | 965 | 12,741 | 44 | -194 | 70 | 1,266 |
| DQfD (2018) | Memory/Replay | 308 | — | 4,696 | 21,793 | 4,638 | 57 | 99 | 2,200 |
| MeDQN (2023) | Memory/Replay | — | — | — | — | — | — | — | — |
| Bootstrapped DQN (2016) | Ensemble | 855 | 2,893 | 2,983 | 15,093 | 100 | — | 93 | 1,591 |
| UCB Q-Ensemble (2018) | Ensemble | 411 | 2,627 | 3,425 | 14,198 | 4 | -1 | 98 | 2,753 |
| Ensemble Bootstrapping (2021) | Ensemble | 406 | — | — | 14,384 | — | — | — | — |
| Posterior Sampling DQN (2023) | Model-Based | 46 | 511 | 1,824 | 4,245 | 0 | -44 | 79 | 363 |
| Parallel Q (PQN, 2024) | Pure Q | 515 | 18,451 | 5,568 | 31,717 | 0 | -89 | 100 | 2,349 |

Per-game scores for the Reaction-Time Control, Strategic Planning, Sparse Rewards, and Dense Rewards task categories, grouped by method type.

Table: Reported Atari benchmark performance, Part II — Large Observation Space / Partially Observable / Stochastic Environments.

| Method (year) | Method type | River Raid | Priv. Eye | Frostbite | Hero | Zaxxon | Berzerk |
|---|---|---|---|---|---|---|---|
| Param Space Noise (2017) | Statistical | — | 100 | 1,310 | — | 8,050 | — |
| C51 (2017) | Statistical | 17,322 | 15,095 | 3,965 | 38,874 | 10,513 | 1,645 |
| NoisyNet (2018) | Statistical | 9,425 | 3,712 | 753 | 6,246 | 6,920 | 905 |
| QR-DQN (2018) | Statistical | 17,571 | 350 | 4,384 | 21,395 | 13,112 | 3,117 |
| IQN (2018) | Statistical | 17,765 | 200 | 4,324 | 28,386 | 21,772 | 1,053 |
| FQF (2019) | Statistical | 23,561 | 140 | 16,473 | 30,926 | 15,180 | 12,422 |
| Nature DQN (2015) | Q-Func. Comp. | 8,316 | 1,788 | 328 | 19,950 | 4,977 | — |
| Double DQN (2016) | Q-Func. Comp. | 12,015 | 670 | 242 | 20,357 | 10,182 | — |
| Dueling DQN (2016) | Q-Func. Comp. | 21,163 | 103 | 4,673 | 20,818 | 13,886 | 3,409 |
| Rainbow DQN (2018) | Q-Func. Comp. | — | 4,234 | 9,591 | 55,887 | 22,210 | 2,546 |
| Prioritized ER (2016) | Memory/Replay | 10,206 | 2,202 | 289 | 15,151 | 9,501 | 644 |
| DQfD (2018) | Memory/Replay | 18,735 | 42,457 | — | 105,929 | — | — |
| Bootstrapped DQN (2016) | Ensemble | 12,845 | 1,813 | 2,181 | 21,021 | 11,492 | — |
| UCB Q-Ensemble (2018) | Ensemble | 15,622 | 1,252 | 1,903 | — | 3,695 | — |
| Ensemble Bootstrapping (2021) | Ensemble | — | 100 | — | — | — | — |
| Posterior Sampling DQN (2023) | Model-Based | 3,858 | 68 | 929 | 7,965 | 4,413 | 386 |
| Parallel Q (PQN, 2024) | Pure Q | 28,764 | 100 | 7,314 | 26,099 | 23,538 | 18,542 |

Per-game scores for the Large Observation Space, Partially Observable, and Stochastic Environments task categories, grouped by method type.
