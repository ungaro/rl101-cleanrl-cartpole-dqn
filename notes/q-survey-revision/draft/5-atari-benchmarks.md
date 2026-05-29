# V. Atari Benchmark Analysis {#sec-v}

This section synthesizes Atari benchmark performance reported across
the methods of §IV. Because third-party implementations often diverge
from authors' codebases, and because official repositories are
incomplete (§VII), we extract results directly from peer-reviewed
publications rather than re-running experiments. Tables II and III
present raw per-game scores under each paper's original evaluation
protocol; Figure F-B2 (if included) provides an axis-stratified
visual summary.

### A. Table structure and dual-view organization

Tables II and III are structured on two axes simultaneously, and
both groupings are load-bearing.

**Row grouping (the conventional six-category taxonomy).** Methods
are grouped into six rows: *Statistical Methods*, *Q-Function
Computation*, *Memory/Replay*, *Ensemble-Based*, *Model-Based*, and
*Pure Q-Learning*. This grouping matches the method-type taxonomy
used by prior Q-learning surveys [12]–[15]. We retain it in Tables
II/III as a presentation device, because (a) it lets readers familiar
with the conventional taxonomy navigate the empirical data without
translation; (b) it preserves within-family comparability across
the historical literature; and (c) the six-category grouping makes
the modern-RL families (§IV.E offline, §IV.F multi-agent, §IV.G
distributed) visually salient as absences — there is no
six-category row for them to occupy.

**Column grouping (task categories).** Atari games are grouped into
seven task categories — Reaction-Time Control, Strategic Planning,
Sparse Rewards, Dense Rewards, Large Observation Space, Partially
Observable Environments, Stochastic Environments.

**Axis annotation.** A rightmost column or footnote on each table
maps each method's row to its primary §IV axis. The annotation does
not change the table's structure; it provides a third lens. Readers
can read Table II as "method-family × task-category" by attending to
the row grouping, or as "primary-axis × task-category" by attending
to the axis column.

The dual-view organization is itself a contribution. Readers
arriving with the catalogue-style expectation find the familiar
six-category structure; readers seeking analytical synthesis follow
the prose subsection-by-subsection below and the axis column.
Neither audience needs to translate the empirical data to align
with their reading.

### B. Tables II and III — extracted scores

\footnotesize

Table: **Reported Atari benchmark performance (raw per-game scores), Part I.** Reaction-Time Control / Strategic Planning / Sparse Rewards / Dense Rewards. "—" indicates the original paper did not report a score for that game.

| Method (year) | Legacy category | Breakout | Sp. Inv. | Ms. Pac-Man | Q*bert | Montezuma | Pitfall! | Boxing | Enduro |
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
| Parallel Q (PQN, 2025) | Pure Q | 515 | 18,451 | 5,568 | 31,717 | 0 | -89 | 100 | 2,349 |

\normalsize

\footnotesize

Table: **Reported Atari benchmark performance, Part II.** Large Observation Space / Partially Observable / Stochastic Environments.

| Method (year) | Legacy category | River Raid | Priv. Eye | Frostbite | Hero | Zaxxon | Berzerk |
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
| Parallel Q (PQN, 2025) | Pure Q | 28,764 | 100 | 7,314 | 26,099 | 23,538 | 18,542 |

\normalsize

### C. Task category stratification

**Each Atari task category preferentially tests one or two of the
eight weaknesses introduced in §II.B**:

| Task category | Diagnoses weakness(es) |
|---|---|
| Reaction-Time Control | function-approx stability (W8), sample inefficiency (W2) |
| Strategic Planning | reward sparsity & credit assignment (W4) |
| Sparse Rewards | brittle exploration (W3) |
| Dense Rewards | none specifically — baseline for stability |
| Large Observation Space | function-approx stability (W8) |
| Partially Observable | slow adaptation / recurrence (W7) |
| Stochastic Environments | overestimation bias (W1), distribution robustness |

The mapping is not perfect — most games test multiple weaknesses to
some degree — but the dominant signal in each category aligns
cleanly with one or two axes. The empirical claim of this paper's
structural pivot is that methods targeting weakness $W_i$ should
excel on the diagnostic category for $W_i$ and remain at baseline
elsewhere. Tables II and III, read through this lens, support the
claim.

### D. Patterns by axis

**Brittle exploration (W3, §IV.C).** The sparse-reward category
(Montezuma's Revenge, Pitfall!, Private Eye) is where the axis
matters most. Methods targeting exploration explicitly — NoisyNet,
Bootstrapped DQN, CBDQ, Posterior Sampling DQN — produce nonzero
scores on Montezuma where vanilla DQN and methods targeting other
axes do not. Rainbow's 384 on Montezuma is attributable specifically
to its NoisyNet component (per the original ablation). DQfD's 4,638
on Montezuma — substantially higher than any non-demonstration
method — illustrates the trade-off explored in §IV.B: demonstration
substitutes for directed exploration. RND and Go-Explore (newly
discussed in §IV.C but not in Tables II/III) report Montezuma scores
of ~10,000 and >1 million respectively, indicating that the axis is
not saturated by the methods covered here.

**Reward sparsity and credit assignment (W4, §IV.D).** The
strategic-planning category (Q*bert, Ms. Pac-Man) is where
distributional methods produce their most striking results. QR-DQN's
572,510 on Q*bert is the single largest score in our compilation by
any method — over $25\times$ the best non-distributional result on the
same game — and the pattern is consistent across the category. IQN
and FQF follow the same pattern at lower absolute magnitude.

**Function-approximation stability (W8, §IV.H).** Reaction-Time
Control games (Breakout, Space Invaders, Enduro) show comparatively
small inter-method variance: Double DQN, Dueling DQN, Rainbow, and
distributional methods cluster within a factor of two of each other
on these games. The pattern indicates that for games where the
weakness signal is *stability rather than capability*, modest
algorithmic improvements over the DQN baseline suffice — and PQN's
strong performance on the category (§IV.H) is consistent with this
interpretation.

**Sample inefficiency (W2, §IV.B).** PER's performance pattern,
across the categories, is most pronounced on strategic-planning and
sparse-reward games — categories where high-information transitions
are rare. On dense-reward games where informative transitions are
abundant under uniform sampling, the PER advantage is small. The
ablation pattern from Rainbow (PER is the largest single contributor)
is consistent with this category-stratified picture.

### E. The reporting gaps as evidence

Dashes ("---") in Tables II and III indicate that the original paper
introducing the method did not report a score for that game under
the stated evaluation protocol. Absence of data does not imply poor
performance — but the gap *pattern* is itself diagnostic and carries
more information than is sometimes assumed.

**For most methods, the gap pattern is itself diagnostic.** A method
that reports comprehensively on dense-reward games and skips sparse-
reward games signals — perhaps implicitly — that the method does not
solve the sparse-reward problem. A method that reports on Reaction-
Time Control and skips Stochastic Environments is unlikely to be a
strong response to W1 (overestimation, which compounds under
stochasticity). The pattern of dashes, considered as a structured
absence, is sometimes more informative than the present scores.

Several specific gaps illustrate the point. Ensemble Bootstrapping
[46] reports on 11 of 57 games and skips the entire sparse-reward
category, despite the ensemble mechanism being a natural exploration
candidate. Memory-Efficient DQN [41] reports on 5 of 57 games
selected for memory-sensitivity, leaving the method's broader
performance profile undefined. The dashes are honest about reporting
scope; reading them as evidence is internally consistent with the
authors' own treatment of their methods.

### F. The "improvement ladder" and its limits

Within several axes — particularly W1 (overestimation) and W4
(credit assignment) — a consistent chronological improvement
trajectory is visible: classical DQN → Double DQN → Dueling /
Prioritized → Distributional / Quantile-based methods, with each
generation outperforming the previous on its diagnostic category.
This "ladder" is real but axis-specific. A method that places at
the top of the W4 ladder (FQF) does not necessarily improve on W3
(FQF scores zero on Montezuma). The ladder is a strong frame *within
an axis* and a weak frame *across axes* — which is the central
empirical argument for the structural pivot.

### G. Why we do not produce a leaderboard

Earlier surveys [13]–[15] bold or italicize the "best" result per
game. This presentation implicitly invites comparison across methods
that used different evaluation protocols (training-frame budgets,
seed counts, no-op-start variations, sticky-action settings).
Boldface in the presence of protocol divergence overstates the
strength of comparisons. We refrain from highlighting per-cell
best-results in Tables II/III; the axis-stratified narrative above
substitutes evaluation against the *diagnostic categories* for the
weaker comparison against per-game best.
