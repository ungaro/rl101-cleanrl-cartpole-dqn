# VI. Empirical Evaluation of Tabular Q-Learning Variants {#sec-vi}

To isolate algorithmic mechanism from the confounds of neural function
approximation, we evaluate four model-free value-based learners in the
tabular setting, where the Q-function is an exact table and any
performance difference is attributable to the update rule alone. The
study is deliberately small and mechanism-focused; it is not a
large-scale benchmark.

**Protocol.** We use three classic discrete environments —
FrozenLake-v1 (4×4, slippery: stochastic transitions), Taxi-v3
(structured, near-deterministic), and CliffWalking (per-step cost with
a cliff penalty) — and four learners: Q-learning, SARSA, Expected
SARSA, and $n$-step Q-learning ($n=3$). Each learner is trained over
**100 random seeds** with an $\varepsilon$-greedy schedule decayed from
1.0 to 0.05; we then evaluate the greedy policy over 300 episodes and
report the mean episodic return with a **95\% bootstrap confidence
interval** (10,000 resamples), following the interval-estimate practice
advocated by [@agarwal_2021_rliable]. The full configuration and a
seeded, single-file reproduction script are provided as supplementary
material.

**Planning oracles, not competitors.** We also report the optimal
policy obtained by dynamic programming (Value and Policy Iteration).
These methods require full access to the transition model $P$ and reward
$R$ and so are *not* model-free competitors; they are reported only as a
*planning upper bound* and are separated from the learners in the table
accordingly.

| Algorithm | FrozenLake-v1$^\dagger$ | Taxi-v3 | CliffWalking |
|---|---|---|---|
| Q-learning | 0.722 [0.701, 0.737] | 7.93 [7.90, 7.96] | $-13.0$ [$-13, -13$] |
| SARSA | 0.692 [0.669, 0.712] | 7.90 [7.87, 7.93] | $-79.9$ [$-114, -51$] |
| Expected SARSA | 0.720 [0.705, 0.732] | 7.93 [7.90, 7.96] | $-17.0$ [$-17, -17$] |
| $n$-step Q ($n{=}3$) | 0.497 [0.451, 0.541] | 7.93 [7.89, 7.96] | $-75.2$ [$-104, -46$] |
| \textit{Planning oracle (VI/PI)} | \textit{0.739} | \textit{7.94} | \textit{$-13.0$} |

: Tabular results: mean episodic return over 100 seeds with 95\%
bootstrap CIs. The planning oracle (full model access) is an upper
bound, not a model-free competitor. $^\dagger$FrozenLake reports success
rate (mean reward, max 1.0).

**What the results show.** Three patterns, each illustrating an axis
from §IV. (i) On *stochastic* FrozenLake, Q-learning and Expected SARSA
(0.722, 0.720) nearly reach the 0.739 oracle, while $n$-step Q collapses
to 0.497 with a wide CI: bootstrapping a multi-step return through
stochastic transitions injects exactly the variance that the
credit-assignment axis ([§IV.D](#sec-iv-d)) trades against bias and the
stability axis ([§IV.H](#sec-iv-h)) warns of. (ii) On
*near-deterministic* Taxi, all four learners converge to within 0.5\% of
the oracle with tight, overlapping CIs — when transitions are
predictable the choice of update rule barely matters, a useful negative
result. (iii) CliffWalking reproduces the canonical on-/off-policy
contrast: *off-policy* Q-learning recovers the optimal cliff-edge path
($-13.0$, matching the oracle exactly, zero variance), whereas the
expectation method prefers the *safe* route one row from the cliff
(Expected SARSA $-17.0$); SARSA and $n$-step Q show large negative means
and high variance because their sampled on-policy targets yield unstable
greedy policies across seeds.

**Scope.** These tabular results are illustrative, not a leaderboard.
They confirm that the axis distinctions of §IV manifest even without
neural approximation, and that statistical rigor changes which
differences are real: SARSA's CliffWalking CI spans $[-114, -51]$, so
any single-seed comparison would be misleading. The same discipline —
many seeds, interval estimates rather than point scores — is what §V
argues the historical Atari literature lacks.
