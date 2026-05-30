# S6. Tabular Experiment: Configuration and Reproduction {#supp-s6}

This section documents the full configuration behind the tabular study
of §VI so that its numbers can be reproduced exactly. The experiment is
implemented in a single self-contained NumPy/Gymnasium script
(`tabular_experiments.py`, provided with this supplement); it requires
no GPU, as tabular Q-learning is scalar table arithmetic with no tensor
workload.

**Protocol.** For each environment and each model-free learner we train
over 100 random seeds, then evaluate the greedy (argmax-$Q$) policy over
300 episodes per seed. We report the mean episodic return across seeds
with a 95\% bootstrap confidence interval (10,000 resamples of the
per-seed means). Exploration uses an $\varepsilon$-greedy schedule
decayed linearly from $\varepsilon=1.0$ to $\varepsilon=0.05$ over the
first 80\% of training and held at $0.05$ thereafter. The $n$-step
learner uses $n=3$.

| Environment | Episodes | $\alpha$ | $\gamma$ | States × Actions |
|---|---|---|---|---|
| FrozenLake-v1 (4×4, slippery) | 15,000 | 0.10 | 0.99 | 16 × 4 |
| Taxi-v3 | 12,000 | 0.10 | 0.99 | 500 × 6 |
| CliffWalking | 2,500 | 0.50 | 0.99 | 48 × 4 |

: Per-environment training configuration. CliffWalking has no built-in
episode limit, so episodes are capped at 500 steps (truncation, with
bootstrapping continued) to bound evaluation of any non-terminating
policy.

**Planning oracles.** The Value/Policy Iteration upper bound is computed
from the environment's exact transition model $P$ and reward $R$
(exposed by Gymnasium for these tabular environments) by value iteration
to a residual tolerance of $10^{-10}$, with terminal-state values held
at zero. Because these methods require full model access, they are
reported only as a planning upper bound, never as a model-free
competitor.

**Full results.** The table below reproduces §VI's results with standard
deviations across the 100 seeds added alongside the 95\% bootstrap CIs.

| Algorithm | FrozenLake-v1 (success rate) | Taxi-v3 (return) | CliffWalking (return) |
|---|---|---|---|
| Q-learning | 0.722 ± 0.090 [0.701, 0.737] | 7.93 ± 0.16 [7.90, 7.96] | $-13.0$ ± 0.0 [$-13, -13$] |
| SARSA | 0.692 ± 0.109 [0.669, 0.712] | 7.90 ± 0.16 [7.87, 7.93] | $-79.9$ ± 162 [$-114, -51$] |
| Expected SARSA | 0.720 ± 0.068 [0.705, 0.732] | 7.93 ± 0.16 [7.90, 7.96] | $-17.0$ ± 0.0 [$-17, -17$] |
| $n$-step Q ($n{=}3$) | 0.497 ± 0.228 [0.451, 0.541] | 7.93 ± 0.16 [7.89, 7.96] | $-75.2$ ± 157 [$-104, -46$] |
| \textit{Planning oracle (VI/PI)} | \textit{0.739} | \textit{7.94} | \textit{$-13.0$} |

: Full tabular results: mean ± standard deviation over 100 seeds, with
95\% bootstrap confidence intervals in brackets. The large standard
deviations for SARSA and $n$-step Q on CliffWalking reflect genuine
across-seed instability of their greedy policies, not measurement noise.

**Reproduction.** Running the script with the default arguments
(`--seeds 100`) regenerates the JSON results file that this table and
§VI are built from; results are deterministic given the seed range. The
run parallelizes trivially across CPU cores (one seed per worker).
