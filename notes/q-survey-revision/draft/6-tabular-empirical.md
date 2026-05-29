# VI. Empirical Evaluation of Tabular Q-Learning Variants

To complement the literature analysis of §IV and the Atari extraction
of §V, we conduct controlled experiments in tabular environments
where true dynamics are known and function approximation is
unnecessary. This setup serves a specific methodological purpose:
**isolating algorithmic design from architectural confound.** Many
deep-RL claims about overestimation (W1), exploration (W3), or
stability (W8) are conflated in evaluation by the presence of
neural-network function approximation, replay buffers, and target
networks. Tabular evaluation removes these confounds and exposes the
algorithmic core.

### A. Experimental setup

We implement each algorithm from scratch and evaluate on three
canonical environments from the Gymnasium suite:

- **FrozenLake-v1** — a stochastic gridworld with sparse terminal
  rewards. Tests exploration (W3) and overestimation under
  stochasticity (W1).
- **Taxi-v3** — a discrete planning task with structured rewards
  and a long horizon. Tests credit assignment (W4) and sample
  efficiency (W2).
- **CliffWalking-v1** — a deterministic gridworld with a
  high-penalty hazard along the shortest path. Tests the on-policy
  vs. off-policy trade-off (SARSA vs. Q-learning) and the
  conservatism induced by Expected SARSA.

Performance is measured by the average reward over the final 10
evaluation episodes, averaged across 5 random seeds. The reported
values are raw — no smoothing is applied.

Two algorithms from the §IV survey are excluded from tabular
evaluation. Bayesian Q-learning [49] is excluded because its
posterior maintenance is prohibitively slow in repeated runs; its
formal contribution is preserved in the §IV review. Neural Fitted Q
Iteration (NFQ) [53] is excluded because it requires function
approximation by construction.

### B. Results

Tables IV, V, and VI report final-episode performance per
environment, averaged across seeds.

[Tables IV, V, VI here.]

### C. Interpretation by axis

The tabular results admit a cleaner axis-attribution than Atari
results because the algorithmic mechanism is the dominant source of
performance variation.

**Foundational baselines (Q-Learning, SARSA, Expected SARSA).** All
three achieve near-optimal performance on FrozenLake (Q-Learning
0.96, SARSA 0.78, Expected SARSA 0.94). Q-Learning's slight edge over
SARSA reflects the off-policy advantage when the optimal policy is
deterministic; SARSA's deficit reflects its on-policy update being
biased toward the exploratory $\epsilon$-greedy policy. On Cliff-
Walking — where the on-policy/off-policy distinction is sharpest —
SARSA's conservative behavior (-21) outperforms Q-Learning's
optimistic one (-39), as expected: SARSA learns the policy actually
being executed, while Q-Learning learns a riskier optimal policy
that ε-greedy execution sometimes plunges off the cliff.

**Multi-Step Q-Learning (§IV.D).** Multi-step achieves 0.90 on
FrozenLake and -57 on Taxi, slightly below baseline. The result is
informative about the limits of $n$-step in pure tabular settings:
without function approximation, the variance penalty of larger $n$
is not offset by the credit-assignment benefit. The empirical
benefits of $n$-step learning observed in deep RL (e.g., as a
Rainbow component) emerge specifically from the interaction with
function approximation.

**Double Q-Learning (§IV.A).** Double Q-Learning performs comparably
to Q-Learning on FrozenLake (0.90) but degrades substantially on
Taxi (-105.5). The Taxi result is consistent with the §IV.A
trade-off discussion: Double Q-Learning trades over-estimation for
under-estimation, and on tasks where the optimal policy requires
optimism about long-horizon rewards (Taxi has a -1 step penalty
incentivizing greedy long-horizon action), under-estimation hurts.
This is exactly the regime that the bias-variance frontier discussion
of §IV.A.E flags as underexplored.

**Planning baselines (Value Iteration, Policy Iteration, MPI,
CVPI).** Value Iteration and Policy Iteration are not RL algorithms
in the strict sense — they require known dynamics — but serve as
optimality references. CVPI [48] consistently matches or exceeds the
RL methods, confirming that the gap between learned and optimal
policies is the cost paid for unknown dynamics. The comparison
quantifies the cost: on Taxi, CVPI scores 5.5 against Q-Learning's
-57.

**MCTS.** MCTS underperforms on Taxi (-6.2) and FrozenLake (0.00) in
the limited-budget regime evaluated. The result reinforces a point
visible throughout this paper: planning methods scale to small
state spaces in laboratory settings but degrade rapidly when the
search budget is constrained relative to environment complexity.

### D. Why tabular matters for the axis argument

The tabular results matter because they *test the axis
attributions of §IV* in a setting where confounds are minimized.
Double Q-Learning's tabular under-estimation pattern, visible on
Taxi, is the same mechanism that drives §IV.A's bias-variance
discussion in the deep setting — but in the tabular setting it can
be observed cleanly. The convergence of tabular and deep evidence on
the same mechanistic claims, when present, is one of the stronger
methodological supports for the problem-first organization.
