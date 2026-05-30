## IV.H. Function-Approximation Instability {#sec-iv-h}

**Weakness.** The *deadly triad* [@sutton_2018_book] names the conjunction
of off-policy learning, bootstrapping, and function approximation: any two
are stable together, but all three admit counterexamples in which the
semi-gradient update
$$
\theta \leftarrow \theta + \alpha\bigl(r + \gamma \max_{a'} Q(s', a'; \theta) - Q(s, a; \theta)\bigr) \nabla_\theta Q(s, a; \theta)
$$
diverges or oscillates. The instability arises because each gradient step
also moves $Q$ at the bootstrap target $Q(s',a')$, making the update
self-referential and potentially expansive — deep Q-learning trained
naively does diverge.

**Mechanisms.** Four families damp the triad, distinguished by *what they
exploit*. (i) *Target decoupling* freezes a copy $Q(\cdot;\theta^-)$ for a
window so that $y = r + \gamma \max_{a'} Q(s', a'; \theta^-)$ no longer
chases its own parameters [@mnih_2015_nature]; this is the single
load-bearing stabilizer of the classical recipe, whether updated by hard
copy or Polyak averaging. (ii) *Architectural decomposition* — Dueling
[@wang_2016_dueling] — routes most of the gradient through a state-value
baseline, reducing noise in the action-conditional term; we re-interpret
its contribution as primarily stability, consistent with the Rainbow
ablation showing it the smallest-impact component
[@hessel_2018_rainbow] (its incidental bias effect appears in
[§IV.A](#sec-iv-a)). (iii) *Normalization* damps the expansive map
directly: PQN [@gallici_2024_pqn] applies LayerNorm [@ba_2016_layernorm]
throughout the body, input BatchNorm, optional L2, and $n$-step returns,
while Munchausen DQN adds a $\log\pi$ reward bonus that implicitly
regularizes toward the previous iterate. (iv) *Consolidation* — MeDQN
[@chen_2023_medqn], kin to elastic weight consolidation
[@kirkpatrick_2017_ewc] — penalizes movement away from past Q-values to
counter the non-stationarity that drives drift.

**Trade-off.** No method dominates, because the recipes trade staleness,
sample efficiency, and complexity against one another. Hard target updates
buy stability at the cost of targets up to $C$ steps stale; soft updates
smooth that staleness but slow responsiveness; consolidation and Munchausen
bonuses add a hyperparameter ($\lambda$, $\tau$) that under- or over-damps.
The pivotal result is PQN's: target networks *and* experience replay — long
held essential — can be removed entirely, with LayerNorm plus parallel
vectorized environments matching PER [@schaul_2016_per], Double DQN
[@hasselt_2016_doubledqn], and Rainbow [@hessel_2018_rainbow] on Atari at up
to $50\times$ less wall-clock. The triad's instability is thus dampable by
normalization alone, the classical stabilizers being sufficient but not
necessary. The catch is that PQN substitutes parallel environments for
replay's sample reuse, so where interaction is expensive (real robotics) the
trade-off may still favor replay retention.

**Open questions.** A formal stability guarantee for normalized Q-learning
under the deadly triad — beyond informal Lipschitz arguments — remains
absent, as does a direct test isolating Dueling's decomposition on vanilla
DQN. Whether online normalization recipes transfer to offline regimes
([§IV.E](#sec-iv-e)), where extrapolation to unsupported actions is not
damped by target networks, is largely unexplored.

| Method (year) | Mechanism | Cost | Best at |
|---|---|---|---|
| Target Network (2015) | Frozen $Q$ copy for bootstrap target | Target staleness ($C$ steps) | Foundational stability |
| Dueling DQN (2016)* | $V(s)$ + centered $A(s,a)$ decomposition | Modest | Many-action states |
| Munchausen DQN (2020) | $\log\pi$ reward bonus, implicit KL | $\tau$ hyperparameter | Implicit regularization |
| MeDQN (2023) | Past-$Q$ consolidation loss | $\lambda$ hyperparameter | Catastrophic forgetting |
| PQN (2024) | LayerNorm + $n$-step + parallel envs; no target net | Compute-structure shift | Modern norm-only recipe |

: Stability methods.
