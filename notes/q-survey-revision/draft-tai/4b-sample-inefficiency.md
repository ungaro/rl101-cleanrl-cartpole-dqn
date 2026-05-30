## IV.B. Sample Inefficiency {#sec-iv-b}

**Weakness.** Vanilla Q-learning consumes each transition once; experience
replay [@mnih_2013_atari] relaxes this by sampling minibatches from a buffer
$\mathcal{D}$, $\theta \leftarrow \theta - \alpha\,\mathbb{E}_{(s,a,r,s')\sim\mathcal{D}}[\nabla_\theta(Q(s,a;\theta)-y)^2]$.
But uniform sampling treats every transition as equally informative, so when
high-information transitions are rare — sparse rewards, low-probability states,
narrow-margin decisions — learning stays bottlenecked on the rate at which
those transitions reappear in the sample.

**Mechanisms.** Four families recur, distinguished by *what they manipulate*.
(i) *Prioritized sampling*: Prioritized Experience Replay (PER)
[@schaul_2016_per] draws transitions in proportion to TD-error magnitude,
$P(i)\propto(|\delta_i|+\epsilon)^\alpha$, with importance-sampling weights
annealed toward 1 to correct the induced bias; it is among the two largest
contributors to Rainbow [@hessel_2018_rainbow]. (ii) *Demonstration
augmentation*: Deep Q-learning from Demonstrations (DQfD) [@hester_2018_dqfd]
seeds the buffer with expert trajectories and adds a large-margin supervised
term, trading exploration cost for demonstration cost to produce far stronger
initial policies. (iii) *Memory-efficient consolidation*: MeDQN
[@chen_2023_medqn] replaces most of the buffer with a consolidation loss that
distills past Q-values from the target network, cutting Atari storage roughly
tenfold; the same mechanism damps function-approximation drift and so doubles
as a stability tool ([§IV.H](#sec-iv-h)). (iv) *Goal relabeling*: Hindsight
Experience Replay (HER) [@andrychowicz_2017_her] rewrites failed
goal-conditioned trajectories as successes under synthetic goals $g'=s_T$,
densifying reward at no environmental cost.

**Trade-off.** No method dominates, because the relevant axes are categorical
rather than a clean bias–variance frontier. PER's prioritization is
asymptotically unbiased but injects variance early in training, tuned through
$\alpha$ and the $\beta$ schedule. DQfD's gain scales with demonstration
quality and is inapplicable when expert data is absent or reward-misaligned.
MeDQN trades buffer storage for the compute of its consolidation loss plus a
weight $\lambda$ to tune per domain. HER applies only to MDPs whose reward
decomposes over goals, leaving non-goal-conditioned sparsity untouched. The
choice therefore turns on environment structure and resource budget, not a
single dominant operating point. Crucially, these gains are tied to the
*online* replay distribution: PER's correction is computed against the current
buffer, so its transfer to the fixed distribution of offline RL
([§IV.E](#sec-iv-e)) is unsettled and empirically mixed.

**Open questions.** FIFO replacement is the unjustified default and discards
the rare early successes of highest information density, leaving adaptive,
utility-aware replacement largely unexplored. And goal relabeling beyond
Cartesian goals — rewards computable only from full trajectory features —
lacks a parameterizable relabeling space.

| Method (year) | Mechanism | Cost | Best at |
|---|---|---|---|
| DQN replay (2013) | Uniform sampling from buffer | Memory | Off-policy baseline |
| PER (2016) | Sample $\propto|\delta_i|^\alpha$, IS-corrected | IS bias, $\alpha$ tuning | Rare high-info transitions |
| DQfD (2018) | Buffer seeded with expert demos | Demonstration availability | Sparse-reward Atari |
| MeDQN (2023) | Consolidation loss compresses buffer | $\lambda$ hyperparameter | Memory-constrained training |
| HER (2017) | Goal relabeling for synthetic reward | Goal-conditioned only | Robotic manipulation |

: Sample-efficiency methods.
