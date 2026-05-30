# IV.B. Sample Inefficiency {#sec-iv-b}

Addresses **W2** of the eight weaknesses introduced in §II.B. Methods
here modify *how transitions are stored, sampled, or augmented* to
extract more learning signal per environment interaction.

### A. The Weakness

Vanilla Q-learning consumes each transition exactly once. Experience
replay [@mnih_2013_atari] relaxed this constraint by storing transitions in a
buffer $\mathcal{D}$ and sampling minibatches for repeated use:

$$
\theta \leftarrow \theta - \alpha \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}\Bigl[\nabla_\theta \bigl(Q(s,a;\theta) - y\bigr)^2\Bigr],
\quad y = r + \gamma \max_{a'} Q(s', a'; \theta^-).
$$

Uniform sampling, however, treats every transition as equally
informative. Transitions where the agent's current prediction is
already accurate contribute little to the gradient yet are sampled
with the same frequency as transitions whose temporal-difference (TD)
error is large. In environments where high-information transitions
are rare — sparse-reward games, low-probability state encounters,
narrow-margin decision points — uniform replay leaves learning
bottlenecked on the rate at which those transitions reappear in the
sample.

Methods in this section modify the replay distribution, the buffer
contents, or both, to concentrate learning capacity where the gradient
is most useful.

### B. Solution Families

Four mechanisms recur across the literature, distinguished by *what
they manipulate*: which transitions get sampled, what additional
transitions get stored, how the buffer is summarized when memory is
limited, and how reward density itself can be increased.

**B.1. Prioritized sampling.** Prioritized Experience Replay (PER)
[@schaul_2016_per] samples transitions in proportion to their TD-error magnitude:

$$
P(i) = \frac{p_i^\beta}{\sum_k p_k^\beta},
\quad p_i = |\delta_i| + \epsilon,
$$

where $\delta_i$ is the TD error of transition $i$ and $\epsilon$ a
small constant ensuring nonzero probability. PER offers two variants:
proportional ($p_i = |\delta_i| + \epsilon$) and rank-based
($p_i = 1/\mathrm{rank}(i)$). Non-uniform sampling introduces bias,
which PER corrects via importance-sampling weights
$w_i = (1/(N \cdot P(i)))^\beta$, annealed toward 1 as training
progresses.

PER is one of the two largest contributors to Rainbow's [@hessel_2018_rainbow]
performance, alongside multi-step learning. Its ablation removal
produces the steepest performance drop of any Rainbow component.

**B.2. Demonstration augmentation.** Deep Q-learning from
Demonstrations (DQfD) [@hester_2018_dqfd] augments the replay buffer with expert
trajectories prior to environment interaction. During an initial
pre-training phase, the loss combines four terms:

$$
\mathcal{L}_\text{DQfD} = \mathcal{L}_{1\text{-step}} + \mathcal{L}_{n\text{-step}} + \mathcal{L}_\text{sup} + \mathcal{L}_{L^2},
$$

where $\mathcal{L}_\text{sup}$ is a large-margin classification loss
encouraging the network to assign higher $Q$-values to actions
demonstrated by the expert. After pre-training, the demonstration
transitions remain in the buffer alongside agent-collected
experience.

DQfD trades exploration cost for demonstration cost: it requires
expert trajectories but produces substantially better initial
policies, achieving state-of-the-art on 11 of 42 Atari games at the
time of publication.

**B.3. Memory-efficient consolidation.** Memory-Efficient DQN
(MeDQN) [@chen_2023_medqn] addresses a different bottleneck: replay buffer storage.
Standard DQN on Atari requires a 7GB buffer; MeDQN compresses this to
~0.7GB by introducing a consolidation loss that distills past Q-values
from a target network into the current network:

$$
\mathcal{L}_\text{V-consolid} = \mathbb{E}_{(s,a) \sim p(\cdot,\cdot)}\Bigl[(Q(s,a;\theta) - \hat Q(s,a;\theta^-))^2\Bigr],
$$

added to the standard DQN loss with weight $\lambda$. Two sampling
schemes for the consolidation distribution $p(s,a)$ exist:
MeDQN(U) samples from a uniform approximation of the state space;
MeDQN(R) samples from a small auxiliary replay buffer of past states.

MeDQN(R) matches or exceeds DQN performance on five selected Atari
games while reducing memory tenfold. Its consolidation loss also
serves a stability function (see §IV.H) — the same mechanism prevents
catastrophic forgetting and damps function-approximation drift.

**B.4. Goal relabeling.** Hindsight Experience Replay (HER) [Andrychowicz
et al. 2017] applies only to goal-conditioned MDPs but yields large
sample-efficiency gains where it does. After collecting a trajectory
$\tau = (s_0, a_0, r_0, \dots, s_T)$ under goal $g$, HER stores not
only $(s_t, a_t, r_t, s_{t+1}, g)$ but also relabeled transitions
$(s_t, a_t, r'_t, s_{t+1}, g')$ where $g' = s_T$ (or any visited
future state) and $r'_t$ is recomputed under $g'$. The trajectory
that failed under the original goal becomes a successful trajectory
under a synthetic goal, densifying the reward signal at no
environmental cost.

HER is the method that unlocked sample-efficient Q-learning for
sparse-reward robotic manipulation tasks and represents one of the
largest single sample-efficiency advances of the past decade.

### C. Trade-offs

- **Sampling bias vs. signal concentration.** PER's importance
  sampling correction is asymptotically unbiased but introduces
  variance, particularly early in training when $\beta < 1$. The
  bias-variance trade-off is controlled by the prioritization
  exponent $\alpha$ and the importance-sampling annealing schedule
  for $\beta$.
- **Demonstration availability.** DQfD's advantage scales with
  demonstration quality and coverage. In domains where expert
  policies are unavailable, expensive to obtain, or
  reward-misaligned, the framework cannot be applied.
- **Memory vs. compute.** MeDQN trades buffer storage for the
  compute cost of the consolidation loss. The trade-off is favorable
  on Atari-scale tasks (7GB → 0.7GB at modest compute increase) but
  the consolidation loss adds a hyperparameter $\lambda$ requiring
  tuning per domain.
- **Goal-conditioning requirement.** HER applies only to MDPs whose
  reward function decomposes naturally over goals. Reward sparsity
  in non-goal-conditioned settings is not addressed.

### D. Empirical Evidence

PER outperforms uniform-replay DQN on 41 of 49 Atari games in the
original evaluation [@schaul_2016_per], with the largest gains on games where
high-error transitions are rare. Improvements concentrate on
strategic-planning games (Q*bert, Ms. Pac-Man) rather than
reaction-time games (Breakout, Space Invaders), consistent with the
mechanism: dense-reward games already sample informative transitions
frequently enough.

DQfD achieves state-of-the-art performance on 11 of 42 Atari games
at publication, with the largest absolute gains on
demonstration-amenable games (Q*bert, River Raid, Hero —
see Tables II–III). Notably, DQfD reaches 42,457 on Private Eye —
the largest single result by any method in Table III on that game,
unmatched by any subsequent non-demonstration method. This is
consistent with the framing of §IV.C: Private Eye's bottleneck is
exploration, and demonstration substitutes for it.

MeDQN's empirical evidence on Atari is restricted to five games but
matches DQN performance at roughly 10% of the memory budget. The
trade-off is favorable for any setting where replay memory is a
binding constraint, including distributed training (see §IV.G).

HER's Atari results do not appear in Tables II–III because the
Arcade Learning Environment is not goal-conditioned. The relevant
empirical evidence is on robotic manipulation benchmarks (OpenAI
Fetch, Push, Slide, Pick-and-Place), where HER agents achieve
near-100% success on tasks that vanilla DQN cannot learn from a
single demonstration.

### E. Open Questions

Three questions on this axis remain open:

1. **Replacement policy.** First-in-first-out (FIFO) is the default
   for finite replay buffers but is rarely justified empirically.
   When the buffer fills, FIFO discards the oldest transitions —
   which include the rare early-training successes that contain the
   highest information density. Adaptive replacement policies that
   retain transitions by learning utility, novelty, or persistent
   high TD error are largely unexplored. A simple adaptive policy
   could plausibly close a substantial fraction of the gap to
   infinite-buffer methods.

2. **Transfer of prioritized sampling to offline RL.** PER's
   correction relies on importance-sampling weights computed against
   the *current* replay distribution. In offline RL (§IV.E), the
   replay distribution is fixed and the gradient direction is
   constrained by support; whether non-uniform sampling helps or
   hurts in that regime is not settled, and per-method offline
   results have been mixed.

3. **Goal relabeling beyond Cartesian goals.** HER's relabeling
   trick depends on the reward function being computable from
   $(s, a, g)$ alone. For reward functions involving full trajectory
   features (sparse safety constraints, multi-step economic
   reasoning), the relabeling space is not obviously parameterizable.
   General-purpose goal relabeling for non-decomposable rewards
   would substantially expand the method's applicability.

### F. Comparison summary

| Method (year) | Legacy category | Mechanism | Primary cost | Best at | Empirical anchor |
|---|---|---|---|---|---|
| DQN replay buffer (2013) | Memory/Replay | Uniform sampling from buffer | Memory | Off-policy stability baseline | Atari first DQN |
| PER (2016) | Memory/Replay | Sample $\propto |\delta_i|^\alpha$, IS-corrected | IS bias, prioritization $\alpha$ | Rare high-info transitions | 41/49 Atari > DQN |
| DQfD (2018) | Memory/Replay | Augment buffer with expert demonstrations | Demonstration availability | Sparse-reward Atari | Private Eye: 42,457 |
| MeDQN (2023) | Memory/Replay | Consolidation loss compresses buffer | $\lambda$ hyperparameter | Memory-constrained training | Atari 7GB → 0.7GB |
| HER (2017) | Memory/Replay | Goal relabeling for synthetic reward | Goal-conditioned only | Robotic manipulation | OpenAI Fetch $\approx 100$\% |

The methods on this axis do not occupy a clean 2D trade-off; the
relevant trade-offs are categorical (memory vs. compute vs.
demonstration availability vs. environment structure). The
comparison table alone suffices.
