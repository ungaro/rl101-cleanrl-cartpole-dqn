## IV.D. Reward Sparsity and Credit Assignment {#sec-iv-d}

**Weakness.** The one-step Q-learning update propagates reward by exactly
one Bellman backup per environment step, so when reward arrives only at
the end of a long trajectory the signal must traverse the whole horizon
through repeated updates before it reaches early states. This is the
credit-assignment problem: apportioning fractional responsibility for a
terminal reward across upstream state-action pairs, which a scalar
expected-return target conveys slowly and with little structure.

**Mechanisms.** Two families attack the weakness by changing *what the
return signal carries*. (i) *Temporal-extent* methods widen the backup:
Multi-Step Q-Learning [@peng_1996_multistep] uses $n$-step bootstrapped
returns with eligibility traces, $y_t^{(n)} = r_t + \gamma r_{t+1} +
\dots + \gamma^{n-1} r_{t+n-1} + \gamma^n \max_{a'} Q(s_{t+n}, a')$, and
the TD($\lambda$) interpolation tunes bias against variance via
$\lambda$; in Rainbow [@hessel_2018_rainbow] three-step returns are the
second-largest performance contributor after prioritized replay. (ii)
*Information-content* methods replace the scalar
$\mathbb{E}[\text{return}]$ with the full return distribution $Z(s,a)$.
C51 [@bellemare_2017_distributional] models $Z$ as a categorical
distribution over 51 fixed atoms on a bounded support
$[V_\text{min},V_\text{max}]$ and minimizes a projected-KL target;
QR-DQN [@dabney_2018_qrdqn] inverts this with fixed probabilities and
*learned* quantile locations under a quantile Huber loss, removing the
projection and the support prior; IQN [@dabney_2018_iqn] samples
quantile fractions $\tau\sim\mathcal{U}(0,1)$ at runtime to learn a
continuous distribution, reaching Rainbow-comparable scores alone; and
FQF [@yang_2019_fqf] additionally *learns* the fractions per
state-action pair, attaining the highest median Atari at roughly 20%
more compute than IQN.

**Trade-off.** No method dominates. Larger $n$ or $\lambda$ propagates
reward faster but inflates variance and off-policy bias, so three-step
is the practical sweet spot for single-stream agents while distributed
throughput (Ape-X, R2D2) makes longer horizons viable. Across the
distributional family, resolution trades against compute along the
monotone-in-median diagonal C51 → QR-DQN → IQN → FQF, yet per-game
behavior is non-monotone. The quantile-regression methods also incur a
*quantile-crossing* pathology: training independent estimators per
$\tau_i$ without enforcing $\tau_i<\tau_j\Rightarrow\theta_i\le\theta_j$
yields non-monotone pseudo-distributions, worst early in training and on
asymmetric rewards; non-crossing variants smooth the curves at small
median gains. Interpretively, this paper reframes the distributional
family as a *credit-assignment* mechanism rather than the conventional
uncertainty grouping ([§IV.G](#sec-iv-g)): all these methods still *act*
on $\mathbb{E}[Z(s,a)]$, so the richer distribution helps *learning*,
not *acting*, and its strength on long-horizon tasks reflects better
backup propagation, not optimism over uncertainty. The flip side is that
richer signal cannot substitute for directed search — on the hardest
exploration games ([§IV.C](#sec-iv-c)) the family scores near zero,
sharpening the W3/W4 separation.

**Open questions.** Convergence guarantees for the quantile family under
function approximation remain substantially weaker than for C51's
projection step, and whether its compositional gain with prioritized
replay is orthogonal or partly redundant is unsettled given IQN's
standalone strength. Whether these architectures support stable
risk-sensitive action selection on tail behavior, rather than acting on
the mean, is largely unexplored.

| Method (year) | Mechanism | Cost | Best at |
|---|---|---|---|
| Multi-Step Q (1996) | $n$-step returns + eligibility traces | Variance under off-policy | Long-horizon credit assignment |
| C51 (2017) | 51 fixed atoms, projected KL | Bounded support pre-specified | Atari mean + median |
| QR-DQN (2018) | $N$ learned quantile locations | Quantile loss, no projection | Dense + strategic Atari |
| IQN (2018) | Runtime-sampled $\tau\sim U(0,1)$ | Cosine fraction embedding | Standalone Rainbow-comparable |
| FQF (2019) | Learned quantile fractions + values | +20% compute vs. IQN | Highest median Atari |

: Credit-assignment and distributional methods.
