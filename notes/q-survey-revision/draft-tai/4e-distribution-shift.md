## IV.E. Distribution Shift (Offline Reinforcement Learning) {#sec-iv-e}

**Weakness.** Q-learning's off-policy property is theoretical: the Bellman
operator is independent of the data-generating policy, but in *offline* RL the
buffer is fixed and the bootstrap target $r + \gamma \max_{a'} Q(s', a')$ ranges
over the entire action space, including out-of-distribution (OOD) actions no
behavior policy $\pi_b$ ever took. The approximator extrapolates arbitrarily at
such points, and iterated backups drive $Q$ unboundedly upward where data
support vanishes — so naive offline Q-learning consistently underperforms even
the behavior policy [@fujimoto_2019_bcq].

**Mechanisms.** Four families respond, distinguished by *what they constrain*.
(i) *Policy constraint* restricts the learned policy toward $\pi_b$: BCQ
[@fujimoto_2019_bcq] samples actions from a learned behavior model plus small
perturbations; BRAC [@wu_2019_brac] adds an explicit KL/Wasserstein divergence
penalty; and AWAC [@nair_2020_awac] uses an advantage-weighted update
$\pi(a\mid s)\propto\pi_b(a\mid s)\exp(A(s,a)/\beta)$ that bridges to online
fine-tuning. (ii) *Value penalty* — CQL [@kumar_2020_cql] adds a regularizer that
pushes $Q$ down at OOD actions and up on in-distribution ones, yielding a
lower bound on $V^\pi$. (iii) *Avoiding the max* — IQL [@kostrikov_2021_iql] drops
the $\max$ entirely, learning a state value via expectile regression and backing
up $Q(s,a)\leftarrow r+\gamma V(s')$, so extrapolation error is structurally
prevented. (iv) *Ensemble diversification* — EDAC [@an_2021_edac] takes a
min over $K$ critics trained to disagree on OOD actions; Cal-QL
[@nakamoto_2023_calql] calibrates CQL's bounds for the offline-to-online bridge.
A newer line couples expressive generative policies to Q-learning: FQL
[@park_2025_fql] trains a one-step flow-matching policy, sidestepping
backpropagation through a diffusion chain, while adjoint-matching variants
tolerate full multi-step generation and recover the optimal behavior-regularized
policy at convergence.

**Trade-off.** No method dominates, and the axis trades cleanly against itself.
Policy-constraint methods are safe but cannot outperform the best in-support
trajectory by much; CQL's penalty weight $\alpha$ is its single most sensitive
hyperparameter and varies across tasks despite adaptive scaling
[@hong_2023_adaptcql]; IQL avoids extrapolation but weakens formal optimality
guarantees; EDAC's $K$-network ensemble multiplies compute and needs per-action
gradients; flow policies trade expressiveness against tractability. The deeper
connection is to [§IV.A](#sec-iv-a): offline OOD action selection *is*
overestimation bias under a fixed buffer — the online $\max$-over-noisy-estimates
problem re-appearing where the correction signal of fresh on-policy samples is
absent. The online debiasing tools transfer imperfectly, which is precisely why
the offline setting demands its own conservatism (CQL, IQL) rather than the
decoupling and ensembles that suffice online.

**Open questions.** A unifying framework recovering all four families as limits
of a single regularization remains absent, and whether Q-learning's mechanism
suits internet-scale heterogeneous data — where sequence-modeling approaches
currently dominate — is empirically open with substantial practical stakes.

| Method (year) | Mechanism | Cost | Best at |
|---|---|---|---|
| BCQ (2019) | Generative behavior model + perturbation | Generator quality | Narrow-support data |
| BRAC (2019) | KL/Wasserstein divergence penalty | Divergence weight tuning | Diverse offline data |
| AWAC (2020) | Advantage-weighted offline → online | $\beta$ tuning | Offline-to-online bridge |
| CQL (2020) | Penalize $Q$ at OOD actions | $\alpha$ sensitivity | Random/medium data |
| IQL (2021) | Expectile $V$, $Q$ backed up via $V(s')$ | $\tau$ tuning | Single-network simplicity |
| EDAC (2021) | Min over $K$-ensemble + gradient diversity | $K\times$ compute | Locomotion strongest |
| Cal-QL (2023) | Calibrated conservative bounds | $\alpha$ tuning | Offline-to-online bridge |
| FQL (2025) | One-step flow-matching policy + Q | Flow training | Expressive offline policies |

: Offline Q-learning methods.
